import json
import re
import os
import warnings
from typing import List, Dict, Any, Union, Set
from dataclasses import dataclass, field
from collections import Counter

import numpy as np
from kiwipiepy import Kiwi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore", category=ResourceWarning)

# 임베딩 모델 및 Kiwi 초기화
embedding_model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
kiwi = Kiwi()


@dataclass
class HybridSearchResult:
    final_score: float
    avg_combined_score: float
    avg_lexical_overlap: float
    avg_cosine_sim_raw: float
    total_queries_count: int         # DB 전체 질문 수
    cat_filtered_count: int          # 1차 카테고리 필터링 통과 수
    must_have_filtered_count: int    # 2차 키워드 필터링 통과 수 (최종 평가 대상)
    query_details: List[Dict[str, Any]]


@dataclass
class KeywordStuffingResult:
    final_score: float
    raw_score: float
    is_stuffing: bool
    noun_penalty: float
    grammatical_penalty: float
    pattern_penalty: float
    noun_ratio: float
    grammatical_ratio: float
    pattern_count: int


@dataclass
class JsonLdEvaluationResult:
    final_score: float
    raw_score: float
    is_valid: bool
    parsing_score: float
    density_score: float
    clothing_score: float
    trust_score: float
    present_attrs_count: int
    trust_count: int


@dataclass
class SingleAltEvaluationResult:
    final_score: float
    has_alt_attribute: bool
    is_stuffing: bool
    clean_alt: str
    s1_presence_score: float
    s2_relevance_score: float
    s3_sentence_score: float


@dataclass
class ImageAltEvaluationResult:
    avg_score: float
    raw_avg_score: float
    is_valid: bool
    total_image_count: int
    stuffing_image_count: int
    detail_results: List[SingleAltEvaluationResult] = field(default_factory=list)


@dataclass
class GEOTotalEvaluationResult:
    total_score: float
    text_ratio_score: float
    hybrid_search: HybridSearchResult
    keyword_stuffing: KeywordStuffingResult
    json_ld: JsonLdEvaluationResult
    image_alt: ImageAltEvaluationResult

    @property
    def formatted_score(self) -> str:
        return f"{int(self.total_score)}점/100점 만점"

    def print_summary(self):
        """평가 결과를 가독성 있게 요약 출력합니다."""
        print("=" * 60)
        print(f" 📊 GEO 평가 종합 결과 : {self.formatted_score}")
        print("=" * 60)
        
        # 1. 텍스트 비율
        print(f"1. 텍스트 비율 점수 : {self.text_ratio_score * 100:.1f} / 100 점")
        
        # 2. 하이브리드 검색
        hs = self.hybrid_search
        print(f"\n2. 하이브리드 검색 평가")
        print(f"    ├─ 최종 점수: {hs.final_score * 100:.1f}점")
        print(f"    ├─ 검색 통과 수: 전체 {hs.total_queries_count}개 중 {hs.must_have_filtered_count}개 통과 (1차 카테고리: {hs.cat_filtered_count}개)")
        print(f"    └─ 세부 지표: 어휘 중복도 {hs.avg_lexical_overlap:.2f} | 코사인 유사도 {hs.avg_cosine_sim_raw:.2f}")

        # 3. 키워드 스터핑
        ks = self.keyword_stuffing
        ks_status = "⚠️ 감점 누적됨" if ks.is_stuffing else "✅ 정상"
        print(f"\n3. 키워드 스터핑 검사")
        print(f"    ├─ 상태: {ks_status}")
        print(f"    └─ 세부 점수: 최종 {ks.final_score:.2f} (명사비율 감점: -{ks.noun_penalty}, 문법감점: -{ks.grammatical_penalty}, 패턴감점: -{ks.pattern_penalty})")

        # 4. JSON-LD 평가
        jld = self.json_ld
        jld_status = "✅ Valid" if jld.is_valid else "❌ Invalid"
        print(f"\n4. JSON-LD 구조화 데이터")
        print(f"    ├─ 상태: {jld_status} (최종 점수: {jld.final_score * 100:.1f}점)")
        print(f"    └─ 항목 점수: 파싱 {jld.parsing_score} | 밀도 {jld.density_score} | 속성 {jld.clothing_score} | 신뢰도 {jld.trust_score}")

        # 5. 이미지 Alt 태그
        alt = self.image_alt
        print(f"\n5. 이미지 Alt 태그 평가")
        print(f"    ├─ 최종 점수: {alt.avg_score * 100:.1f}점 (유효성 통과 여부: {alt.is_valid})")
        print(f"    └─ 이미지 현황: 총 {alt.total_image_count}개 중 스터핑 감지 {alt.stuffing_image_count}개")
        print("=" * 60)


class GEOScorer:
    def __init__(self):
        pass

    def _clean_text(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return ""

        # 1. Null 바이트 및 제어 문자 제거
        text = text.replace('\x00', '').replace('\u0000', '')
        
        # 2. 인코딩 안전성 확보
        text = text.encode('utf-8', 'ignore').decode('utf-8', 'ignore')
        
        # 3. 제어문자/특수유니코드 전처리
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)

        return text.strip()

    # -------------------------------------------------------------------------
    # Kiwi 헬퍼 메서드 (명사 / 전체 형태소 / POS 태깅)
    # -------------------------------------------------------------------------
    def _extract_nouns(self, text: str) -> List[str]:
        if not text:
            return []
        tokens = kiwi.tokenize(text)
        return [t.form for t in tokens if t.tag.startswith('N')]

    def _extract_morphs(self, text: str) -> List[str]:
        if not text:
            return []
        tokens = kiwi.tokenize(text)
        return [t.form for t in tokens]

    def _extract_pos(self, text: str) -> List[tuple]:
        if not text:
            return []
        tokens = kiwi.tokenize(text)
        return [(t.form, t.tag) for t in tokens]

    # =========================================================================
    # ① 본문 텍스트 비율 (기획안 완벽 반영)
    # =========================================================================
    def calculate_text_ratio(self, body_text: str, image_text: str) -> float:
        clean_body = self._clean_text(body_text)
        clean_image = self._clean_text(image_text)
        
        len_body = len(clean_body)
        len_image = len(clean_image)
        total = len_body + len_image
        
        if total == 0:
            return 0.0
        
        raw_ratio = len_body / total

        # 기획안 내용 1: Text Ratio 구간별 산출
        if raw_ratio < 0.1:
            ratio_score = 0.0
        elif raw_ratio >= 0.5:
            ratio_score = 1.0
        else:
            ratio_score = (raw_ratio - 0.1) / 0.3

        # 기획안 내용 2: Chunk 단위 글자 수 가중치 W(N) = min(1.0, N / 800)
        weight = min(1.0, len_body / 800.0)

        # Text Ratio Final Score = Text Ratio × W(N)
        final_score = ratio_score * weight
        return round(min(1.0, final_score), 4)

    # =========================================================================
    # ② 2단계 필터링 기반 하이브리드 검색
    # =========================================================================
    def calculate_hybrid_search(
        self, 
        user_queries: List[Dict[str, Any]], 
        product_cat: str, 
        product_name: str, 
        page_text: str
    ) -> HybridSearchResult:
        clean_page = self._clean_text(page_text)
        clean_prod_name = self._clean_text(product_name)
        
        if not clean_page or not user_queries:
            return HybridSearchResult(0.0, 0.0, 0.0, 0.0, len(user_queries) if user_queries else 0, 0, 0, [])

        total_queries = len(user_queries)
        
        # [1차 필터링] 카테고리 일치 여부
        cat_filtered_queries = [
            q for q in user_queries 
            if q.get("category") == product_cat
        ]
        cat_count = len(cat_filtered_queries)

        if not cat_filtered_queries:
            return HybridSearchResult(0.0, 0.0, 0.0, 0.0, total_queries, 0, 0, [])

        # [2차 필터링] must_have 키워드 중 1개라도 포함 여부
        final_queries = []
        for q in cat_filtered_queries:
            must_have_list = q.get("must_have", [])
            if any(mh in clean_prod_name for mh in must_have_list if mh):
                final_queries.append(q)
                
        must_have_count = len(final_queries)

        if not final_queries:
            return HybridSearchResult(0.0, 0.0, 0.0, 0.0, total_queries, cat_count, 0, [])

        # [3차] Kiwi 기반 형태소/명사 추출
        page_nouns = self._extract_nouns(clean_page)
        page_nouns_set = set(page_nouns) if page_nouns else set(self._extract_morphs(clean_page))
        
        # 1. 페이지 인코딩
        p_vec = embedding_model.encode([clean_page])

        # 2. 유효 쿼리 텍스트 정돈 및 배치 인코딩
        valid_query_objs = []
        query_strs = []
        for q_obj in final_queries:
            q_text = self._clean_text(q_obj.get("query_text", ""))
            if q_text:
                valid_query_objs.append(q_obj)
                query_strs.append(q_text)

        if not query_strs:
            return HybridSearchResult(0.0, 0.0, 0.0, 0.0, total_queries, cat_count, must_have_count, [])

        q_vecs = embedding_model.encode(query_strs, show_progress_bar=False)
        cos_sims = cosine_similarity(q_vecs, p_vec).flatten()

        query_details, lexical_overlap_list, cos_raw_list, combined_score_list = [], [], [], []

        for idx, q_obj in enumerate(valid_query_objs):
            query_str = query_strs[idx]

            query_tokens = self._extract_nouns(query_str) or self._extract_morphs(query_str)

            matched_count = sum(1 for token in query_tokens if token in page_nouns_set) if query_tokens else 0
            lexical_overlap = matched_count / len(query_tokens) if query_tokens else 0.0

            # 코사인 유사도 범위 클리핑 (0.0 ~ 1.0)
            cos_sim_raw = max(0.0, min(1.0, float(cos_sims[idx])))
            query_hybrid_score = min(1.0, (lexical_overlap * 0.5) + (cos_sim_raw * 0.5))

            lexical_overlap_list.append(lexical_overlap)
            cos_raw_list.append(cos_sim_raw)
            combined_score_list.append(query_hybrid_score)

            query_details.append({
                "query": query_str,
                "category": q_obj.get("category"),
                "must_have": q_obj.get("must_have"),
                "lexical_overlap": round(lexical_overlap, 4),
                "cosine_sim_raw": round(cos_sim_raw, 4),
                "query_hybrid_score": round(query_hybrid_score, 4)
            })

        if not combined_score_list:
            return HybridSearchResult(0.0, 0.0, 0.0, 0.0, total_queries, cat_count, must_have_count, [])

        n = len(combined_score_list)
        avg_combined = min(1.0, sum(combined_score_list) / n)

        return HybridSearchResult(
            final_score=round(avg_combined, 4),
            avg_combined_score=round(avg_combined, 4),
            avg_lexical_overlap=round(sum(lexical_overlap_list) / n, 4),
            avg_cosine_sim_raw=round(sum(cos_raw_list) / n, 4),
            total_queries_count=total_queries,
            cat_filtered_count=cat_count,
            must_have_filtered_count=must_have_count,
            query_details=query_details
        )

    # =========================================================================
    # ③ 본문 키워드 스터핑 (텍스트 없을 경우 0점 반환)
    # =========================================================================
    def calculate_keyword_stuffing(self, page_text: str) -> KeywordStuffingResult:
        clean_page = self._clean_text(page_text)
        # ⚠️ 보정: 본문이 없으면 만점(1.0)이 아니라 0점 반환
        if not clean_page:
            return KeywordStuffingResult(0.0, 0.0, False, 0.0, 0.0, 0.0, 0.0, 0.0, 0)

        pos_tags = self._extract_pos(clean_page)
        if not pos_tags:
            return KeywordStuffingResult(0.0, 0.0, False, 0.0, 0.0, 0.0, 0.0, 0.0, 0)

        total_tokens = len(pos_tags)
        nouns = [word for word, tag in pos_tags if tag.startswith('N')]
        particles = [word for word, tag in pos_tags if tag.startswith('J')]
        endings = [word for word, tag in pos_tags if tag.startswith('E')]

        noun_ratio = len(nouns) / total_tokens
        grammatical_ratio = (len(particles) + len(endings)) / total_tokens

        # 내용 1. 명사 비율 50% 초과 시 최대 0.33점 감점
        noun_penalty = min(0.33, ((noun_ratio - 0.5) / 0.5) * 0.33) if noun_ratio > 0.5 else 0.0
        
        # 내용 2. 조사/어미 비율 20% 미만 시 최대 0.34점 감점
        grammatical_penalty = min(0.34, ((0.2 - grammatical_ratio) / 0.2) * 0.34) if grammatical_ratio < 0.2 else 0.0

        # 내용 3. 키워드 스터핑 요소의 어미 반복 시 최대 0.33점 감점 (~룩, ~핏 등)
        pattern_count = sum(1 for n in nouns if n.endswith('룩') or n.endswith('핏'))
        pattern_penalty = min(0.33, ((pattern_count - 1) / 4) * 0.33) if pattern_count >= 2 else 0.0

        # 연속적 비례 감점 적용 (하드 threshold 없음)
        raw_score = max(0.0, 1.0 - (noun_penalty + grammatical_penalty + pattern_penalty))
        is_stuffing = raw_score < 0.6  # 모니터링용 플래그

        return KeywordStuffingResult(
            final_score=round(raw_score, 3),
            raw_score=round(raw_score, 3),
            is_stuffing=is_stuffing,
            noun_penalty=round(noun_penalty, 3),
            grammatical_penalty=round(grammatical_penalty, 3),
            pattern_penalty=round(pattern_penalty, 3),
            noun_ratio=round(noun_ratio, 2),
            grammatical_ratio=round(grammatical_ratio, 2),
            pattern_count=pattern_count
        )

    def _flatten_json_items(self, items: Any) -> List[dict]:
        flat = []
        if isinstance(items, list):
            for item in items:
                flat.extend(self._flatten_json_items(item))
        elif isinstance(items, dict):
            flat.append(items)
        return flat

    def _extract_json_ld_attributes(self, data: Any) -> Set[str]:
        """offers 및 hasVariant 하위 노드까지 깊이 탐색하여 필수 속성을 수집합니다."""
        target_attrs = {"material", "color", "size", "category"}
        found_attrs = set()

        def _search(d):
            if not isinstance(d, dict):
                return
            for k, v in d.items():
                if k in target_attrs and v:
                    found_attrs.add(k)
                if isinstance(v, dict):
                    _search(v)
                elif isinstance(v, list):
                    for item in v:
                        if isinstance(item, dict):
                            _search(item)

        _search(data)
        return found_attrs

    # =========================================================================
    # ④ JSON-LD 구조화 데이터 평가 (기획안 완벽 반영)
    # =========================================================================
    def calculate_json_ld_score(self, json_ld_input: Union[str, list, dict]) -> JsonLdEvaluationResult:
        default_fail = JsonLdEvaluationResult(0.0, 0.0, False, 0.0, 0.0, 0.0, 0.0, 0, 0)

        if not json_ld_input:
            return default_fail

        raw_data = None
        if isinstance(json_ld_input, str):
            clean_str = json_ld_input.strip()
            if not clean_str:
                return default_fail
            try:
                raw_data = json.loads(clean_str)
            except Exception:
                return default_fail
        else:
            raw_data = json_ld_input

        flat_dicts = self._flatten_json_items(raw_data)

        target_product = None
        has_graph = False

        for item in flat_dicts:
            if item.get("@type") == "Product":
                target_product = item
                break
            elif "@graph" in item:
                has_graph = True
                graph_items = self._flatten_json_items(item.get("@graph", []))
                for g_item in graph_items:
                    if g_item.get("@type") == "Product":
                        target_product = g_item
                        break
                if target_product:
                    break

        has_product = target_product is not None

        # 1. JSON-LD 파싱 및 유효성 점수 산출
        if has_product and has_graph:
            parsing_score = 1.0
        elif has_product:
            parsing_score = 0.7
        elif has_graph:
            parsing_score = 0.3
        else:
            return default_fail

        data = target_product if target_product else {}

        # 2. 텍스트 정보 밀도 점수 산출 (글자 수 기준)
        desc = data.get("description", "") if isinstance(data, dict) else ""
        desc_clean = re.sub(r"[^\w\s]", "", str(desc)).strip()
        desc_len = len(desc_clean)

        if desc_len >= 100:
            density_score = 1.0
        elif desc_len >= 50:
            density_score = 0.7
        elif desc_len >= 1:
            density_score = 0.35
        else:
            density_score = 0.0

        # 3. 의류 커머스 특화 속성 충족도 (속성 수 / 4)
        found_attrs = self._extract_json_ld_attributes(data)
        present_attrs = len(found_attrs)
        clothing_score = min(1.0, present_attrs / 4.0)

        # 4. 신뢰도 및 엔티티 연결성 (속성 수 / 3)
        trust_count = 0
        if isinstance(data, dict):
            # @graph 또는 sameAs
            if "sameAs" in data or has_graph:
                trust_count += 1
            if "shippingDetails" in data:
                trust_count += 1
            if "hasMerchantReturnPolicy" in data:
                trust_count += 1
        trust_score = min(1.0, trust_count / 3.0)

        # 4개 항목의 산술 평균으로 final_score 산출
        raw_score = round(
            min(1.0, (parsing_score + density_score + clothing_score + trust_score) / 4.0), 4
        )

        return JsonLdEvaluationResult(
            final_score=raw_score,
            raw_score=raw_score,
            is_valid=has_product,
            parsing_score=round(parsing_score, 2),
            density_score=round(density_score, 2),
            clothing_score=round(clothing_score, 2),
            trust_score=round(trust_score, 2),
            present_attrs_count=present_attrs,
            trust_count=trust_count,
        )

    # =========================================================================
    # ⑤ 이미지 alt 속성 평가 지표 (개선안 반영)
    # =========================================================================
    def evaluate_single_alt(self, alt_text: str) -> SingleAltEvaluationResult:
        clean_alt = self._clean_text(alt_text) if alt_text else ""

        # ⚠️ 보정: alt 속성이 없거나 빈 값이면(has_alt가 False면) 즉시 0점 반환
        if not clean_alt:
            return SingleAltEvaluationResult(
                final_score=0.0,
                has_alt_attribute=False,
                is_stuffing=False,
                clean_alt="",
                s1_presence_score=0.0,
                s2_relevance_score=0.0,
                s3_sentence_score=0.0
            )

        # 1. 속성 존재 여부 점수 (존재하므로 0.20점 부여)
        s1 = 0.20

        # 2. alt 값의 적절성 평가 (0.30점)
        # 확장자명이나 무의미한 관리 코드(img_01, image_1 등) 포함 여부 검사
        invalid_patterns = r'(\.(jpg|jpeg|png|gif|webp|bmp)$)|(img_\d+|image_\d+|picture_\d+)'
        has_invalid_pattern = bool(re.search(invalid_patterns, clean_alt, re.IGNORECASE))
        s2 = 0.0 if has_invalid_pattern else 0.30

        # 3. GEO 관점의 문장 완성도 평가
        pos_tags = self._extract_pos(clean_alt)
        nouns = [w for w, p in pos_tags if p.startswith('N')]
        josa = [w for w, p in pos_tags if p.startswith('J')]
        verbs_adjs_modifiers = [w for w, p in pos_tags if p.startswith('V') or p.startswith('M')]

        # [1단계: 키워드 스터핑 탐지]
        is_stuffing = False

        # 조건 1) 명사가 4개 이상 나열
        cond1 = len(nouns) >= 4
        # 조건 2) 연결어미(EC) 없이 관형사형 어미(ETM) 부착 용언이 2회 이상 연속 나열
        etm_continuous_count = 0
        for i in range(len(pos_tags) - 1):
            curr_tag = pos_tags[i][1]
            next_tag = pos_tags[i+1][1]
            if curr_tag == 'ETM' and next_tag != 'EC':
                etm_continuous_count += 1
        cond2 = etm_continuous_count >= 2

        # 조건 3) 문장 전체 명사 비율 80% 이상 & 조사 0개
        total_tokens = len(pos_tags)
        noun_ratio = len(nouns) / total_tokens if total_tokens > 0 else 0.0
        cond3 = (noun_ratio >= 0.80) and (len(josa) == 0)

        if cond1 or cond2 or cond3:
            is_stuffing = True

        # 점수 산정
        if is_stuffing:
            s3 = 0.0  # 키워드 스터핑 해당 시 0점
        else:
            # 1단계 통과 시 기본 +0.20 부여 후 문장 구조 세부 평가
            has_josa = len(josa) > 0
            has_verb_or_modifier = len(verbs_adjs_modifiers) > 0

            if has_josa and has_verb_or_modifier:
                s3 = 0.50  # 조사, 용언/수식어 모두 존재
            elif has_verb_or_modifier and not has_josa:
                s3 = 0.40  # 용언/수식어 존재, 조사 미존재
            elif has_josa and not has_verb_or_modifier:
                s3 = 0.30  # 조사 존재, 용언/수식어 미존재
            else:
                s3 = 0.20  # 조사, 용언/수식어 모두 미존재

        final_score = round(min(1.0, s1 + s2 + s3), 3)

        return SingleAltEvaluationResult(
            final_score=final_score,
            has_alt_attribute=True,
            is_stuffing=is_stuffing,
            clean_alt=clean_alt,
            s1_presence_score=s1,
            s2_relevance_score=s2,
            s3_sentence_score=s3
        )

    def calculate_avg_alt_score(self, image_list: List[Dict[str, Any]]) -> ImageAltEvaluationResult:
        # ⚠️ 보정: 이미지가 아예 없으면 1.0(만점)이 아니라 0점 반환
        if not image_list:
            return ImageAltEvaluationResult(0.0, 0.0, False, 0, 0, [])

        details = [self.evaluate_single_alt(img.get('alt')) for img in image_list]
        raw_avg = round(sum(d.final_score for d in details) / len(details), 3)

        return ImageAltEvaluationResult(
            avg_score=raw_avg,
            raw_avg_score=raw_avg,
            is_valid=True,
            total_image_count=len(image_list),
            stuffing_image_count=sum(1 for d in details if d.is_stuffing),
            detail_results=details
        )

    # =========================================================================
    # Total 평가 함수 (5개 항목 가중치 산출)
    # =========================================================================
    def evaluate_page(
        self,
        body_text: str,
        image_text: str,
        user_queries: List[Dict[str, Any]],
        product_cat: str,
        product_name: str,
        json_ld_str: Union[str, list, dict] = "",
        image_list: List[Dict[str, Any]] = None
    ) -> GEOTotalEvaluationResult:

        if image_list is None:
            image_list = []
        page_text = f"{body_text} {image_text}".strip()

        s1 = self.calculate_text_ratio(body_text=body_text, image_text=image_text)
        s2 = self.calculate_hybrid_search(
            user_queries=user_queries, 
            product_cat=product_cat, 
            product_name=product_name, 
            page_text=page_text
        )
        s3 = self.calculate_keyword_stuffing(page_text=page_text)
        s4 = self.calculate_json_ld_score(json_ld_input=json_ld_str)
        s5 = self.calculate_avg_alt_score(image_list=image_list)

        # 5개 개별 지표(각각 0.0~1.0)의 평균으로 100점 만점 계산
        total_score = round(((s1 + s2.final_score + s3.final_score + s4.final_score + s5.avg_score) / 5.0) * 100, 0)

        return GEOTotalEvaluationResult(
            total_score=total_score,
            text_ratio_score=round(s1, 2),
            hybrid_search=s2,
            keyword_stuffing=s3,
            json_ld=s4,
            image_alt=s5
        )