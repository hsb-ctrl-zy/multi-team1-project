import json
import re
import warnings
from typing import List, Dict, Any, Union
from dataclasses import dataclass, field

from konlpy.tag import Komoran
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore", category=ResourceWarning)

komoran = Komoran()
embedding_model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

@dataclass
class HybridSearchResult:
    final_score: float
    avg_combined_score: float
    avg_lexical_overlap: float
    avg_cosine_sim_raw: float
    total_queries_count: int       # DB 전체 질문 수
    cat_filtered_count: int        # 1차 카테고리 필터링 통과 수
    must_have_filtered_count: int  # 2차 키워드 필터링 통과 수 (최종 평가 대상)
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
    is_text_image: bool
    clean_alt: str
    is_stuffing: bool
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
        print(f"   ├─ 최종 점수: {hs.final_score * 100:.1f}점")
        print(f"   ├─ 검색 통과 수: 전체 {hs.total_queries_count}개 중 {hs.must_have_filtered_count}개 통과 (1차 카테고리: {hs.cat_filtered_count}개)")
        print(f"   └─ 세부 지표: 어휘 중복도 {hs.avg_lexical_overlap:.2f} | 코사인 유사도 {hs.avg_cosine_sim_raw:.2f}")

        # 3. 키워드 스터핑
        ks = self.keyword_stuffing
        ks_status = "⚠️ 감지됨 (0점 처리)" if ks.is_stuffing else "✅ 정상"
        print(f"\n3. 키워드 스터핑 검사")
        print(f"   ├─ 상태: {ks_status}")
        print(f"   └─ 세부 점수: 원점수 {ks.raw_score:.2f} (명사비율 감점: -{ks.noun_penalty}, 문법감점: -{ks.grammatical_penalty})")

        # 4. JSON-LD 평가
        jld = self.json_ld
        jld_status = "✅ Valid" if jld.is_valid else "❌ Invalid"
        print(f"\n4. JSON-LD 구조화 데이터")
        print(f"   ├─ 상태: {jld_status} (최종 점수: {jld.final_score * 100:.1f}점)")
        print(f"   └─ 항목 점수: 파싱 {jld.parsing_score} | 밀도 {jld.density_score} | 속성 {jld.clothing_score} | 신뢰도 {jld.trust_score}")

        # 5. 이미지 Alt 태그
        alt = self.image_alt
        print(f"\n5. 이미지 Alt 태그 평가")
        print(f"   ├─ 최종 점수: {alt.avg_score * 100:.1f}점 (유효성 통과 여부: {alt.is_valid})")
        print(f"   └─ 이미지 현황: 총 {alt.total_image_count}개 중 스터핑 의심 {alt.stuffing_image_count}개")
        print("=" * 60)

class GEOScorer:
    def __init__(self):
        pass

    def _clean_text(self, text: str) -> str:
            if not text:
                return ""
            
            # 1. str 타입 보장
            if not isinstance(text, str):
                text = str(text)

            # 2. 유효하지 않은 UTF-8 바이트 및 깨진 유니코드 문자 제거 (핵심)
            text = text.encode('utf-8', 'ignore').decode('utf-8', 'ignore')

            # 3. 제어 문자 및 개행/공백 정돈
            text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
            return re.sub(r'\s+', ' ', text).strip()

    def calculate_text_ratio(self, body_text: str, image_text: str) -> float:
        clean_body = self._clean_text(body_text)
        clean_image = self._clean_text(image_text)
        total = len(clean_body) + len(clean_image)
        if total == 0:
            return 0.0
        ratio = len(clean_body) / total
        return ratio if ratio >= 0.2 else 0.0

    # =========================================================================
    # ② 2단계 필터링 기반 하이브리드 검색 (카테고리 1차 -> must_have 2차 -> Hybrid Search)
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
        
        # 0. 기본 예외 처리
        if not clean_page or not user_queries:
            return HybridSearchResult(0.0, 0.0, 0.0, 0.0, len(user_queries), 0, 0, [])

        total_queries = len(user_queries)
        
        # ---------------------------------------------------------------------
        # [1차 필터링] DB 질문의 category와 상품의 product_cat 일치 여부
        # ---------------------------------------------------------------------
        cat_filtered_queries = [
            q for q in user_queries 
            if q.get("category") == product_cat
        ]
        cat_count = len(cat_filtered_queries)

        if not cat_filtered_queries:
            return HybridSearchResult(0.0, 0.0, 0.0, 0.0, total_queries, 0, 0, [])

        # ---------------------------------------------------------------------
        # [2차 필터링] must_have 단어 중 1개라도 product_name에 포함되어 있는지 검사
        # ---------------------------------------------------------------------
        final_queries = []
        for q in cat_filtered_queries:
            must_have_list = q.get("must_have", [])
            
            # must_have가 빈 리스트면 필터 통과 처리할지, 탈락시킬지 판단 필요.
            # (기본적으로 must_have 단어 중 단 하나라도 상품명에 들어있으면 채택)
            if any(mh in clean_prod_name for mh in must_have_list if mh):
                final_queries.append(q)
                
        must_have_count = len(final_queries)

        if not final_queries:
            return HybridSearchResult(0.0, 0.0, 0.0, 0.0, total_queries, cat_count, 0, [])

        # ---------------------------------------------------------------------
        # [3차 필터링 완료 대상] 남은 질문들에 대해 Hybrid Search 진행
        # ---------------------------------------------------------------------
        page_nouns = komoran.nouns(clean_page)
        page_nouns_set = set(page_nouns) if page_nouns else set(komoran.morphs(clean_page))
        p_vec = embedding_model.encode([clean_page])

        query_details, lexical_overlap_list, cos_raw_list, combined_score_list = [], [], [], []

        for q_obj in final_queries:
            query_str = self._clean_text(q_obj.get("query_text", ""))
            if not query_str:
                continue

            query_tokens = komoran.nouns(query_str) or komoran.morphs(query_str)
            matched_count = sum(1 for token in query_tokens if token in page_nouns_set) if query_tokens else 0
            lexical_overlap = matched_count / len(query_tokens) if query_tokens else 0.0

            q_vec = embedding_model.encode([query_str])
            cos_sim_raw = float(cosine_similarity(q_vec, p_vec)[0][0])
            
            query_hybrid_score = (lexical_overlap * 0.5) + (cos_sim_raw * 0.5)

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
        avg_combined = sum(combined_score_list) / n

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

    def calculate_keyword_stuffing(self, page_text: str) -> KeywordStuffingResult:
        clean_page = self._clean_text(page_text)
        if not clean_page:
            return KeywordStuffingResult(0.0, 0.0, True, 0.0, 0.0, 0.0, 0.0, 0.0, 0)

        pos_tags = komoran.pos(clean_page)
        if not pos_tags:
            return KeywordStuffingResult(0.0, 0.0, True, 0.0, 0.0, 0.0, 0.0, 0.0, 0)

        total_tokens = len(pos_tags)
        nouns = [word for word, tag in pos_tags if tag.startswith('N')]
        particles = [word for word, tag in pos_tags if tag.startswith('J')]
        endings = [word for word, tag in pos_tags if tag.startswith('E')]

        noun_ratio = len(nouns) / total_tokens
        grammatical_ratio = (len(particles) + len(endings)) / total_tokens

        noun_penalty = min(0.333, ((noun_ratio - 0.5) / 0.5) * 0.333) if noun_ratio > 0.5 else 0.0
        grammatical_penalty = min(0.333, ((0.2 - grammatical_ratio) / 0.2) * 0.333) if grammatical_ratio < 0.2 else 0.0

        pattern_count = sum(1 for n in nouns if n.endswith('룩') or n.endswith('핏'))
        pattern_penalty = min(0.333, ((pattern_count - 1) / 4) * 0.333) if pattern_count >= 2 else 0.0

        raw_score = max(0.0, 1.0 - (noun_penalty + grammatical_penalty + pattern_penalty))
        is_stuffing = raw_score < 0.6

        return KeywordStuffingResult(
            final_score=raw_score if not is_stuffing else 0.0,
            raw_score=raw_score,
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

    def calculate_json_ld_score(self, json_ld_input: Union[str, list, dict]) -> JsonLdEvaluationResult:
        default_fail = JsonLdEvaluationResult(0.0, 0.0, False, 0.0, 0.0, 0.0, 0.0, 0, 0)

        if not json_ld_input:
            print("[JSON-LD] 경고: 입력값이 비어있습니다.")
            return default_fail

        raw_data = None
        if isinstance(json_ld_input, str):
            clean_str = json_ld_input.strip()
            if not clean_str:
                print("[JSON-LD] 경고: 빈 문자열입니다.")
                return default_fail
            try:
                raw_data = json.loads(clean_str)
            except Exception as e:
                print(f"[JSON-LD] JSON 파싱 실패: {e}")
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

        if has_product and has_graph:
            parsing_score = 1.0
        elif has_product:
            parsing_score = 0.7
        elif has_graph:
            parsing_score = 0.3
        else:
            print("[JSON-LD] 'Product' 타입 또는 '@graph' 구조를 찾지 못했습니다.")
            return default_fail

        data = target_product

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

        attr_keys = ["material", "color", "size", "category"]
        present_attrs = sum(1 for k in attr_keys if isinstance(data, dict) and k in data)
        clothing_score = present_attrs / 4.0

        trust_count = 0
        if isinstance(data, dict):
            if has_graph or "sameAs" in data:
                trust_count += 1
            if "shippingDetails" in data:
                trust_count += 1
            if "hasMerchantReturnPolicy" in data:
                trust_count += 1
        trust_score = trust_count / 3.0

        raw_score = round(
            (parsing_score * 0.25) + (density_score * 0.25) + (clothing_score * 0.25) + (trust_score * 0.25), 4
        )
        
        final_score = raw_score
        is_valid = has_product

        return JsonLdEvaluationResult(
            final_score=final_score,
            raw_score=raw_score,
            is_valid=is_valid,
            parsing_score=round(parsing_score, 2),
            density_score=round(density_score, 2),
            clothing_score=round(clothing_score, 2),
            trust_score=round(trust_score, 2),
            present_attrs_count=present_attrs,
            trust_count=trust_count,
        )

    def evaluate_single_alt(self, alt_text: str, is_text_image: bool) -> SingleAltEvaluationResult:
        if alt_text is None:
            return SingleAltEvaluationResult(0.0, False, is_text_image, "", False, 0.0, 0.0, 0.0)

        s1 = 0.25
        clean_alt = self._clean_text(alt_text)
        s2 = (0.25 if clean_alt == "" else 0.0) if not is_text_image else (0.25 if len(clean_alt) >= 10 else 0.0)

        if clean_alt == "":
            return SingleAltEvaluationResult(round(s1 + s2, 3), True, is_text_image, "", False, s1, s2, 0.0)

        pos_tags = komoran.pos(clean_alt)
        total_tokens = len(pos_tags)

        nouns = [w for w, p in pos_tags if p.startswith('N')]
        josa = [w for w, p in pos_tags if p.startswith('J')]
        verbs_adjs_modifiers = [w for w, p in pos_tags if p.startswith('V') or p.startswith('M')]

        is_stuffing = False
        if len(nouns) >= 4:
            is_stuffing = True

        if not is_stuffing:
            etm_without_ec_count = sum(
                1 for i in range(len(pos_tags) - 1)
                if pos_tags[i][1] == 'ETM' and pos_tags[i+1][1] != 'EC'
            )
            if etm_without_ec_count >= 2:
                is_stuffing = True

        if not is_stuffing and total_tokens > 0:
            if (len(nouns) / total_tokens >= 0.8) and len(josa) == 0:
                is_stuffing = True

        if is_stuffing:
            s3 = 0.0
        else:
            has_josa = len(josa) > 0
            has_verb_or_modifier = len(verbs_adjs_modifiers) > 0
            if has_josa and has_verb_or_modifier:
                s3 = 0.25
            elif has_verb_or_modifier and not has_josa:
                s3 = 0.16
            elif has_josa and not has_verb_or_modifier:
                s3 = 0.08
            else:
                s3 = 0.0

        return SingleAltEvaluationResult(
            final_score=round(s1 + s2 + s3, 3),
            has_alt_attribute=True,
            is_text_image=is_text_image,
            clean_alt=clean_alt,
            is_stuffing=is_stuffing,
            s1_presence_score=s1,
            s2_relevance_score=s2,
            s3_sentence_score=s3
        )

    def calculate_avg_alt_score(self, image_list: List[Dict[str, Any]]) -> ImageAltEvaluationResult:
        if not image_list:
            return ImageAltEvaluationResult(1.0, 1.0, True, 0, 0, [])

        details = [self.evaluate_single_alt(img.get('alt'), img.get('is_text_image', False)) for img in image_list]
        raw_avg = round(sum(d.final_score for d in details) / len(details), 3)
        is_valid = raw_avg >= 0.2

        return ImageAltEvaluationResult(
            avg_score=raw_avg if is_valid else 0.0,
            raw_avg_score=raw_avg,
            is_valid=is_valid,
            total_image_count=len(image_list),
            stuffing_image_count=sum(1 for d in details if d.is_stuffing),
            detail_results=details
        )

    # =========================================================================
    # Total 평가 함수 (product_cat 및 product_name 인자 추가 반영)
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

        total_score = round(((s1 + s2.final_score + s3.final_score + s4.final_score + s5.avg_score) / 5.0) * 100, 0)

        return GEOTotalEvaluationResult(
            total_score=total_score,
            text_ratio_score=round(s1, 2),
            hybrid_search=s2,
            keyword_stuffing=s3,
            json_ld=s4,
            image_alt=s5
        )