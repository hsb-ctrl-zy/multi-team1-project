import json
import re
import warnings
from typing import List, Dict, Any, Union
from collections import Counter
from dataclasses import dataclass, field

from konlpy.tag import Komoran
from rank_bm25 import BM25Okapi
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
    avg_cosine_sim_thresh: float
    is_passed_threshold: bool
    query_details: List[Dict[str, Any]]

@dataclass
class KeywordStuffingResult:
    final_score: float       # Threshold 적용 후 최종 반영 점수
    raw_score: float         # Threshold 적용 전 감점만 반영된 원래 점수
    is_stuffing: bool        # 스터핑(어뷰징) 판정 여부 (True/False)
    noun_penalty: float      # 명사 비율 페널티
    grammatical_penalty: float # 문법 요소 미비 페널티
    pattern_penalty: float   # 패턴 반복 페널티
    noun_ratio: float        # 명사 비율 (통계용)
    grammatical_ratio: float # 문법 요소 비율 (통계용)
    pattern_count: int       # 패턴 감지 횟수 (통계용)

@dataclass
class JsonLdEvaluationResult:
    final_score: float        # Threshold(0.3) 적용 후 최종 점수
    raw_score: float          # Threshold 적용 전 원본 가중치 합산 점수
    is_valid: bool            # Threshold(0.3) 통과 여부 (True/False)
    parsing_score: float      # 1. 파싱 및 유형 점수 (0.0~1.0)
    density_score: float      # 2. Description 정보 밀도 점수 (0.0~1.0)
    clothing_score: float     # 3. 의류 커머스 특화 속성 점수 (0.0~1.0)
    trust_score: float        # 4. 신뢰도 및 엔티티 연결성 점수 (0.0~1.0)
    present_attrs_count: int  # 존재 속성 개수 (원피스/의류 속성)
    trust_count: int          # 존재 신뢰도 속성 개수

@dataclass
class SingleAltEvaluationResult:
    final_score: float           # 단일 이미지 최종 합산 점수 (0.0 ~ 0.75)
    has_alt_attribute: bool      # 1. alt 속성 존재 여부
    is_text_image: bool          # 이미지 유형 (텍스트 포함 여부)
    clean_alt: str               # 정제된 alt 텍스트
    is_stuffing: bool            # 3-1. 키워드 스터핑 감지 여부
    s1_presence_score: float     # 1. 속성 존재 점수 (0.0 or 0.25)
    s2_relevance_score: float    # 2. 적절성 점수 (0.0 or 0.25)
    s3_sentence_score: float     # 3. 문장 완성도 점수 (0.0, 0.08, 0.16, 0.25)

@dataclass
class ImageAltEvaluationResult:
    avg_score: float             # Threshold(0.2) 적용 후 최종 평균 점수
    raw_avg_score: float         # Threshold 적용 전 원본 평균 점수
    is_valid: bool               # Threshold 통과 여부 (>= 0.2)
    total_image_count: int       # 전체 이미지 수
    stuffing_image_count: int    # 스터핑 감지된 이미지 수
    detail_results: List[SingleAltEvaluationResult] = field(default_factory=list)

@dataclass
class GEOTotalEvaluationResult:
    total_score: float                       # 최종 GEO 종합 점수 (0 ~ 100)
    text_ratio_score: float                  # ① 본문 텍스트 비율 점수
    hybrid_search: HybridSearchResult        # ② 하이브리드 검색 결과 (객체)
    keyword_stuffing: KeywordStuffingResult  # ③ 키워드 스터핑 결과 (객체)
    json_ld: JsonLdEvaluationResult          # ④ JSON-LD 구조화 데이터 결과 (객체)
    image_alt: ImageAltEvaluationResult      # ⑤ 이미지 Alt 평가 결과 (객체)

    # ✨ 문자열 호출 시 바로 "60점/100점 만점" 형식으로 반환하는 속성
    @property
    def formatted_score(self) -> str:
        # 소수점 없이 깔끔하게 보이고 싶다면 int(self.total_score_100) 사용
        return f"{int(self.total_score)}점/100점 만점"

class GEOScorer:
    def __init__(self):
        pass

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        return re.sub(r'\s+', ' ', text).strip()

    # ① 본문 텍스트 비율 (Text Ratio)
    def calculate_text_ratio(self, body_text: str, image_text: str) -> float:
        clean_body = self._clean_text(body_text)
        clean_image = self._clean_text(image_text)

        body_len = len(clean_body)
        image_len = len(clean_image)
        total = body_len + image_len
        if total == 0:
            return 0.0
        
        ratio = body_len / total
        return ratio if ratio >= 0.2 else 0.0
    
    # ==========================================
    # ② 하이브리드 검색 모델 점수 (키워드 오버랩 적용)
    # ==========================================
    def calculate_hybrid_search(self, user_queries: Union[str, List[str]], page_text: str) -> HybridSearchResult:
        clean_page = self._clean_text(page_text)
        if not clean_page:
            return HybridSearchResult(0.0, 0.0, 0.0, 0.0, 0.0, False, [])

        if isinstance(user_queries, str):
            user_queries = [user_queries]

        cleaned_queries = [self._clean_text(q) for q in user_queries if self._clean_text(q)]
        if not cleaned_queries:
            return HybridSearchResult(0.0, 0.0, 0.0, 0.0, 0.0, False, [])

        # 본문 명사 추출 및 탐색 성능을 위한 set 변환
        page_nouns = komoran.nouns(clean_page)
        page_nouns_set = set(page_nouns) if page_nouns else set(komoran.morphs(clean_page))
        
        p_vec = embedding_model.encode([clean_page])

        query_details = []
        lexical_overlap_list = []
        cos_raw_list = []
        cos_thresh_list = []
        combined_score_list = []

        for query in cleaned_queries:
            query_tokens = komoran.nouns(query)
            if not query_tokens:
                query_tokens = komoran.morphs(query)

            # 1. 키워드 오버랩(Lexical Overlap) 계산
            if query_tokens:
                matched_count = sum(1 for token in query_tokens if token in page_nouns_set)
                lexical_overlap = matched_count / len(query_tokens)
            else:
                lexical_overlap = 0.0

            # 2. 코사인 유사도(Semantic Similarity) 계산
            q_vec = embedding_model.encode([query])
            cos_sim_raw = float(cosine_similarity(q_vec, p_vec)[0][0])
            cos_sim_after_thresh = 0.0 if cos_sim_raw < 0.5 else max(0.0, min(cos_sim_raw, 1.0))

            # 3. 결합 점수 (키워드 매칭 50% : 의미 유사도 50%)
            query_hybrid_score = (lexical_overlap * 0.5) + (cos_sim_after_thresh * 0.5)

            # 기록용 저장
            lexical_overlap_list.append(lexical_overlap)
            cos_raw_list.append(cos_sim_raw)
            cos_thresh_list.append(cos_sim_after_thresh)
            combined_score_list.append(query_hybrid_score)

            query_details.append({
                "query": query,
                "lexical_overlap": round(lexical_overlap, 4),        # 키워드 매칭 비율 (0.0 ~ 1.0)
                "cosine_sim_raw": round(cos_sim_raw, 4),             # 코사인 유사도 원본
                "cosine_sim_after_thresh": round(cos_sim_after_thresh, 4),
                "query_hybrid_score": round(query_hybrid_score, 4)   # 개별 결합 점수
            })

        # 항목별 평균값 산출
        n = len(cleaned_queries)
        avg_lexical_overlap = sum(lexical_overlap_list) / n
        avg_cosine_sim_raw = sum(cos_raw_list) / n
        avg_cosine_sim_thresh = sum(cos_thresh_list) / n
        avg_combined_score = sum(combined_score_list) / n

        is_passed = avg_combined_score >= 0.6
        final_score = round(avg_combined_score, 2) if is_passed else 0.0

        # 결과를 HybridSearchResult 객체로 리턴
        return HybridSearchResult(
            final_score=final_score,
            avg_combined_score=round(avg_combined_score, 4),
            avg_lexical_overlap=round(avg_lexical_overlap, 4),
            avg_cosine_sim_raw=round(avg_cosine_sim_raw, 4),
            avg_cosine_sim_thresh=round(avg_cosine_sim_thresh, 4),
            is_passed_threshold=is_passed,
            query_details=query_details
        )


    # ==========================================
    # ③ 키워드 스터핑(Keyword Stuffing) 검출 (3가지 영역 균등 배분 모델)
    # ==========================================
    def calculate_keyword_stuffing(self, page_text: str) -> KeywordStuffingResult:
        clean_page = self._clean_text(page_text)
        if not clean_page:
            return KeywordStuffingResult(
                final_score=0.0, raw_score=0.0, is_stuffing=True,
                noun_penalty=0.0, grammatical_penalty=0.0, pattern_penalty=0.0,
                noun_ratio=0.0, grammatical_ratio=0.0, pattern_count=0
            )

        pos_tags = komoran.pos(clean_page)
        if not pos_tags:
            return KeywordStuffingResult(
                final_score=0.0, raw_score=0.0, is_stuffing=True,
                noun_penalty=0.0, grammatical_penalty=0.0, pattern_penalty=0.0,
                noun_ratio=0.0, grammatical_ratio=0.0, pattern_count=0
            )

        total_tokens = len(pos_tags)

        # 1. 태그별 형태소 추출
        nouns = [word for word, tag in pos_tags if tag.startswith('N')]        # 명사(N)
        particles = [word for word, tag in pos_tags if tag.startswith('J')]    # 조사(J)
        endings = [word for word, tag in pos_tags if tag.startswith('E')]      # 어미(E)

        noun_ratio = len(nouns) / total_tokens
        grammatical_ratio = (len(particles) + len(endings)) / total_tokens

        # 2. 감점 항목 계산 (각 영역별 최대 0.333 감점)
        
        # [영역 1] 명사 비율 감점 (기준: 50% 초과 시 초과분에 비례하여 감점)
        noun_penalty = 0.0
        if noun_ratio > 0.5:
            # 50%~100% 구간을 0.0~0.333 감점으로 매핑
            noun_penalty = min(0.333, ((noun_ratio - 0.5) / 0.5) * 0.333)

        # [영역 2] 문법 요소 미비 감점 (기준: 조사/어미 비율 20% 미만 시 부족분에 비례하여 감점)
        grammatical_penalty = 0.0
        if grammatical_ratio < 0.2:
            # 0%~20% 구간의 부족분을 0.0~0.333 감점으로 매핑
            grammatical_penalty = min(0.333, ((0.2 - grammatical_ratio) / 0.2) * 0.333)

        # [영역 3] 특정 접미 패턴 반복 감점 (기준: ~룩, ~핏 등 2회 이상부터 감점)
        pattern_count = sum(1 for n in nouns if n.endswith('룩') or n.endswith('핏'))
        pattern_penalty = 0.0
        if pattern_count >= 2:
            # 2회~5회 이상 구간을 0.0~0.333 감점으로 매핑
            pattern_penalty = min(0.333, ((pattern_count - 1) / 4) * 0.333)

        # 3. 최종 점수 계산 (기본 1.0 - 총 감점액)
        total_penalty = noun_penalty + grammatical_penalty + pattern_penalty
        raw_score = max(0.0, 1.0 - total_penalty)

        # 4. Threshold 적용: 0.6 미만은 스터핑(어뷰징)으로 간주하여 0.0 반환
        is_stuffing = raw_score < 0.6
        final_score = raw_score if not is_stuffing else 0.0

        return KeywordStuffingResult(
                final_score=final_score,
                raw_score=raw_score,
                is_stuffing=is_stuffing,
                noun_penalty=round(noun_penalty, 3),
                grammatical_penalty=round(grammatical_penalty, 3),
                pattern_penalty=round(pattern_penalty, 3),
                noun_ratio=round(noun_ratio, 2),
                grammatical_ratio=round(grammatical_ratio, 2),
                pattern_count=pattern_count
            )

    # ④ JSON-LD 구조화 데이터 평가 (dataclass 상세 리턴 버전)
    def calculate_json_ld_score(self, json_ld_str: str) -> JsonLdEvaluationResult:
        # 빈 문자열 또는 파싱 실패 시 기본 0점 리턴
        default_fail = JsonLdEvaluationResult(
            final_score=0.0, raw_score=0.0, is_valid=False,
            parsing_score=0.0, density_score=0.0, clothing_score=0.0, trust_score=0.0,
            present_attrs_count=0, trust_count=0
        )

        if not json_ld_str or not json_ld_str.strip():
            return default_fail

        try:
            data = json.loads(json_ld_str)
        except Exception:
            return default_fail

        # 1. 파싱 및 유형 평가
        has_product = False
        has_graph = False

        if isinstance(data, dict):
            if "@graph" in data:
                has_graph = True
                for item in data.get("@graph", []):
                    if isinstance(item, dict) and item.get("@type") == "Product":
                        has_product = True
                        data = item
                        break
            elif data.get("@type") == "Product":
                has_product = True

        if has_product and has_graph:
            parsing_score = 1.0
        elif has_product:
            parsing_score = 0.7
        elif has_graph:
            parsing_score = 0.3
        else:
            return default_fail

        # 2. Description 글자 수 기준
        desc = data.get("description", "") if isinstance(data, dict) else ""
        desc_clean = re.sub(r"[^\w\s]", "", desc).strip()
        desc_len = len(desc_clean)

        if desc_len >= 100:
            density_score = 1.0
        elif desc_len >= 50:
            density_score = 0.7
        elif desc_len >= 1:
            density_score = 0.35
        else:
            density_score = 0.0

        # 3. 의류 커머스 특화 속성 (4개 항목)
        attr_keys = ["material", "color", "size", "category"]
        present_attrs = sum(1 for k in attr_keys if isinstance(data, dict) and k in data)
        clothing_score = present_attrs / 4.0

        # 4. 신뢰도 및 엔티티 연결성 (3개 항목)
        trust_count = 0
        if isinstance(data, dict):
            if has_graph or "sameAs" in data:
                trust_count += 1
            if "shippingDetails" in data:
                trust_count += 1
            if "hasMerchantReturnPolicy" in data:
                trust_count += 1
        trust_score = trust_count / 3.0

        # 가중치 결합 (4개 요소 합산)
        raw_score = (parsing_score * 0.25) + (density_score * 0.25) + (clothing_score * 0.25) + (trust_score * 0.25)
        raw_score = round(raw_score, 3)

        # Threshold(0.3) 적용
        is_valid = raw_score >= 0.3
        final_score = raw_score if is_valid else 0.0

        return JsonLdEvaluationResult(
            final_score=final_score,
            raw_score=raw_score,
            is_valid=is_valid,
            parsing_score=round(parsing_score, 2),
            density_score=round(density_score, 2),
            clothing_score=round(clothing_score, 2),
            trust_score=round(trust_score, 2),
            present_attrs_count=present_attrs,
            trust_count=trust_count
        )

    # ==========================================
    # ⑤ 이미지 Alt 속성 평가 (기획안 완벽 반영)
    # ==========================================
    
    # ⑤-1. 단일 이미지 평가 (기획안 4개 항목 + 스터핑 ETM/EC 규칙 반영)
    def evaluate_single_alt(self, alt_text: str, is_text_image: bool) -> SingleAltEvaluationResult:
        if alt_text is None:
            return SingleAltEvaluationResult(
                final_score=0.0, has_alt_attribute=False, is_text_image=is_text_image,
                clean_alt="", is_stuffing=False, s1_presence_score=0.0,
                s2_relevance_score=0.0, s3_sentence_score=0.0
            )

        s1 = 0.25
        clean_alt = self._clean_text(alt_text)

        # 2. 적절성 평가
        if not is_text_image:
            s2 = 0.25 if clean_alt == "" else 0.0
        else:
            s2 = 0.25 if len(clean_alt) >= 10 else 0.0

        if clean_alt == "":
            return SingleAltEvaluationResult(
                final_score=round(s1 + s2, 3), has_alt_attribute=True,
                is_text_image=is_text_image, clean_alt="", is_stuffing=False,
                s1_presence_score=s1, s2_relevance_score=s2, s3_sentence_score=0.0
            )

        # 3. GEO 문장 완성도 및 스터핑 검사
        pos_tags = komoran.pos(clean_alt)
        total_tokens = len(pos_tags)

        nouns = [w for w, p in pos_tags if p.startswith('N')]
        josa = [w for w, p in pos_tags if p.startswith('J')]
        verbs_adjs_modifiers = [w for w, p in pos_tags if p.startswith('V') or p.startswith('M')]

        # 3-1. 스터핑 탐지
        is_stuffing = False
        if len(nouns) >= 4:
            is_stuffing = True

        if not is_stuffing:
            etm_without_ec_count = 0
            for i in range(len(pos_tags) - 1):
                curr_p = pos_tags[i][1]
                next_p = pos_tags[i+1][1]
                if curr_p == 'ETM' and next_p != 'EC':
                    etm_without_ec_count += 1
            if etm_without_ec_count >= 2:
                is_stuffing = True

        if not is_stuffing and total_tokens > 0:
            if (len(nouns) / total_tokens >= 0.8) and len(josa) == 0:
                is_stuffing = True

        # 3-2. 문장 구성도 배점 (기획안 0.25, 0.16, 0.08, 0.0 반영)
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

        final_score = round(s1 + s2 + s3, 3)

        return SingleAltEvaluationResult(
            final_score=final_score, has_alt_attribute=True,
            is_text_image=is_text_image, clean_alt=clean_alt,
            is_stuffing=is_stuffing, s1_presence_score=s1,
            s2_relevance_score=s2, s3_sentence_score=s3
        )

    # ⑤-2. 페이지 전체 이미지 집계 (Threshold 0.2 반영 및 단일 평가 결과 리스트 포함)
    def calculate_avg_alt_score(self, image_list: List[Dict[str, Any]]) -> ImageAltEvaluationResult:
        if not image_list:
            return ImageAltEvaluationResult(
                avg_score=1.0, raw_avg_score=1.0, is_valid=True,
                total_image_count=0, stuffing_image_count=0, detail_results=[]
            )

        details = [
            self.evaluate_single_alt(img.get('alt'), img.get('is_text_image', False))
            for img in image_list
        ]

        raw_avg = round(sum(d.final_score for d in details) / len(details), 3)
        is_valid = raw_avg >= 0.2
        avg_score = raw_avg if is_valid else 0.0
        stuffing_count = sum(1 for d in details if d.is_stuffing)

        return ImageAltEvaluationResult(
            avg_score=avg_score, raw_avg_score=raw_avg, is_valid=is_valid,
            total_image_count=len(image_list), stuffing_image_count=stuffing_count,
            detail_results=details
        )
    
    # 🏆 [통합 메인 함수] 모든 평가를 한 번에 실행하고 종합 점수 산출
    def evaluate_page(
        self,
        body_text: str,
        image_text: str,
        user_queries: Union[str, List[str]],
        json_ld_str: str = "",
        image_list: List[Dict[str, Any]] = None
    ) -> GEOTotalEvaluationResult:
        
        if image_list is None:
            image_list = []
        # 전체 페이지 텍스트 = 본문 텍스트 + 이미지 텍스트
        page_text = f"{body_text} {image_text}".strip()

        # 1. 각 평가 항목 실행
        s1 = self.calculate_text_ratio(body_text=body_text, image_text=image_text)
        s2 = self.calculate_hybrid_search(user_queries=user_queries, page_text=page_text)
        s3 = self.calculate_keyword_stuffing(page_text=page_text)
        s4 = self.calculate_json_ld_score(json_ld_str=json_ld_str)
        s5 = self.calculate_avg_alt_score(image_list=image_list)

        # 2. 최종 가중치/평균 점수 계산 (5개 항목 산술 평균)
        total_score = round(((s1 + s2.final_score + s3.final_score + s4.final_score + s5.avg_score) / 5.0)*100, 0)

        # 3. 통합 결과 객체 반환
        return GEOTotalEvaluationResult(
            total_score=total_score,
            text_ratio_score=round(s1, 2),
            hybrid_search=s2,
            keyword_stuffing=s3,
            json_ld=s4,
            image_alt=s5
        )    
    
# =======================================================================================================================
