import json
import re
import warnings
from typing import List, Dict, Any, Union
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

class GEOScorer:
    def __init__(self):
        pass

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        return re.sub(r'\s+', ' ', text).strip()

    def calculate_text_ratio(self, body_text: str, image_text: str) -> float:
        clean_body = self._clean_text(body_text)
        clean_image = self._clean_text(image_text)
        total = len(clean_body) + len(clean_image)
        if total == 0:
            return 0.0
        ratio = len(clean_body) / total
        return ratio if ratio >= 0.2 else 0.0

    # ==========================================
    # ② 하이브리드 검색 (Threshold 전면 제거 버전)
    # ==========================================
    def calculate_hybrid_search(self, user_queries: Union[str, List[str]], page_text: str) -> HybridSearchResult:
        clean_page = self._clean_text(page_text)
        if not clean_page:
            return HybridSearchResult(0.0, 0.0, 0.0, 0.0, [])

        if isinstance(user_queries, str):
            user_queries = [user_queries]

        cleaned_queries = [self._clean_text(q) for q in user_queries if self._clean_text(q)]
        if not cleaned_queries:
            return HybridSearchResult(0.0, 0.0, 0.0, 0.0, [])

        page_nouns = komoran.nouns(clean_page)
        page_nouns_set = set(page_nouns) if page_nouns else set(komoran.morphs(clean_page))
        p_vec = embedding_model.encode([clean_page])

        query_details, lexical_overlap_list, cos_raw_list, combined_score_list = [], [], [], []

        for query in cleaned_queries:
            query_tokens = komoran.nouns(query) or komoran.morphs(query)
            matched_count = sum(1 for token in query_tokens if token in page_nouns_set) if query_tokens else 0
            lexical_overlap = matched_count / len(query_tokens) if query_tokens else 0.0

            q_vec = embedding_model.encode([query])
            cos_sim_raw = float(cosine_similarity(q_vec, p_vec)[0][0])
            
            # Threshold 절삭 로직 제거 (원 코사인 유사도 그대로 반영)
            query_hybrid_score = (lexical_overlap * 0.5) + (cos_sim_raw * 0.5)

            lexical_overlap_list.append(lexical_overlap)
            cos_raw_list.append(cos_sim_raw)
            combined_score_list.append(query_hybrid_score)

            query_details.append({
                "query": query,
                "lexical_overlap": round(lexical_overlap, 4),
                "cosine_sim_raw": round(cos_sim_raw, 4),
                "query_hybrid_score": round(query_hybrid_score, 4)
            })

        n = len(cleaned_queries)
        avg_combined = sum(combined_score_list) / n

        # Threshold(0.6) 조건 없이 계산된 그대로 final_score 반환
        return HybridSearchResult(
            final_score=round(avg_combined, 4),
            avg_combined_score=round(avg_combined, 4),
            avg_lexical_overlap=round(sum(lexical_overlap_list) / n, 4),
            avg_cosine_sim_raw=round(sum(cos_raw_list) / n, 4),
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

    # 리스트를 재귀적으로 평탄화하여 딕셔너리 객체들을 모두 추출하는 헬퍼 함수
    def _flatten_json_items(self, items: Any) -> List[dict]:
        flat = []
        if isinstance(items, list):
            for item in items:
                flat.extend(self._flatten_json_items(item))
        elif isinstance(items, dict):
            flat.append(items)
        return flat

    # ==========================================
    # ④ json-ld 평가 (Threshold 완전 제거 버전)
    # ==========================================
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

        # 다중/중첩 리스트 완전 벗기기
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

        print("\n================ [JSON-LD 파싱 완료 내용] ================")
        print(json.dumps(target_product, ensure_ascii=False, indent=2))
        print("=========================================================\n")

        data = target_product

        # 2. Description 평가
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

        # 3. 의류 속성 평가
        attr_keys = ["material", "color", "size", "category"]
        present_attrs = sum(1 for k in attr_keys if isinstance(data, dict) and k in data)
        clothing_score = present_attrs / 4.0

        # 4. 신뢰도 평가
        trust_count = 0
        if isinstance(data, dict):
            if has_graph or "sameAs" in data:
                trust_count += 1
            if "shippingDetails" in data:
                trust_count += 1
            if "hasMerchantReturnPolicy" in data:
                trust_count += 1
        trust_score = trust_count / 3.0

        # 원본 점수 계산 (4개 항목 평균)
        raw_score = round(
            (parsing_score * 0.25) + (density_score * 0.25) + (clothing_score * 0.25) + (trust_score * 0.25), 4
        )
        
        # Threshold 절삭 삭제: raw_score를 그대로 final_score로 사용
        final_score = raw_score
        is_valid = has_product  # 파싱 및 Product 존재 여부만 표현

        print(f"[JSON-LD 점수] 최종점수(raw): {final_score}, Description길이: {desc_len}, 의류속성수: {present_attrs}")

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

    def evaluate_page(
        self,
        body_text: str,
        image_text: str,
        user_queries: Union[str, List[str]],
        json_ld_str: Union[str, list, dict] = "",
        image_list: List[Dict[str, Any]] = None
    ) -> GEOTotalEvaluationResult:

        if image_list is None:
            image_list = []
        page_text = f"{body_text} {image_text}".strip()

        s1 = self.calculate_text_ratio(body_text=body_text, image_text=image_text)
        s2 = self.calculate_hybrid_search(user_queries=user_queries, page_text=page_text)
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