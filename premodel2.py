import json
import re
import warnings
from typing import List, Dict, Any, Union
from collections import Counter

from konlpy.tag import Komoran
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Python 3.12 스레드 종속성 종료 관련 자원 경고 무시
warnings.filterwarnings("ignore", category=ResourceWarning)

# 형태소 분석기 및 임베딩 모델 로드
komoran = Komoran()
embedding_model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")


class GEOScorer:
    def __init__(self):
        pass

    # 💡 텍스트 정제 헬퍼 함수 (Komoran 줄바꿈 예외 및 연속 공백 정제)
    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        return re.sub(r'\s+', ' ', text).strip()

    # ==========================================
    # ① 본문 텍스트 비율 (Text Ratio)
    # ==========================================
    def calculate_text_ratio(self, body_text_len: int, image_text_len: int) -> float:
        total = body_text_len + image_text_len
        if total == 0:
            return 0.0
        
        ratio = body_text_len / total
        return ratio if ratio >= 0.2 else 0.0

    # ==========================================
    # ② 하이브리드 검색 모델 점수 (음수 방지 및 범위 고정)
    # ==========================================
    def calculate_hybrid_search(self, user_queries: Union[str, List[str]], page_text: str) -> float:
        clean_page = self._clean_text(page_text)
        if not clean_page:
            return 0.0

        if isinstance(user_queries, str):
            user_queries = [user_queries]

        cleaned_queries = [self._clean_text(q) for q in user_queries if self._clean_text(q)]
        if not cleaned_queries:
            return 0.0

        page_tokens = komoran.morphs(clean_page)
        bm25 = BM25Okapi([page_tokens])
        p_vec = embedding_model.encode([clean_page])

        total_hybrid_score = 0.0

        for query in cleaned_queries:
            query_tokens = komoran.morphs(query)
            bm25_raw = bm25.get_scores(query_tokens)[0]
            
            # 💡 [핵심] min/max를 사용해 점수 범위를 [0.0 ~ 1.0]으로 강제 고정
            bm25_score = max(0.0, min(bm25_raw / 1.5, 1.0))

            q_vec = embedding_model.encode([query])
            cos_sim = float(cosine_similarity(q_vec, p_vec)[0][0])
            
            # 💡 코사인 유사도도 음수 방지 (0.0 ~ 1.0)
            cos_sim = max(0.0, min(cos_sim, 1.0))

            # 단일 쿼리 하이브리드 점수 (BM25 40% + Semantic 60%)
            query_hybrid_score = (bm25_score * 0.4) + (cos_sim * 0.6)
            total_hybrid_score += query_hybrid_score

        # 전체 쿼리에 대한 평균 점수 산출
        avg_hybrid_score = total_hybrid_score / len(cleaned_queries)
        
        # 💡 최종 점수도 0.0 ~ 1.0 범위를 벗어나지 않도록 최종 안전장치
        final_score = max(0.0, min(avg_hybrid_score, 1.0))
        return round(final_score, 2)

    # ==========================================
    # ③ 본문 키워드 스터핑 (Keyword Stuffing)
    # ==========================================
    def calculate_keyword_stuffing(self, page_text: str) -> float:
        clean_page = self._clean_text(page_text)
        if not clean_page:
            return 1.0

        nouns = komoran.nouns(clean_page)
        total_nouns = len(nouns)
        
        if total_nouns == 0:
            return 1.0

        counts = Counter(nouns).most_common(3)
        
        max_noun_count = counts[0][1] if len(counts) >= 1 else 0
        top3_sum_count = sum([c[1] for c in counts])

        max_noun_ratio = max_noun_count / total_nouns
        top3_density = top3_sum_count / total_nouns

        stuffing_score = 1.0 - (max_noun_ratio * 0.5 + top3_density * 0.5)
        return stuffing_score if stuffing_score >= 0.6 else 0.0

    # ==========================================
    # ④ JSON-LD 구조화 데이터 평가
    # ==========================================
    def calculate_json_ld_score(self, json_ld_str: str) -> float:
        try:
            data = json.loads(json_ld_str)
        except Exception:
            return 0.0

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
            parsing_score = 0.8
        elif has_graph:
            parsing_score = 0.4
        else:
            return 0.0

        desc = data.get("description", "") if isinstance(data, dict) else ""
        desc_clean = re.sub(r"[^\w\s]", "", desc).strip()
        desc_len = len(desc_clean)

        if desc_len >= 60:
            density_score = 1.0
        elif desc_len >= 30:
            density_score = 0.7
        elif desc_len > 0:
            density_score = 0.4
        else:
            density_score = 0.0

        attr_keys = ["material", "color", "size", "category"]
        present_attrs = sum(1 for k in attr_keys if isinstance(data, dict) and k in data)
        clothing_score = present_attrs / 4.0

        total_json_score = (parsing_score * 0.4) + (density_score * 0.3) + (clothing_score * 0.3)
        return total_json_score

    # ==========================================
    # ⑤ 이미지 Alt 속성 평가
    # ==========================================
    def evaluate_single_alt(self, alt_text: str, is_text_image: bool) -> float:
        if alt_text is None:
            return 0.0
        
        clean_alt = self._clean_text(alt_text)
        s1 = 0.25

        if not is_text_image:
            s2 = 0.25 if (clean_alt == "" or len(clean_alt) >= 2) else 0.0
        else:
            s2 = 0.25 if len(clean_alt) > 5 else 0.0

        pos_tags = komoran.pos(clean_alt) if clean_alt else []
        nouns = [w for w, p in pos_tags if p.startswith('N')]
        josa = [w for w, p in pos_tags if p.startswith('J')]
        verbs_adjs = [w for w, p in pos_tags if p.startswith('V')]

        if len(nouns) >= 6 and len(josa) == 0:
            s3 = 0.0
        else:
            has_josa = len(josa) > 0
            has_verb_adj = len(verbs_adjs) > 0

            if has_josa and has_verb_adj:
                s3 = 0.25
            elif has_verb_adj or has_josa:
                s3 = 0.16
            else:
                s3 = 0.08

        return min(s1 + s2 + s3, 1.0)

    def calculate_avg_alt_score(self, image_list: List[Dict[str, Any]]) -> float:
        if not image_list:
            return 1.0

        scores = [self.evaluate_single_alt(img.get('alt'), img.get('is_text_image', False)) for img in image_list]
        avg_score = sum(scores) / len(scores)
        return avg_score if avg_score >= 0.2 else 0.0


# ==========================================
# 실행 테스트 메인 영역
# ==========================================
if __name__ == "__main__":
    scorer = GEOScorer()

    # 💡 다중 쿼리 세트 정의 (자연어 질의, 키워드, 속성 조합)
    queries = [
        "봄에 입기 좋은 화이트 오버핏 드레스 셔츠 추천해줘",
        "봄 화이트 오버핏 셔츠",
        "100% 코튼 남성 루즈핏 데일리 셔츠"
    ]

    # ------------------------------------------
    # [케이스 1] 일반 비최적화 쇼핑몰 예시
    # ------------------------------------------
    text_1 = "화이트셔츠 남여공용 오버핏 셔츠 봄신상 데이트룩 하객룩 추천"
    json_ld_1 = "{}"
    images_1 = [
        {"alt": "detail_page_01.jpg", "is_text_image": True},
        {"alt": "", "is_text_image": False}
    ]

    s1_1 = scorer.calculate_text_ratio(body_text_len=len(text_1), image_text_len=500)
    s2_1 = scorer.calculate_hybrid_search(user_queries=queries, page_text=text_1)
    s3_1 = scorer.calculate_keyword_stuffing(page_text=text_1)
    s4_1 = scorer.calculate_json_ld_score(json_ld_str=json_ld_1)
    s5_1 = scorer.calculate_avg_alt_score(image_list=images_1)
    total_score_1 = round((s1_1 + s2_1 + s3_1 + s4_1 + s5_1) / 5.0, 3)

    print("=== 📊 [케이스 1] 일반 비최적화 쇼핑몰 평가 결과 ===")
    print(f"① 본문 텍스트 비율 점수 : {s1_1:.2f}")
    print(f"② 다중 쿼리 하이브리드 검색 점수 : {s2_1:.2f}")
    print(f"③ 키워드 스터핑 안전 점수: {s3_1:.2f}")
    print(f"④ JSON-LD 구조화 점수 : {s4_1:.2f}")
    print(f"⑤ 이미지 Alt 속성 평가 점수: {s5_1:.2f}")
    print("-----------------------------------")
    print(f"🏆 최종 GEO Score: {total_score_1} / 1.00\n")

    # ------------------------------------------
    # [케이스 2] GEO 최적화 모범 답안 예시
    # ------------------------------------------
    text_2 = """
    봄 시즌에 입기 좋은 화이트 오버핏 드레스 셔츠입니다. 
    100% 프리미엄 코튼 소재로 제작되어 트렌디한 오버핏 실루엣과 편안한 착용감을 동시에 제공합니다. 
    청바지나 슬랙스와 함께 연출하여 봄철 데일리룩으로 활용하기 좋습니다.
    """
    
    json_ld_2 = """
    {
        "@context": "https://schema.org/",
        "@graph": [
            {
                "@type": "Product",
                "name": "봄 화이트 오버핏 드레스 셔츠",
                "description": "봄 시즌에 입기 좋은 100% 프리미엄 코튼 소재의 화이트 오버핏 드레스 셔츠입니다. 루즈한 실루엣으로 편안하고 스타일리시한 데일리룩 연출이 가능합니다.",
                "material": "Cotton 100%",
                "color": "White",
                "size": "Free",
                "category": "Shirts"
            }
        ]
    }
    """
    
    images_2 = [
        {"alt": "모델이 봄 시즌용 화이트 오버핏 드레스 셔츠를 착용한 정면 사진입니다.", "is_text_image": True},
        {"alt": "100% 프리미엄 코튼 셔츠 원단의 부드러운 질감을 확대하여 보여주는 사진입니다.", "is_text_image": True}
    ]

    s1_2 = scorer.calculate_text_ratio(body_text_len=len(text_2), image_text_len=0)
    s2_2 = scorer.calculate_hybrid_search(user_queries=queries, page_text=text_2)
    s3_2 = scorer.calculate_keyword_stuffing(page_text=text_2)
    s4_2 = scorer.calculate_json_ld_score(json_ld_str=json_ld_2)
    s5_2 = scorer.calculate_avg_alt_score(image_list=images_2)
    total_score_2 = round((s1_2 + s2_2 + s3_2 + s4_2 + s5_2) / 5.0, 3)

    print("=== 🌟 [케이스 2] GEO 최적화 모범 답안 평가 결과 ===")
    print(f"① 본문 텍스트 비율 점수 : {s1_2:.2f}")
    print(f"② 다중 쿼리 하이브리드 검색 점수 : {s2_2:.2f}")
    print(f"③ 키워드 스터핑 안전 점수: {s3_2:.2f}")
    print(f"④ JSON-LD 구조화 점수 : {s4_2:.2f}")
    print(f"⑤ 이미지 Alt 속성 평가 점수: {s5_2:.2f}")
    print("-----------------------------------")
    print(f"🏆 최종 GEO Score: {total_score_2} / 1.00")