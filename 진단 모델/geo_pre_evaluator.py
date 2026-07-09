import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os
from collections import Counter

# 방금 설치한 Java 경로를 환경변수에 명시적으로 주입 (쥬피터 재시작 없이 바로 실행되도록)
os.environ['JAVA_HOME'] = r'C:\Program Files\Microsoft\jdk-17.0.19.10-hotspot'
from konlpy.tag import Okt

class GeoPreEvaluatorPandas:
    def __init__(self):
        # TF-IDF 벡터화기 초기화
        self.vectorizer = TfidfVectorizer()
        # 한국어 형태소 분석기 초기화 (Okt)
        self.okt = Okt()

    def normalize_korean_text(self, text: str) -> str:
        """
        Okt 형태소 분석기를 사용하여 텍스트를 명사, 동사, 형용사 등의 어간 단위로 정제합니다.
        stem=True 옵션을 통해 '시원한', '시원함' -> '시원하다' 로 자동 통일됨
        '땀흡수' -> '땀', '흡수' 로 분리됨
        """
        if not text:
            return ""
        tokens = self.okt.pos(text, stem=True)
        # 명사, 동사, 형용사만 추출
        words = [word for word, pos in tokens if pos in ['Noun', 'Verb', 'Adjective']]
        return " ".join(words)

    def evaluate_single_product(self, df_source: pd.DataFrame, df_image: pd.DataFrame, df_query: pd.DataFrame) -> pd.DataFrame:
        """
        단일 상품(Page)에 대한 데이터프레임을 입력받아 사전 평가 지표(14~18p)를 계산하여 반환합니다.
        
        :param df_source: 원천 데이터 (단일 상품, 1행)
        :param df_image: 해당 상품의 이미지 데이터 (여러 행 가능)
        :param df_query: 소비자 질문 데이터 (단일 쿼리, 1행)
        :return: df_eval (Page_id와 5가지 핵심 사전 평가 지표 컬럼만 포함된 1행짜리 데이터프레임)
        """
        
        # 1. 단일 상품이므로, df_source의 첫 번째 행 추출
        row = df_source.iloc[0]
        page_id = row['Page_id']
        html_text = str(row['Text_contents']) if pd.notna(row['Text_contents']) else ""
        has_json_ld = bool(row['Has_json_ld']) if pd.notna(row['Has_json_ld']) else False
        
        # 2. 이미지 데이터 (단일 상품에 속한 모든 이미지) 합산
        if not df_image.empty:
            combined_image_text = ' '.join([str(i) for i in df_image['Image_text'] if pd.notna(i)])
            total_images = len(df_image)
            alt_count = df_image['Has_alt'].sum()
        else:
            combined_image_text = ""
            total_images = 0
            alt_count = 0
            
        # 3. 소비자 질문 키워드 확보
        query_keywords = ""
        if not df_query.empty and 'Extracted_keyword' in df_query.columns:
            query_keywords = str(df_query['Extracted_keyword'].iloc[0])
            
        # --- 지표 1) 텍스트 비율 (Text Ratio) ---
        len_html = len(html_text.replace(" ", ""))
        len_img = len(combined_image_text.replace(" ", ""))
        total_len = len_html + len_img
        text_ratio = (len_html / total_len) if total_len > 0 else 0.0
        
        # --- 지표 2) 코사인 유사도 (Cosine Similarity) ---
        full_text = f"{html_text} {combined_image_text}".strip()
        cosine_sim = 0.0
        if full_text and query_keywords:
            # 형태소 분석기를 통해 텍스트 정규화 (형태소 단위로 쪼개서 비교)
            norm_full_text = self.normalize_korean_text(full_text)
            norm_query_keywords = self.normalize_korean_text(query_keywords)
            
            try:
                tfidf_matrix = self.vectorizer.fit_transform([norm_full_text, norm_query_keywords])
                cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            except ValueError:
                pass
        
        # --- 지표 3) 키워드 스터핑 정도 (Keyword Stuffing) - 동적 다중 추출 방식 ---
        keyword_density = 0.0
        if full_text:
            # 특수문자 제거 후 띄어쓰기 기준으로 단어 분리
            clean_text = re.sub(r'[^\w\s]', '', full_text)
            # 길이가 2 이상인 의미 있는 단어들만 추출
            words = [w for w in clean_text.split() if len(w) >= 2]
            
            if len(words) > 0:
                # 가장 많이 등장한 상위 3개 단어 추출 (다중 키워드 도배 방어)
                top_words = Counter(words).most_common(3)
                top_words_count = sum([count for word, count in top_words])
                
                # 상위 3개 키워드의 통합 스터핑 밀도 계산 (100을 곱하지 않고 0.0 ~ 1.0 스케일로 통일)
                keyword_density = top_words_count / len(words)
                
        # --- 지표 4) JSON-LD 구조화 여부 ---
        json_ld_score = bool(has_json_ld) # True or False
        
        # --- 지표 5) 이미지 alt 속성 존재 여부 (이진 분류) ---
        if total_images > 0:
            alt_score = bool(alt_count == total_images)
        else:
            alt_score = True
            
        # --- 머신러닝 Feature Importance 기반 100점 환산 로직 (시뮬레이션) ---
        # 1. 코사인 유사도 (중요도 40%): 0.2 이상이면 만점(40점)
        score_cosine = min((cosine_sim / 0.2) * 40, 40) if cosine_sim > 0 else 0
        
        # 2. 키워드 스터핑 (중요도 20%): 0.05 이하 만점, 0.15 이상 0점 (감점형)
        if keyword_density <= 0.05:
            score_stuffing = 20
        elif keyword_density >= 0.15:
            score_stuffing = 0
        else:
            # 0.05 초과 0.15 미만 구간 (0.1 당 20점 감점 -> * 200)
            score_stuffing = 20 - ((keyword_density - 0.05) * 200)
            
        # 3. 텍스트 비율 (중요도 15%): 0.3 이상이면 만점
        score_text_ratio = min((text_ratio / 0.3) * 15, 15)
        
        # 4. JSON-LD (중요도 15%): 존재하면 15점
        score_json = 15 if json_ld_score else 0
        
        # 5. Alt 태그 (중요도 10%): 존재하면 10점
        score_alt = 10 if alt_score else 0
        
        total_score = round(score_cosine + score_stuffing + score_text_ratio + score_json + score_alt, 1)

        result_df = pd.DataFrame([{
            'Page_id': df_source['Page_id'].iloc[0],
            'Text_Ratio': round(text_ratio, 4),
            'Cosine_Similarity': round(cosine_sim, 4),
            'Keyword_Stuffing': round(keyword_density, 2),
            'Has_Json_Ld': json_ld_score,
            'Has_Alt_Text': alt_score,
            'Total_Score': total_score  # 100점 만점 최종 점수
        }])
        
        return result_df

if __name__ == "__main__":
    import json
    
    df_query_dummy = pd.DataFrame({
        'Query_id': [1],
        'Query_text': ['여름에 땀 안차고 시원하게 입을 수 있는 오버핏 냉감 반팔티 추천해주세요'],
        'Extracted_keyword': ['여름, 시원한, 냉감, 오버핏, 반팔티, 추천, 땀, 흡수, 쿨링']
    })
    
    # target_keyword 매개변수가 완전히 삭제되었습니다!
    evaluator = GeoPreEvaluatorPandas()
    
    print("=== [상품 1: 대기업 (탑텐)] ===")
    df_source_topten = pd.DataFrame({
        'Page_id': [1],
        'Brand_name': ['탑텐 (대기업)'],
        'Brand_type': ['대기업'],
        'Product_name': ['쿨에어 냉감 오버핏 반팔티'],
        'Text_contents': ['무더운 여름, 땀을 빠르게 흡수하고 건조시키는 고기능성 쿨링 원단을 적용한 냉감 오버핏 반팔 티셔츠입니다. 뛰어난 통기성과 부드러운 촉감으로 한여름에도 시원하고 쾌적한 착용감을 선사합니다.'],
        'Raw_html': ['<html>...</html>'],
        'Has_json_ld': [True],
        'Json_ld_contents': ['{"@type": "Product", "name": "쿨에어 냉감 반팔티"}']
    })
    df_image_topten = pd.DataFrame({
        'Image_id': [101, 102],
        'Page_id': [1, 1],
        'Image_sequence': [1, 2],
        'Image_text': ['탑텐 쿨에어 기능성 냉감 소재 흡한속건', '상세 사이즈 가이드 (S, M, L, XL)'],
        'Image_url': ['url1', 'url2'],
        'Has_alt': [True, False],
        'alt_contents': ['여름용 냉감 오버핏 반팔티 착용샷', None]
    })
    
    res_topten = evaluator.evaluate_single_product(df_source_topten, df_image_topten, df_query_dummy)
    print("-> 텍스트 본문:", df_source_topten['Text_contents'].iloc[0])
    print("-> 평가 결과:\n", json.dumps(res_topten.iloc[0].to_dict(), indent=2, ensure_ascii=False))
    
    
    print("\n=== [상품 2: 소상공인 (동대문몰)] ===")
    df_source_small = pd.DataFrame({
        'Page_id': [2],
        'Brand_name': ['동대문개인몰 (소상공인)'],
        'Brand_type': ['소상공인'],
        'Product_name': ['무지티셔츠 남녀공용 빅사이즈'],
        # 소상공인이 '무지티'라는 단어를 5번이나 반복해서 사용함
        'Text_contents': ['무지티 반팔 무지티셔츠 빅사이즈 무지티. 땀흡수 잘됨 무지티 시원함. 퀄리티 좋은 무지티 무지티 무지티셔츠.'],
        'Raw_html': ['<html>...</html>'],
        'Has_json_ld': [False],
        'Json_ld_contents': [None]
    })
    df_image_small = pd.DataFrame({
        'Image_id': [103],
        'Page_id': [2],
        'Image_sequence': [1],
        'Image_text': ['최고의 무지티 한정특가 무지티셔츠'],
        'Image_url': ['url3'],
        'Has_alt': [False],
        'alt_contents': [None]
    })
    
    res_small = evaluator.evaluate_single_product(df_source_small, df_image_small, df_query_dummy)
    print("-> 텍스트 본문:", df_source_small['Text_contents'].iloc[0])
    print("-> 평가 결과:\n", json.dumps(res_small.iloc[0].to_dict(), indent=2, ensure_ascii=False))
