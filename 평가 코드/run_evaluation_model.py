import pymysql
import json
import os
import random
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

# ==========================================
# 1. 환경 설정 (인수인계 시 아래 경로 및 DB 정보를 사용자 환경에 맞게 수정해주세요)
# ==========================================
# MySQL 데이터베이스 접속 정보
DB_CONFIG = {
    'host': '211.212.230.86',
    'user': 'teamuser',
    'password': '1638',
    'database': 'geo_db',
    'port': 3306,
    'cursorclass': pymysql.cursors.DictCursor
}

# 평가 기준 프롬프트 조합이 들어있는 JSON 파일 경로를 입력하세요.
EVAL_PROMPTS_FILE = r"평가_프롬프트_JSON_파일_경로를_여기에_입력하세요.json"

# 평가 결과 CSV 파일이 저장될 폴더 경로를 입력하세요.
OUTPUT_DIR = r"저장할_폴더_경로를_여기에_입력하세요"

# LM Studio 로컬 서버 주소 (보통 기본값과 동일하므로 수정하지 않아도 됩니다)
LM_STUDIO_URL = "http://localhost:1234/v1"

# LM Studio에 로드된 평가용 모델명을 기재해주세요
MODEL_NAME = "모델명을_여기에_입력하세요"  # 예: gemma-4-12b-it

# 한 번에 평가할 상품 개수 설정 (대기업 5개 + 소상공인 5개 = 총 10개)
# 주의: 무리 가지 않는 선에서 조절 (초기 세팅: 10개)
ITEMS_PER_BRAND_TYPE = 5 

# 한 평가 기준(콤보)당 평가할 총 질문 개수
MAX_EVAL_QUESTIONS = 5

# ==========================================
# 2. DB 검색 보조 함수 (필수 키워드 및 옵션 가산점 적용)
# ==========================================
def fetch_products(cursor, query_cat, query_keywords, query_options, brand_type, limit):
    """
    1. 대분류(query_cat)와 필수 키워드(query_keywords)는 무조건 포함 (WHERE 절)
    2. 옵션(query_options)은 있으면 가산점 (ORDER BY 절에서 일치 개수순으로 정렬)
    3. 대기업/소상공인 비율 맞추기 위해 brand_type 별로 각각 호출
    """
    # 필수 조건 쿼리 (대분류 일치 + 텍스트에 필수 키워드 포함)
    base_query = f"SELECT page_id, brand_name, product_name, text_contents, json_ld_contents FROM raw_data_table WHERE brand_type = %s AND product_cat = %s"
    params = [brand_type, query_cat]
    
    for kw in query_keywords:
        base_query += " AND (product_name LIKE %s OR text_contents LIKE %s)"
        params.extend([f"%{kw}%", f"%{kw}%"])
        
    # 옵션 가산점 쿼리 (ORDER BY에 옵션 매칭 횟수 합산)
    if query_options:
        option_cases = []
        for opt in query_options:
            option_cases.append(f"(IF(product_name LIKE %s OR text_contents LIKE %s, 1, 0))")
            params.extend([f"%{opt}%", f"%{opt}%"])
            
        bonus_score_sql = " + ".join(option_cases)
        base_query += f" ORDER BY ({bonus_score_sql}) DESC"
    else:
        # 옵션이 없으면 순차 정렬
        base_query += " ORDER BY page_id ASC"
        
    base_query += f" LIMIT {limit}"
    
    cursor.execute(base_query, params)
    return cursor.fetchall()

def get_alt_texts(cursor, page_id):
    """
    이미지 텍스트(OCR)는 제외하고 순수 alt 속성만 가져옵니다.
    image_data_table에서 해당 page_id의 alt_contents를 수집
    """
    cursor.execute("SELECT alt_contents FROM image_data_table WHERE page_id = %s AND has_alt = 1", (page_id,))
    rows = cursor.fetchall()
    return " | ".join([r['alt_contents'] for r in rows if r['alt_contents']])

# ==========================================
# 3. 메인 실행 로직
# ==========================================
def main():
    # 결과 폴더 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("시스템 프롬프트를 불러옵니다...")
    try:
        with open(EVAL_PROMPTS_FILE, 'r', encoding='utf-8') as f:
            eval_prompts = json.load(f)
    except FileNotFoundError:
        print(f"[{EVAL_PROMPTS_FILE}] 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        return
        
    # LM Studio API 연결
    client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")
    
    # DB 연결
    connection = pymysql.connect(**DB_CONFIG)
    
    try:
        with connection.cursor() as cursor:
            # 1. 수백 개의 질문 중 무작위로 가져오기 (나중에 MAX_EVAL_QUESTIONS 만큼 자름)
            cursor.execute("SELECT * FROM question_table ORDER BY RAND()")
            questions = list(cursor.fetchall())
            
            for combo_key, system_prompt in eval_prompts.items():
                if combo_key == "_comment":
                    continue
                    
                print(f"\n==========================================")
                print(f"평가 기준 [{combo_key}] 처리를 시작합니다...")
                print(f"==========================================")
                
                # 파일명 특수문자 안전하게 치환
                output_filename = f"결과-평가모델_gemma_2차{combo_key}.csv"
                output_filename = output_filename.replace('+', '_').replace(' ', '_').replace('·', '')
                output_path = os.path.join(OUTPUT_DIR, output_filename)
                
                results = []
                processed_query_ids = set()
                
                # 이어하기(Resume) 기능
                if os.path.exists(output_path):
                    try:
                        existing_df = pd.read_csv(output_path)
                        # 에러난 항목 제외하고 정상 결과만 유지
                        valid_existing = existing_df[~existing_df['evaluation_result'].astype(str).str.startswith("ERROR:", na=False)]
                        processed_query_ids = set(str(x) for x in valid_existing['query_id'].tolist())
                        results = valid_existing.to_dict('records')
                        if len(processed_query_ids) > 0:
                            print(f"👉 [{output_filename}] 이전에 처리된 {len(processed_query_ids)}개의 데이터를 불러와 이어서 진행합니다.")
                    except Exception as e:
                        print(f"기존 파일을 읽는 중 오류가 발생했습니다. 새로 시작합니다: {e}")
                        
                # 아직 처리 안 된 질문들만 필터링 (이어하기 위함)
                target_questions = [q for q in questions if str(q['query_id']) not in processed_query_ids]
                
                # 목표 개수(MAX_EVAL_QUESTIONS)까지만 잘라서 진행 (무한 증식 방지)
                needed_count = MAX_EVAL_QUESTIONS - len(processed_query_ids)
                if needed_count <= 0:
                    print(f"✅ 해당 기준은 이미 {MAX_EVAL_QUESTIONS}개의 평가가 완료되었습니다. 건너뜁니다.")
                    continue
                    
                target_questions = target_questions[:needed_count]
                
                for q in tqdm(target_questions, desc="질문 처리 중"):
                    q_id = q['query_id']
                    q_text = q['query_text']
                    q_cat = q['query_cat']
                    q_keywords = json.loads(q['query_keyword']) if q['query_keyword'] else []
                    q_options = json.loads(q['query_option']) if q['query_option'] else []
                    
                    # 2. 대기업 5개 / 소상공인 5개 가져오기
                    corp_products = fetch_products(cursor, q_cat, q_keywords, q_options, '대기업', ITEMS_PER_BRAND_TYPE)
                    small_products = fetch_products(cursor, q_cat, q_keywords, q_options, '소상공인', ITEMS_PER_BRAND_TYPE)
                    all_candidates = corp_products + small_products
                    
                    if not all_candidates:
                        # 조건에 맞는 상품이 하나도 없는 경우 AI를 호출하지 않고 에러 처리
                        results.append({
                            "query_id": q_id,
                            "query_text": q_text,
                            "input_page_ids": "",
                            "evaluation_result": "ERROR: 검색된 후보 상품이 없습니다."
                        })
                        continue
                    
                    # 3. 데이터 포맷팅
                    candidate_str_list = []
                    for idx, prod in enumerate(all_candidates):
                        alt_text = get_alt_texts(cursor, prod['page_id'])
                        
                        # AI 평가를 위한 후보 상품 문자열 조립
                        candidate_info = (
                            f"후보 {idx+1}:\n"
                            f"- 고유번호(page_id): {prod['page_id']}\n"
                            f"- 브랜드명: {prod['brand_name']}\n"
                            f"- 상품명: {prod['product_name']}\n"
                            f"- 본문 텍스트: {prod['text_contents']}\n"
                            f"- JSON-LD: {prod['json_ld_contents']}\n"
                            f"- Alt 속성: {alt_text}\n"
                        )
                        candidate_str_list.append(candidate_info)
                        
                    final_user_content = (
                        f"소비자 질문: {q_text}\n\n"
                        f"[검색된 상품 후보 리스트]\n"
                        + "\n".join(candidate_str_list)
                    )
                    
                    # 4. LM Studio 평가 AI 실행
                    try:
                        response = client.chat.completions.create(
                            model=MODEL_NAME,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": final_user_content}
                            ],
                            temperature=0.1,  # 평가 모델은 일관성이 중요하므로 보수적으로 설정
                            max_tokens=4096,
                        )
                        eval_result = response.choices[0].message.content
                    except Exception as e:
                        eval_result = f"ERROR: {e}"
                        
                    # input_page_ids를 기록하여 추후 로깅에 사용
                    input_page_ids = ",".join([str(p['page_id']) for p in all_candidates])
                    results.append({
                        "query_id": q_id,
                        "query_text": q_text,
                        "input_page_ids": input_page_ids,
                        "evaluation_result": eval_result
                    })
                    
                    # 매 질문마다 실시간 덮어쓰기 저장
                    try:
                        pd.DataFrame(results).to_csv(output_path, index=False, encoding='utf-8-sig')
                    except PermissionError:
                        pass
                        
                # 콤보별 완료 메시지
                print(f"✅ 해당 기준 평가 및 저장 완료: {output_path}")

    finally:
        connection.close()

if __name__ == '__main__':
    main()
