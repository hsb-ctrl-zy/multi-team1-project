import os
import json
import pymysql
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

# ==========================================
# 1. 설정 정보 (인수인계 시 아래 경로 및 DB 정보를 사용자 환경에 맞게 수정해주세요)
# ==========================================
# 교정 결과 CSV 파일이 저장될 폴더 경로를 입력하세요. (예: r"C:\Users\홍길동\Desktop\결과폴더")
OUTPUT_DIR = r"저장할_폴더_경로를_여기에_입력하세요"

# 프롬프트 조합이 들어있는 JSON 파일 경로를 입력하세요.
COMBO_PROMPTS_FILE = r"시스템_프롬프트_JSON_파일_경로를_여기에_입력하세요.json"

# LM Studio 로컬 서버 주소 (보통 기본값과 동일하므로 수정하지 않아도 됩니다)
LM_STUDIO_URL = "http://localhost:1234/v1"

# LM Studio에 로드된 모델명을 기재해주세요 (예: gemma-2-9b, qwen-7b 등)
MODEL_NAME = "모델명을_여기에_입력하세요"  

# 한 번에 처리할 최대 상품 개수를 설정합니다. (원하는 개수로 수정 가능)
MAX_ITEMS = 20

# MySQL 데이터베이스 접속 정보
DB_CONFIG = {
    'host': '데이터베이스_IP주소_입력',
    'user': '데이터베이스_아이디_입력',
    'password': '데이터베이스_비밀번호_입력',
    'database': '데이터베이스_이름_입력',
    'port': 3306,
    'cursorclass': pymysql.cursors.DictCursor
}

# ==========================================
# 2. 보조 함수
# ==========================================
def get_alt_texts(cursor, page_id):
    """
    image_data_table에서 특정 상품(page_id)의 대체 텍스트(alt_contents)를 모두 가져와
    ' | ' 기호로 연결해주는 함수입니다.
    """
    cursor.execute("SELECT alt_contents FROM image_data_table WHERE page_id = %s AND has_alt = 1", (page_id,))
    rows = cursor.fetchall()
    return " | ".join([r['alt_contents'] for r in rows if r['alt_contents']])

# ==========================================
# 3. 메인 실행 로직
# ==========================================
def main():
    # 결과물을 저장할 폴더가 없다면 새로 만듭니다.
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 콤보 프롬프트 JSON 파일을 불러옵니다.
    try:
        with open(COMBO_PROMPTS_FILE, "r", encoding="utf-8") as f:
            combo_prompts = json.load(f)
    except FileNotFoundError:
        print(f"[{COMBO_PROMPTS_FILE}] 파일을 찾을 수 없습니다. 경로를 다시 확인해주세요.")
        return

    # OpenAI 라이브러리를 사용해 로컬 LM Studio와 연결합니다.
    client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")
    
    # DB 접속 시작
    connection = pymysql.connect(**DB_CONFIG)
    
    try:
        with connection.cursor() as cursor:
            print(f"데이터베이스에서 교정 대상 상품(최대 {MAX_ITEMS}개)을 불러옵니다...")
            # 대기업 브랜드는 제외하고(소상공인만 타겟팅), 상품명(product_name) 가나다순으로 정렬하여 불러옵니다.
            cursor.execute("SELECT page_id, product_name, text_contents, image_text, json_ld_contents FROM raw_data_table WHERE brand_type != '대기업' OR brand_type IS NULL ORDER BY product_name LIMIT %s", (MAX_ITEMS,))
            products = cursor.fetchall()
            
            # JSON에 정의된 여러 프롬프트 조합을 순회하며 교정을 진행합니다.
            for combo_key, system_prompt in combo_prompts.items():
                if combo_key == "_comment":  # 주석용 데이터는 건너뜁니다.
                    continue
                    
                print(f"\n==========================================")
                print(f"기준 조합 [{combo_key}] 처리를 시작합니다...")
                print(f"==========================================")
                
                # 저장될 CSV 파일 이름 지정 (원하는 파일명 규칙으로 수정 가능합니다)
                output_filename = f"결과-gemma2차(0.3)_{combo_key}.csv"
                output_path = os.path.join(OUTPUT_DIR, output_filename)
                
                results = []
                processed_page_ids = set()
                
                # 이어하기(Resume) 기능: 파일이 이미 있다면 처리된 내역을 불러와서 중복 처리를 방지합니다.
                if os.path.exists(output_path):
                    try:
                        existing_df = pd.read_csv(output_path)
                        # 에러가 난 항목은 재시도를 위해 제외하고, 정상 처리된 항목만 추출합니다.
                        valid_existing = existing_df[~existing_df['corrected_result'].astype(str).str.startswith("ERROR:", na=False)]
                        
                        # 과거 데이터 호환: 기존 결과에 page_id가 없는 경우 DB 상품명으로 역추적하여 매핑합니다.
                        results = valid_existing.to_dict('records')
                        title_to_id = {p['product_name']: p['page_id'] for p in products}
                        
                        for r in results:
                            if pd.isna(r.get('page_id')) or r.get('page_id') is None:
                                r['page_id'] = title_to_id.get(r['title'])
                                
                        processed_page_ids = set(r['page_id'] for r in results if r['page_id'] is not None)
                        
                        if len(processed_page_ids) > 0:
                            print(f"👉 [{output_filename}] 이전에 성공적으로 처리된 {len(processed_page_ids)}개의 데이터를 불러와 이어서 진행합니다.")
                    except Exception as e:
                        print(f"기존 CSV 파일을 읽는 중 오류가 발생했습니다. 새로 시작합니다: {e}")
                
                # 진행률(%)을 표시하며 상품 단위로 교정 진행
                for prod in tqdm(products, desc=f"조합 [{combo_key}] 진행중"):
                    page_id = prod['page_id']
                    
                    # 이미 처리된 상품(이어하기)은 건너뜁니다.
                    if page_id in processed_page_ids:
                        continue
                        
                    # 본문 텍스트와 이미지 속 텍스트(OCR)를 합칩니다.
                    text_str = str(prod['text_contents'] or '')
                    img_txt_str = str(prod['image_text'] or '')
                    combined_text = (text_str + "\n" + img_txt_str).strip()
                    
                    # 해당 상품의 대체 텍스트(alt_contents)를 가져옵니다.
                    alt_text = get_alt_texts(cursor, page_id)
                    
                    # AI에게 전달할 최종 페이로드(입력값) 조합
                    user_content = json.dumps({
                        "title": prod['product_name'],
                        "text": combined_text,
                        "jsonld": prod['json_ld_contents'],
                        "alt": alt_text
                    }, ensure_ascii=False, indent=2)
                    
                    try:
                        # AI 모델에 교정 요청
                        response = client.chat.completions.create(
                            model=MODEL_NAME,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_content}
                            ],
                            temperature=0.3, # 모델의 창의성 (낮을수록 보수적이고 일관된 답변)
                            max_tokens=4096,
                        )
                        corrected_text = response.choices[0].message.content
                            
                    except Exception as e:
                        print(f"\n[Error] '{prod['product_name']}' 처리 중 에러 발생: {e}")
                        corrected_text = f"ERROR: {e}"
                        
                    # CSV 파일 한 줄(행)에 들어갈 데이터 구성
                    result_item = {
                        "page_id": page_id,
                        "title": prod['product_name'],
                        "text": combined_text,
                        "jsonld": prod['json_ld_contents'],
                        "alt": alt_text,
                        "corrected_result": corrected_text
                    }
                    results.append(result_item)
                    
                    # 만약 스크립트가 강제 종료되더라도 날아가지 않도록 매 상품마다 실시간 CSV 덮어쓰기 저장
                    try:
                        pd.DataFrame(results).to_csv(output_path, index=False, encoding='utf-8-sig')
                    except PermissionError:
                        # 엑셀로 해당 파일을 열어보고 있을 때 발생하는 에러 무시
                        pass
                        
                print(f"\n✅ 기준 조합 [{combo_key}] 처리가 완료되어 {output_path}에 저장되었습니다.")
                
    finally:
        # 모든 작업이 끝나면 안전하게 DB 연결을 종료합니다.
        connection.close()

    print("\n모든 작업이 완료되었습니다!")

if __name__ == '__main__':
    main()
