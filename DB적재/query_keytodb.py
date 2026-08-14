import os
import json
import pandas as pd
from pathlib import Path
import requests
from urllib.parse import quote_plus
from sqlalchemy import create_engine, text

# 1. 환경변수 로드 (.env 파싱)
CURRENT_DIR = Path(__file__).resolve().parent
ENV_PATH = CURRENT_DIR / "geo.env"

if ENV_PATH.exists():
    with open(ENV_PATH, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

# 비밀번호 특수문자 안전 인코딩 및 Engine 생성
encoded_password = quote_plus(DB_PASSWORD) if DB_PASSWORD else ""
ENGINE_URL = f"mysql+pymysql://{DB_USER}:{encoded_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(ENGINE_URL)

# 기존
# LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"

# 이미지의 엔드포인트 주소로 변경
LM_STUDIO_URL = "http://192.168.55.242:1234/v1/chat/completions"

# 2. 시스템 프롬프트 정의
SYSTEM_PROMPT = """너는 쇼핑몰 DB 검색을 위한 핵심 키워드 및 카테고리 추출기야.
사용자의 질문을 분석하여 DB 조회에 필요한 대분류 카테고리와 검색 키워드를 JSON 형식으로 추출해줘.

[대분류(category) 선택 기준]
다음 6개 대분류 중 질문과 가장 잘 맞는 1개를 선택할 것:
- "상의" (티셔츠, 셔츠, 블라우스, 맨투맨, 후드, 니트 등)
- "바지" (슬랙스, 청바지, 팬츠, 반바지, 트레이닝 바지 등)
- "치마" (스커트, 롱스커트, 미니스커트, 롱치마 등)
- "아우터" (자켓, 코트, 패딩, 가디건, 점퍼, 집업 등)
- "세트" (원피스, 드레스, 투피스, 트레이닝세트, 파자마세트 등)
- "기타" (가방, 신발, 모자, 악세서리, 속옷, 수영복 등)

[키워드 추출 규칙]
1. 'must_have': 필수 핵심 품목 단어. 동일 품목을 뜻하는 유의어/동의어는 반드시 같이 must_have에 포함할 것.
   - 예: 민소매/나시 -> ["민소매", "나시", "티셔츠"]
   - 예: 슬랙스 -> ["슬랙스", "팬츠"]

2. 'options': 핏(오버핏, 슬림핏 등), 소재(린넨, 데님 등), 디자인 패턴(골지, 단가라 등) 특성 명사만 추출할 것.
   - 예: ["오버핏"]
3. 엄격한 금지 규칙 (위반 시 시스템 오류 발생):
    - 영문 혼용 및 알파벳 표기 금지: 알파벳(영어) 직접 작성이나 한영 병기('린 linen') 절대 금지
    (단, '팬츠', '스커트', '린넨', '오버핏' 등 한국어로 표기된 외래어는 사용 가능)
   - 단어 절단 금지 ('오버' X -> '오버핏' O)
   - 연령/성별 수식어 제외 (여성용, 20대 등 X)

[출력 양식]
다른 설명이나 인사말 없이 오직 아래 JSON 형식으로만 출력할 것:
{
  "category": "선택한 대분류",
  "must_have": ["핵심품목1", "직접유의어2"],
  "options": ["옵션1", "옵션2"]
}"""

# 3. LM Studio 호출 함수
def call_lm_studio(user_query: str) -> dict:
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "local-model",  # LM Studio 400 Client Error 방지용 필드 명시
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query}
        ],
        "temperature": 0.1,
        #"response_format": {"type": "json_object"}
    }
    
    try:
        response = requests.post(LM_STUDIO_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        content = response.json()['choices'][0]['message']['content']
        
        parsed_json = json.loads(content)
        return parsed_json
    except Exception as e:
        print(f"LM Studio 호출 및 파싱 오류 ('{user_query}'): {e}")
        return {"category": "기타", "must_have": [], "options": []}

# 4. MySQL 적재 메인 함수
def process_and_insert_queries(csv_filepath: str):
    # 기존대로 첫 줄을 헤더로 읽어옴
    df = pd.read_csv(csv_filepath)
    question_column = df.columns[0]
    questions = df[question_column].dropna().tolist()

    # SQLAlchemy 실행을 위해 text() 객체 및 네임드 파라미터(:변수명) 사용
    insert_sql = text("""
        INSERT INTO question_table (query_text, query_cat, query_keyword, query_option)
        VALUES (:query_text, :query_cat, :query_keyword, :query_option)
    """)

    print(f"총 {len(questions)}개의 질문 처리를 시작합니다...")

    with engine.begin() as conn:
        for idx, q_text in enumerate(questions, 1):
            result = call_lm_studio(q_text)
            
            query_cat = result.get("category", "기타")
            query_keyword = json.dumps(result.get("must_have", []), ensure_ascii=False)
            query_option = json.dumps(result.get("options", []), ensure_ascii=False)
            
            # 파라미터 딕셔너리 전달
            conn.execute(insert_sql, {
                "query_text": q_text,
                "query_cat": query_cat,
                "query_keyword": query_keyword,
                "query_option": query_option
            })
            
            if idx % 10 == 0 or idx == len(questions):
                print(f"[{idx}/{len(questions)}] 진행 완료")
                
    print("모든 질문의 파싱 및 DB 적재가 완료되었습니다.")

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    csv_path = BASE_DIR / "gemini_collected_data.csv"
    
    process_and_insert_queries(str(csv_path))