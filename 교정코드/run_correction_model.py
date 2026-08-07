import os
import json
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

# ==========================================
# 설정 정보
# ==========================================
INPUT_CSV = r"입력값(다운받은파일경로).csv"
OUTPUT_DIR = r"저장경로"
LM_STUDIO_URL = "http://localhost:1234/v1"
MODEL_NAME = "qwen-7b"  # LM Studio에 로드된 모델명
MAX_ITEMS = 50

# ==========================================
# 데이터 및 프롬프트 로드
# ==========================================
print("데이터와 프롬프트를 불러옵니다...")
df = pd.read_csv(INPUT_CSV)

# 상품명(title) 기준 가나다순 정렬 후 상위 100개 추출
df = df.sort_values(by='title', ascending=True).head(MAX_ITEMS)
# 결측치(NaN) 빈 문자열로 처리
df = df.fillna("")

# LM Studio Client 초기화
client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")

# ==========================================
# ==========================================
# 기준 조합 순차 처리
# ==========================================
# JSON 파일 로드 (Combo)
COMBO_PROMPTS_FILE = r"프롬프트경로.json"

try:
    with open(COMBO_PROMPTS_FILE, "r", encoding="utf-8") as f:
        combo_prompts = json.load(f)
except FileNotFoundError:
    print(f"[{COMBO_PROMPTS_FILE}] 파일을 찾을 수 없습니다. 경로를 확인해 주세요.")
    combo_prompts = {}

for combo_key, system_prompt in combo_prompts.items():
    print(f"\n==========================================")
    print(f"기준 조합 [{combo_key}] 처리를 시작합니다...")
    print(f"==========================================")
    
    # 파일명 예시: 결과-라마_1+3.csv
    output_filename = f"결과-라마_{combo_key}.csv"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    results = []
    processed_titles = set()
    
    # 이어하기(Resume) 기능: 기존 파일이 존재하면 불러와서 이미 처리된 항목은 건너뜀
    if os.path.exists(output_path):
        try:
            existing_df = pd.read_csv(output_path)
            # 에러가 발생했던 건 다시 시도하기 위해 'ERROR:'로 시작하지 않는 정상 항목만 인정
            valid_existing = existing_df[~existing_df['corrected_result'].astype(str).str.startswith("ERROR:", na=False)]
            processed_titles = set(valid_existing['title'].tolist())
            results = valid_existing.to_dict('records')
            if len(processed_titles) > 0:
                print(f"👉 [{output_filename}] 이전에 성공적으로 처리된 {len(processed_titles)}개의 데이터를 불러와 이어서 진행합니다.")
        except Exception as e:
            print(f"기존 파일을 읽는 중 오류가 발생했습니다. 새로 시작합니다: {e}")
    
    # tqdm으로 진행률 표시
    for index, row in tqdm(df.iterrows(), total=len(df), desc=f"조합 [{combo_key}] 진행중"):
        # 이미 처리된 상품은 건너뛰기
        if row['title'] in processed_titles:
            continue
            
        # 유저에게 전달할 데이터 포맷팅
        user_content = json.dumps({
            "title": row['title'],
            "text": row['text'],
            "jsonld": row['jsonld'],
            "alt": row['alt']
        }, ensure_ascii=False, indent=2)
        
        try:
            # API 요청
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.7,
                max_tokens=4096,
            )
            
            corrected_text = response.choices[0].message.content
            
            # 결과 저장용 딕셔너리 구성
            result_item = row.to_dict()
            result_item['corrected_result'] = corrected_text
            results.append(result_item)
            
        except Exception as e:
            # 에러 발생 시 건너뛰기
            print(f"\n[Error] '{row['title']}' 처리 중 에러 발생: {e}")
            result_item = row.to_dict()
            result_item['corrected_result'] = f"ERROR: {e}"
            results.append(result_item)
            
        # 매 상품마다 파일 덮어쓰기 형태로 실시간 저장 (오류가 났을 때도 저장됨)
        try:
            pd.DataFrame(results).to_csv(output_path, index=False, encoding='utf-8-sig')
        except PermissionError:
            # 엑셀 등 다른 프로그램에서 파일을 열고 있어서 저장이 실패한 경우 무시하고 넘어갑니다.
            pass
            
    try:
        pd.DataFrame(results).to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n✅ 기준 조합 [{combo_key}] 처리가 완료되어 {output_path}에 최종 저장되었습니다.")
    except PermissionError:
        print(f"\n⚠️ 기준 조합 [{combo_key}] 처리가 완료되었으나, 엑셀 등에서 파일({output_filename})이 열려 있어 최종 저장에 실패했습니다. 파일을 닫아주세요.")

print("\n모든 작업이 완료되었습니다!")
