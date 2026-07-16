"""
Universal OCR Pipeline using Qwen2-VL
이 스크립트는 쇼핑몰 상품 이미지(단일 브랜드 및 다중 카테고리 구조)에서 
텍스트를 추출하기 위한 통합 파이프라인입니다.

[지원하는 데이터 폴더 구조]
1. 카테고리 모드 (CATEGORY_MODE)
   - 구조: dataset/대분류/소분류/상품목록.csv (다중 폴더 재귀 탐색)
   - 이미지: dataset/대분류/소분류/이미지/이미지명.png
2. 브랜드 모드 (BRAND_MODE)
   - 구조: dataset/브랜드명/csv/브랜드명_상품목록.csv
   - 이미지: dataset/브랜드명/이미지명.png
"""

import os
import glob
import time
import base64
import sys
import io
import pandas as pd
from io import BytesIO
from openai import OpenAI

# 콘솔 출력 인코딩 오류 방지 (Jupyter Notebook 등에서 강제 종료 방지용)
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

try:
    from PIL import Image
except ImportError:
    os.system('pip install pillow pandas openai')
    from PIL import Image

# ==========================================
# 1. 사용자 설정 (깃허브 업로드용 템플릿)
# ==========================================
# 여러분의 로컬/서버 데이터셋 경로로 변경하세요.
# 예: "./dataset/CJ_Categories" 또는 "./dataset/Brands"
BASE_DATA_DIR = "./dataset/your_images"

# 실행 모드 선택 ('CATEGORY' 또는 'BRAND')
PIPELINE_MODE = "CATEGORY"

# 로컬 LLM (LM Studio, vLLM 등) API 주소
LM_STUDIO_URL = "http://localhost:1234/v1"
MODEL_NAME = "qwen2-vl-7b-instruct"

# ==========================================
# 2. 이미지 전처리 및 슬라이싱 함수
# ==========================================
def encode_and_slice_image(image_path, max_slice_height=616):
    """
    세로로 긴 상세 페이지 이미지를 모델이 인식하기 좋은 크기로 자릅니다.
    Qwen2-VL 모델 권장 해상도(28의 배수)를 준수합니다.
    """
    base64_list = []
    try:
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # 해상도를 줄여 VRAM 사용량과 처리 속도 최적화
            width, height = img.size
            new_width = 784
            ratio = new_width / float(width)
            new_height = int(height * ratio)
            
            # 28의 배수로 맞춤 (Qwen2-VL Vision Encoder 요구사항)
            new_height = max(28, (new_height // 28) * 28)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            width, height = img.size

            # 이미지가 너무 길면 max_slice_height 기준으로 여러 장으로 분할
            num_slices = (height + max_slice_height - 1) // max_slice_height
            
            for i in range(num_slices):
                top = i * max_slice_height
                bottom = min((i + 1) * max_slice_height, height)
                
                slice_img = img.crop((0, top, width, bottom))
                slice_height = bottom - top
                
                # 잘라낸 조각의 높이도 28의 배수가 되도록 흰색 여백(Padding) 추가
                if slice_height % 28 != 0:
                    pad_height = ((slice_height // 28) + 1) * 28
                    padded_img = Image.new('RGB', (width, pad_height), (255, 255, 255))
                    padded_img.paste(slice_img, (0, 0))
                    slice_img = padded_img

                buffered = BytesIO()
                slice_img.save(buffered, format="JPEG", quality=85)
                b64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                base64_list.append(b64_str)
    except Exception as e:
        print(f"이미지 처리 중 에러 발생 ({image_path}): {e}")
    return base64_list

# ==========================================
# 3. OCR 텍스트 추출 및 필터링 로직
# ==========================================
def extract_text_from_slices(client, slices):
    """
    잘라낸 이미지 조각들을 로컬 AI에 보내서 텍스트를 추출하고,
    환각(Hallucination)이나 무의미한 문장을 필터링합니다.
    """
    extracted_pieces = []
    for b64 in slices:
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME, 
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a professional OCR system. Extract all visible text exactly as it appears. DO NOT describe the image or colors. Output ONLY 'NONE' if no text."
                    },
                    {
                        "role": "user", 
                        "content": [
                            {"type": "text", "text": "Extract all visible text from this image slice."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                        ]
                    }
                ],
                temperature=0.1, # 헛소리(환각) 방지를 위해 창의성 최소화
                top_p=1.0,
                max_tokens=500
            )
            text = response.choices[0].message.content.strip()
            
            # API 과부하 방지 쿨다운
            time.sleep(0.5)
            
            # --- 텍스트 필터링 (환각 및 AI 멘트 제거) ---
            lines = text.split('\n')
            cleaned_lines = []
            for line in lines:
                lower_line = line.strip().lower()
                if not lower_line: continue
                
                # 시각 모델 특유의 '이미지 묘사' 문구 제거
                if lower_line.startswith('the image') or lower_line.startswith('here is'): continue
                if 'the image shows' in lower_line or '이 이미지는' in lower_line: continue
                # 특수문자나 무의미한 기호만 있는 줄 제거
                if set(lower_line) <= set(' ?!-.,_'): continue
                
                cleaned_lines.append(line)
            
            text = '\n'.join(cleaned_lines).strip()
            if text and text not in ['NONE', '없음']:
                extracted_pieces.append(text)
                
        except Exception as e:
            print(f"  -> [에러] 슬라이스 OCR 추출 실패: {e}")
            continue
            
    return "\n".join(extracted_pieces) + "\n"

# ==========================================
# 4. 개별 디렉터리 처리 프로세스
# ==========================================
def process_directory(client, product_csv_path, img_dir, detail_csv_path):
    csv_dir = os.path.dirname(product_csv_path)
    output_csv_path = os.path.join(csv_dir, "상품목록_with_ocr.csv")
    
    # 이어하기(Resume) 기능: 기존 작업 파일이 있으면 그걸 불러와서 이어서 진행
    if os.path.exists(output_csv_path):
        df_products = pd.read_csv(output_csv_path)
    else:
        df_products = pd.read_csv(product_csv_path)
        
    df_details = pd.read_csv(detail_csv_path)

    # 텍스트 결과 저장용 컬럼 초기화
    if 'Text_contents' not in df_products.columns:
        df_products['Text_contents'] = ""

    for idx, row in df_products.iterrows():
        existing_text = str(row.get('Text_contents', ''))
        
        # 이미 텍스트가 성공적으로 추출된 상품은 건너뛰기
        if existing_text and existing_text.lower() != 'nan' and existing_text != 'ERROR_OR_EMPTY':
            print(f"[{idx+1}/{len(df_products)}] 패스 (이미 완료됨) - {row['상품명']}")
            continue
            
        product_name = row['상품명']
        print(f"\n[{idx+1}/{len(df_products)}] 상품명: {product_name} 처리 중...")
        
        # 해당 상품에 매칭되는 상세 이미지들만 불러오기
        product_details = df_details[df_details['상품명'] == product_name].sort_values(by='상세이미지순번')
        
        full_extracted_text = ""
        total_images = len(product_details)
        
        for img_idx, (_, detail_row) in enumerate(product_details.iterrows(), 1):
            # 파일명 매칭: '이미지파일명' 컬럼이 있으면 우선 사용, 없으면 URL에서 파싱
            if '이미지파일명' in detail_row and pd.notna(detail_row['이미지파일명']):
                file_name = str(detail_row['이미지파일명'])
            else:
                img_url = str(detail_row.get('상세이미지주소링크', ''))
                file_name = img_url.split('/')[-1]
                
            local_image_path = os.path.join(img_dir, file_name)
            
            if not os.path.exists(local_image_path):
                print(f"  -> [{img_idx}/{total_images}] 파일 누락 스킵 ({file_name})")
                continue
                
            print(f"  -> [{img_idx}/{total_images}] OCR 추출 진행 중... ({file_name})")
            
            # 1. 이미지 슬라이싱
            slices = encode_and_slice_image(local_image_path)
            if not slices: continue
            
            # 2. 텍스트 추출 및 병합
            extracted_text = extract_text_from_slices(client, slices)
            if extracted_text.strip():
                full_extracted_text += extracted_text

        # 3. 추출된 전체 텍스트를 원본 데이터프레임에 업데이트
        if full_extracted_text.strip():
            df_products.at[idx, 'Text_contents'] = full_extracted_text.strip()
        else:
            df_products.at[idx, 'Text_contents'] = "ERROR_OR_EMPTY"
            
        # 작업 안전성을 위해 상품 1개 완료시마다 실시간으로 덮어쓰기 저장 (중단 대비)
        try:
            df_products.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        except PermissionError:
            # 사용자가 엑셀 파일을 켜둬서 저장이 안 될 경우 백업 파일로 저장
            backup_path = output_csv_path.replace('.csv', '_backup.csv')
            df_products.to_csv(backup_path, index=False, encoding='utf-8-sig')
            print(f"  -> [!] 엑셀 파일이 켜져 있어 백업 파일로 저장했습니다.")
            
    print(f"\n[*] 처리 완료! 결과물: {output_csv_path}")

# ==========================================
# 5. 메인 함수 (폴더 구조별 분기 처리)
# ==========================================
def main():
    client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")
    
    if PIPELINE_MODE == "CATEGORY":
        # 카테고리 모드: 하위 폴더의 모든 '상품목록.csv'를 재귀적으로 탐색 (예: CJ온스타일)
        product_csvs = glob.glob(os.path.join(BASE_DATA_DIR, "**", "상품목록.csv"), recursive=True)
        if not product_csvs:
            print(f"[!] '{BASE_DATA_DIR}' 내에 '상품목록.csv'가 없습니다. 경로를 확인하세요.")
            return
            
        print(f"[*] 총 {len(product_csvs)}개의 카테고리를 발견했습니다.")
        for product_csv in product_csvs:
            csv_dir = os.path.dirname(product_csv)
            
            # 이미지 폴더 찾기 분기 ('이미지' 또는 '상세이미지' 폴더)
            img_dir = os.path.join(csv_dir, "이미지")
            if not os.path.exists(img_dir):
                img_dir = os.path.join(csv_dir, "상세이미지")
                
            # 상세이미지 CSV 찾기
            detail_csv = os.path.join(csv_dir, "상세이미지목록.csv")
            if not os.path.exists(detail_csv):
                detail_csv = os.path.join(csv_dir, "상세이미지.csv")
                
            if os.path.exists(detail_csv):
                process_directory(client, product_csv, img_dir, detail_csv)
                
    elif PIPELINE_MODE == "BRAND":
        # 단일 브랜드 모드: 특정 브랜드 폴더 내의 CSV 처리 (예: 미니포에)
        csv_dir = os.path.join(BASE_DATA_DIR, "csv")
        product_csvs = glob.glob(os.path.join(csv_dir, "*_상품목록.csv"))
        
        if not product_csvs:
            print(f"[!] '{csv_dir}' 내에 상품목록 CSV가 없습니다.")
            return
            
        product_csv = product_csvs[0]
        # 파일명에서 브랜드명 추출 (예: 미니포에_상품목록.csv -> 미니포에)
        brand_name = os.path.basename(product_csv).split("_")[0]
        detail_csv = os.path.join(csv_dir, f"{brand_name}_상세이미지목록.csv")
        
        if os.path.exists(detail_csv):
            # 브랜드 모드는 보통 BASE_DATA_DIR 자체가 이미지 폴더인 경우가 많음
            process_directory(client, product_csv, BASE_DATA_DIR, detail_csv)

    print("\n[*] 모든 OCR 파이프라인 처리가 완료되었습니다!")

if __name__ == "__main__":
    main()

#혹시 궁금한거 있으면 알려주세요. 이 코드로 돌린건 아니라 오류 있을 수 있습니다.