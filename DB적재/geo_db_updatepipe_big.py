import json
import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

# -------------------------------------------------------------------
# 0. geo.env 파일 위치 지정 및 환경변수 로드
# -------------------------------------------------------------------
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

ENGINE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(ENGINE_URL)


# -------------------------------------------------------------------
# 1. 파일별 전처리 헬퍼 함수
# -------------------------------------------------------------------
def process_product_csv(file_path: Path) -> pd.DataFrame:
    """끝이 product인 CSV 파일 처리"""
    print(f"📦 [Product CSV] Reading: {file_path.name}")

    target_cols = ["성별", "대분류", "소분류", "상품명"]

    try:
        df = pd.read_csv(file_path, usecols=lambda c: c in target_cols)
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, usecols=lambda c: c in target_cols, encoding="cp949")

    cat_cols = ["성별", "대분류", "소분류"]

    def join_categories(row):
        cats = [
            str(val).strip()
            for val in row
            if pd.notna(val) and str(val).strip()
        ]
        return " > ".join(cats)

    available_cat_cols = [c for c in cat_cols if c in df.columns]
    product_cat = (
        df[available_cat_cols].apply(join_categories, axis=1)
        if available_cat_cols
        else ""
    )

    result_df = pd.DataFrame(
        {
            "brand_name": "CJ",
            "brand_type": "대기업",
            "product_name": df["상품명"].astype(str).str.strip(),
            "product_cat": product_cat,
        }
    )
    print(f"   └ 완료 ({len(result_df):,}행 | 브랜드: CJ | 유형: 대기업)")
    return result_df


def process_image_csv(file_path: Path) -> pd.DataFrame:
    """끝이 image인 CSV 파일 처리"""
    print(f"🖼️ [Image CSV] Reading: {file_path.name}")

    try:
        df = pd.read_csv(file_path)
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="cp949")

    df["brand_name"] = "CJ"

    col_mapping = {
        "상품명": "product_name",
        "상세이미지순번": "image_sequence",
        "상세이미지주소링크": "image_url",
    }
    df = df.rename(columns=col_mapping)

    df["product_name"] = df["product_name"].astype(str).str.strip()
    df["image_url"] = df["image_url"].astype(str).str.strip()

    print(f"   └ 완료 ({len(df):,}행)")
    return df


def process_jl_jsonl(file_path: Path) -> dict:
    """오직 jsonld_blocks 데이터만 가공하여 추출하는 함수"""
    print(f"📄 [JSONL] Reading: {file_path.name}")
    jsonl_dict = {}

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            raw_p_name = item.get("상품명")
            if not raw_p_name:
                continue
            p_name = str(raw_p_name).strip()

            # 대기업용: jsonld_blocks 타겟팅
            p_jsonld = item.get("jsonld_blocks")

            if p_name and p_jsonld is not None:
                jsonl_dict[p_name] = p_jsonld

    print(f"   └ 완료 (매핑 항목: {len(jsonl_dict):,}개)")
    return jsonl_dict


def convert_to_json_str(val):
    # 1. None이거나 값 자체가 없는 경우
    if val is None:
        return None

    # 2. dict 또는 list 타입인 경우 (pd.isna 검사를 거치지 않고 바로 json 변환)
    if isinstance(val, (dict, list)):
        try:
            return json.dumps(val, ensure_ascii=False)
        except Exception:
            return str(val)

    # 3. 단일 값(문자열, 숫자 등)에 대한 결측치(NaN) 검사
    try:
        if pd.isna(val):
            return None
    except Exception:
        pass

    return str(val)


# -------------------------------------------------------------------
# 2. 메인 실행 파이프라인
# -------------------------------------------------------------------
def run_full_pipeline(target_directory: Path = CURRENT_DIR):
    print("=" * 65)
    print("🚀 전체 데이터 수집 및 DB 적재 파이프라인 시작 (대기업 전용)")
    print("=" * 65)

    dir_path = Path(target_directory)
    product_dfs = []
    image_dfs = []
    combined_jsonl_dict = {}

    used_files = []
    unused_data_files = []

    # ---------------------------------------------------------------
    # PHASE 1: 파일 수집 및 수집 데이터 병합
    # ---------------------------------------------------------------
    print("\n📁 [PHASE 1] 원천 파일 탐색 및 파싱 중...")
    for file_path in dir_path.rglob("*"):
        if not file_path.is_file():
            continue

        if "complete" in file_path.parts:
            continue

        if (
            file_path.suffix.lower() in [".py", ".env", ".ipynb"]
            or file_path.name.endswith(".env")
            or file_path.name.startswith(".")
        ):
            continue

        file_stem = file_path.stem
        parts = file_stem.split("_")
        file_type = parts[-1]
        ext = file_path.suffix.lower()

        is_data_extension = ext in [".csv", ".jsonl", ".jl", ".json"]

        if file_type == "product" and ext == ".csv":
            product_dfs.append(process_product_csv(file_path))
            used_files.append(file_path)
        elif file_type == "image" and ext == ".csv":
            image_dfs.append(process_image_csv(file_path))
            used_files.append(file_path)
        elif file_type == "jl" and ext in [".jsonl", ".jl"]:
            combined_jsonl_dict.update(process_jl_jsonl(file_path))
            used_files.append(file_path)
        else:
            if is_data_extension:
                unused_data_files.append(file_path)

    print("\n" + "-" * 65)
    print("📁 [파일 수집 리포트]")
    print(f"   ✅ 사용된 데이터 파일 수: {len(used_files)}개")
    print(f"   ⚠️ 사용되지 않은 데이터 파일 수(오타/규칙 미준수): {len(unused_data_files)}개")

    if unused_data_files:
        print("\n   [⚠️ 스킵된 파일 목록 (파일명 확인 필요)]")
        for uf in unused_data_files:
            print(f"    - {uf.relative_to(dir_path)}")
    print("-" * 65)

    if not product_dfs:
        print("❌ 적재할 Product 데이터가 없습니다. 파이프라인을 종료합니다.")
        return

    raw_product_df = pd.concat(product_dfs, ignore_index=True)
    raw_image_df = (
        pd.concat(image_dfs, ignore_index=True) if image_dfs else pd.DataFrame()
    )

    # ---------------------------------------------------------------
    # PHASE 2: raw_data_table 전처리 및 DB 적재
    # ---------------------------------------------------------------
    print("\n📦 [PHASE 2] Product 데이터 전처리 & raw_data_table 적재...")
    p_df = raw_product_df.copy()

    # 상품명 공백 정제 및 교차 중복제거
    p_df["product_name"] = p_df["product_name"].astype(str).str.strip()

    p_before_len = len(p_df)
    p_df = p_df.drop_duplicates(subset=["product_name"], keep="first")
    p_after_len = len(p_df)
    if p_before_len != p_after_len:
        print(f"   ⚠️ [중복 상품 정제] {p_before_len - p_after_len:,}건의 중복 상품 제거됨")

    # JSONL 매칭
    p_df["json_ld_contents"] = p_df["product_name"].map(combined_jsonl_dict)
    p_df["has_json_ld"] = p_df["json_ld_contents"].notna()

    # JSON 형변환
    p_df["json_ld_contents"] = p_df["json_ld_contents"].apply(convert_to_json_str)

    if "brand_type" not in p_df.columns:
        p_df["brand_type"] = "대기업"

    if "text_contents" not in p_df.columns or p_df["text_contents"].isnull().all():
        p_df["text_contents"] = ""
    else:
        p_df["text_contents"] = p_df["text_contents"].fillna("")

    p_brands = p_df["brand_name"].unique().tolist()
    matched_json_count = p_df["has_json_ld"].sum()

    print(f"   📊 Product 총 적재 행 수: {len(p_df):,}개")
    print(f"   🎯 JSON-LD 매칭 성공 수: {matched_json_count:,}개 / {len(p_df):,}개")
    print(f"   🏷️ 지정된 브랜드: {', '.join(map(str, p_brands))}")

    try:
        raw_target_cols = [
            "brand_name",
            "product_name",
            "product_cat",
            "json_ld_contents",
            "has_json_ld",
            "brand_type",
            "text_contents",
        ]
        p_df_db = p_df[[c for c in raw_target_cols if c in p_df.columns]]
        p_df_db.to_sql(
            name="raw_data_table",
            con=engine,
            if_exists="append",
            index=False,
            chunksize=1000,
        )
        print("   🚀 raw_data_table 적재 완료!")
    except Exception as e:
        print(f"   ❌ raw_data_table 적재 실패: {e}")
        return

    # ---------------------------------------------------------------
    # PHASE 3: image_data_table 전처리 및 DB 적재
    # ---------------------------------------------------------------
    if raw_image_df.empty:
        print("\n⚠️ 이미지 데이터가 존재하지 않아 이미지 적재를 스킵합니다.")
        return

    print("\n🖼️ [PHASE 3] Image 데이터 전처리 & image_data_table 적재...")

    query = "SELECT page_id, brand_name, product_name FROM raw_data_table"
    product_id_df = pd.read_sql(query, con=engine)

    img_df = raw_image_df.copy()
    raw_total_cnt = len(img_df)
    print(f"   📊 [1] 원천 이미지 파일 총 행 수: {raw_total_cnt:,}건")

    # alt 컬럼 리네임
    alt_rename_map = {}
    if "alt속성존재여부" in img_df.columns:
        alt_rename_map["alt속성존재여부"] = "has_alt"
    if "alt속성값" in img_df.columns:
        alt_rename_map["alt속성값"] = "alt_contents"

    if alt_rename_map:
        img_df = img_df.rename(columns=alt_rename_map)

    # 문자열 정제 및 page_id 매칭
    img_df["product_name"] = img_df["product_name"].astype(str).str.strip()
    product_id_df["product_name"] = product_id_df["product_name"].astype(str).str.strip()

    page_map = product_id_df.drop_duplicates(subset=["product_name"]).set_index("product_name")["page_id"].to_dict()
    img_df["page_id"] = img_df["product_name"].map(page_map)

    missing_page_id_cnt = img_df["page_id"].isnull().sum()
    if missing_page_id_cnt > 0:
        print(f"   ⚠️ [2] raw_data_table에 product_name이 없어 매칭 실패한 이미지: {missing_page_id_cnt:,}건")

    img_db_df = img_df.dropna(subset=["page_id"]).copy()
    img_db_df["page_id"] = img_db_df["page_id"].astype(int)

    # 기본 컬럼 정제
    if "image_text" not in img_db_df.columns or img_db_df["image_text"].isnull().all():
        img_db_df["image_text"] = ""
    else:
        img_db_df["image_text"] = img_db_df["image_text"].fillna("")

    if "has_alt" in img_db_df.columns:
        img_db_df["has_alt"] = img_db_df["has_alt"].astype(str).str.upper().isin(["TRUE", "1", "Y", "O", "유", "YES"])
    else:
        img_db_df["has_alt"] = False

    if "alt_contents" in img_db_df.columns:
        img_db_df["alt_contents"] = img_db_df["alt_contents"].fillna("").astype(str)
        img_db_df["alt_contents"] = img_db_df["alt_contents"].replace("", None)
    else:
        img_db_df["alt_contents"] = None

    # 타겟 컬럼 정리
    img_target_cols = [
        "page_id",
        "brand_name",
        "image_sequence",
        "image_text",
        "image_url",
        "has_alt",
        "alt_contents",
    ]
    img_db_df = img_db_df[[c for c in img_target_cols if c in img_db_df.columns]]

    # (page_id, image_sequence) 복합키 기준 중복 제거
    before_len = len(img_db_df)
    img_db_df = img_db_df.drop_duplicates(
        subset=["page_id", "image_sequence"], keep="first"
    )
    after_len = len(img_db_df)

    if before_len != after_len:
        print(f"   ⚠️ [3] 중복 이미지 순서 데이터 정제됨: {before_len - after_len:,}건")

    img_brands = img_db_df["brand_name"].unique().tolist()
    print(f"   🚀 [최종 DB 적재 대상] Image 총 행 수: {len(img_db_df):,}개")
    print(f"   🏷️ 이미지 브랜드 ({len(img_brands)}개): {', '.join(map(str, img_brands))}")

    try:
        img_db_df.to_sql(
            name="image_data_table",
            con=engine,
            if_exists="append",
            index=False,
            chunksize=1000,
        )
        print("   ✅ image_data_table 적재 완료!")
    except Exception as e:
        print(f"   ❌ image_data_table 적재 실패: {e}")
        return

    print("\n" + "=" * 65)
    print("🎉 모든 데이터 적재 파이프라인 완료!")
    print("=" * 65)


# -------------------------------------------------------------------
# 3. 파이프라인 실행
# -------------------------------------------------------------------
if __name__ == "__main__":
    run_full_pipeline()