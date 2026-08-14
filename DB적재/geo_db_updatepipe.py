import json
import os
from collections import defaultdict
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

# -------------------------------------------------------------------
# 0. geo.env 파일 위치 지정 및 DB 엔진 설정
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
    """text 파일이 없을 때 실행되는 Product CSV 처리 함수 (text_contents는 빈값)"""
    print(f"📦 [Product CSV] Reading: {file_path.name}")
    target_cols = ["쇼핑몰명", "대분류", "중분류", "소분류", "상품명"]

    try:
        df = pd.read_csv(file_path, usecols=lambda c: c in target_cols)
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, usecols=lambda c: c in target_cols, encoding="cp949")

    cat_cols = ["대분류", "중분류", "소분류"]

    def join_categories(row):
        cats = [
            str(val).strip()
            for val in row
            if pd.notna(val) and str(val).strip()
        ]
        return " > ".join(cats)

    existing_cat_cols = [c for c in cat_cols if c in df.columns]
    product_cat = df[existing_cat_cols].apply(join_categories, axis=1) if existing_cat_cols else ""

    result_df = pd.DataFrame(
        {
            "brand_name": df["쇼핑몰명"].astype(str).str.strip() if "쇼핑몰명" in df.columns else "",
            "product_name": df["상품명"].astype(str).str.strip() if "상품명" in df.columns else "",
            "product_cat": product_cat,
            "text_contents": "",
        }
    )
    print(f"   └ 완료 ({len(result_df):,}행)")
    return result_df


def process_text_csv(file_path: Path) -> pd.DataFrame:
    """text 파일이 존재할 때 단독 실행되는 Text CSV 처리 함수 (본문텍스트 포함)"""
    print(f"📝 [Text CSV] Reading: {file_path.name}")
    target_cols = ["쇼핑몰명", "대분류", "중분류", "소분류", "상품명", "본문텍스트"]

    try:
        df = pd.read_csv(file_path, usecols=lambda c: c in target_cols)
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, usecols=lambda c: c in target_cols, encoding="cp949")

    cat_cols = ["대분류", "중분류", "소분류"]

    def join_categories(row):
        cats = [
            str(val).strip()
            for val in row
            if pd.notna(val) and str(val).strip()
        ]
        return " > ".join(cats)

    existing_cat_cols = [c for c in cat_cols if c in df.columns]
    product_cat = df[existing_cat_cols].apply(join_categories, axis=1) if existing_cat_cols else ""

    text_contents = (
        df["본문텍스트"].fillna("").astype(str).str.strip()
        if "본문텍스트" in df.columns
        else ""
    )

    result_df = pd.DataFrame(
        {
            "brand_name": df["쇼핑몰명"].astype(str).str.strip() if "쇼핑몰명" in df.columns else "",
            "product_name": df["상품명"].astype(str).str.strip() if "상품명" in df.columns else "",
            "product_cat": product_cat,
            "text_contents": text_contents,
        }
    )
    print(f"   └ 완료 ({len(result_df):,}행)")
    return result_df


def process_image_csv(file_path: Path) -> pd.DataFrame:
    """끝이 image인 CSV 파일 처리"""
    print(f"🖼️ [Image CSV] Reading: {file_path.name}")

    try:
        df = pd.read_csv(file_path)
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="cp949")

    if "브랜드" in df.columns:
        df = df.rename(columns={"브랜드": "brand_name"})
    elif "쇼핑몰명" in df.columns:
        df = df.rename(columns={"쇼핑몰명": "brand_name"})

    col_mapping = {
        "상품명": "product_name",
        "상세이미지순번": "image_sequence",
        "상세이미지주소링크": "image_url",
    }
    df = df.rename(columns=col_mapping)

    df["product_name"] = df["product_name"].astype(str).str.strip()
    df["image_url"] = df["image_url"].astype(str).str.strip()

    if "brand_name" in df.columns:
        df["brand_name"] = df["brand_name"].astype(str).str.strip()

    print(f"   └ 완료 ({len(df):,}행)")
    return df


def process_jl_jsonl(file_path: Path) -> dict:
    """끝이 jl인 JSONL 파일 처리"""
    print(f"📄 [JSONL] Reading: {file_path.name}")
    jsonl_dict = {}

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            p_name = item.get("상품명")
            p_jsonld = item.get("product_jsonld")

            if p_name:
                jsonl_dict[str(p_name).strip()] = p_jsonld

    print(f"   └ 완료 (매핑 항목: {len(jsonl_dict):,}개)")
    return jsonl_dict


def convert_to_json_str(val):
    if pd.isna(val) or val is None:
        return None
    if isinstance(val, (dict, list)):
        return json.dumps(val, ensure_ascii=False)
    return str(val)


# -------------------------------------------------------------------
# 2. 메인 실행 파이프라인
# -------------------------------------------------------------------
def run_full_pipeline(target_directory: Path = CURRENT_DIR):
    print("=" * 65)
    print("🚀 전체 데이터 수집 및 DB 적재 파이프라인 시작")
    print("=" * 65)

    dir_path = Path(target_directory)
    tagged_files = defaultdict(dict)
    combined_jsonl_dict = {}

    # ---------------------------------------------------------------
    # PHASE 1: 파일 수집 및 그룹핑
    # ---------------------------------------------------------------
    print("\n📁 [PHASE 1] 원천 파일 탐색 및 파싱 중...")
    for file_path in dir_path.rglob("*"):
        if not file_path.is_file():
            continue

        if "complete" in file_path.parts:
            continue

        if (
            file_path.suffix in [".py", ".env", ".ipynb"]
            or file_path.name.endswith(".env")
            or file_path.name.startswith(".")
        ):
            continue

        file_stem = file_path.stem
        parts = file_stem.split("_")
        file_type = parts[-1]  # product, text, image, jl
        tag_prefix = "_".join(parts[:-1])

        if file_type == "product" and file_path.suffix == ".csv":
            tagged_files[tag_prefix]["product"] = file_path
        elif file_type == "text" and file_path.suffix == ".csv":
            tagged_files[tag_prefix]["text"] = file_path
        elif file_type == "image" and file_path.suffix == ".csv":
            tagged_files[tag_prefix]["image"] = file_path
        elif file_type == "jl" and file_path.suffix in [".jsonl", ".jl"]:
            combined_jsonl_dict.update(process_jl_jsonl(file_path))

    # ---------------------------------------------------------------
    # PHASE 2: 태그별 배타적 실행 (Text가 있으면 Text만, 없으면 Product)
    # ---------------------------------------------------------------
    product_dfs = []
    image_dfs = []

    for tag, files in tagged_files.items():
        if "text" in files:
            p_df = process_text_csv(files["text"])
            product_dfs.append(p_df)
        elif "product" in files:
            p_df = process_product_csv(files["product"])
            product_dfs.append(p_df)

        if "image" in files:
            image_dfs.append(process_image_csv(files["image"]))

    if not product_dfs:
        print("❌ 적재할 Product/Text 데이터가 없습니다. 파이프라인을 종료합니다.")
        return

    raw_product_df = pd.concat(product_dfs, ignore_index=True)
    raw_image_df = (
        pd.concat(image_dfs, ignore_index=True) if image_dfs else pd.DataFrame()
    )

    # ---------------------------------------------------------------
    # PHASE 2-2: raw_data_table 적재 준비
    # ---------------------------------------------------------------
    print("\n📦 [PHASE 2] Product 데이터 전처리 & raw_data_table 적재...")
    p_df = raw_product_df.copy()

    # 상품명 공백 정제
    p_df["product_name"] = p_df["product_name"].astype(str).str.strip()

    # 카테고리 파일 간 상품 중복 정제
    p_before_len = len(p_df)
    p_df = p_df.drop_duplicates(subset=["product_name"], keep="first")
    p_after_len = len(p_df)
    if p_before_len != p_after_len:
        print(f"   ⚠️ [중복 상품 정제] {p_before_len - p_after_len:,}건의 교차 중복 상품 제거됨")

    # JSONL 매칭
    p_df["json_ld_contents"] = p_df["product_name"].map(combined_jsonl_dict)
    p_df["has_json_ld"] = p_df["json_ld_contents"].notna()

    # JSON 형변환
    p_df["json_ld_contents"] = p_df["json_ld_contents"].apply(
        convert_to_json_str
    )

    # 기본 컬럼 설정
    if "brand_type" not in p_df.columns:
        p_df["brand_type"] = "소상공인"

    p_df["text_contents"] = p_df["text_contents"].fillna("").astype(str)

    # 모니터링 로그
    p_brands = p_df["brand_name"].unique().tolist()
    matched_json_count = p_df["has_json_ld"].sum()
    text_filled_count = (p_df["text_contents"] != "").sum()

    print(f"   📊 Product 총 적재 행 수: {len(p_df):,}개")
    print(f"   🎯 JSON-LD 매칭 성공 수: {matched_json_count:,}개 / {len(p_df):,}개")
    print(f"   📝 본문 텍스트 수집 수: {text_filled_count:,}개 / {len(p_df):,}개")
    print(
        f"   🏷️ 감지된 브랜드 ({len(p_brands)}개): {', '.join(map(str, img_brands))}"
    )

    # raw_data_table DB 적재
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

    # DB에서 방금 적재된 page_id 가져오기
    query = "SELECT page_id, brand_name, product_name FROM raw_data_table"
    product_id_df = pd.read_sql(query, con=engine)

    img_df = raw_image_df.copy()
    raw_total_cnt = len(img_df)
    print(f"   📊 [1] 원천 이미지 파일 총 행 수: {raw_total_cnt:,}건")

    # alt 관련 컬럼 매핑
    alt_rename_map = {}
    if "alt속성존재여부" in img_df.columns:
        alt_rename_map["alt속성존재여부"] = "has_alt"
    if "alt속성값" in img_df.columns:
        alt_rename_map["alt속성값"] = "alt_contents"

    if alt_rename_map:
        img_df = img_df.rename(columns=alt_rename_map)

    # 공백 정제
    img_df["product_name"] = img_df["product_name"].astype(str).str.strip()
    product_id_df["product_name"] = product_id_df["product_name"].astype(str).str.strip()

    if "brand_name" in img_df.columns:
        img_df["brand_name"] = img_df["brand_name"].astype(str).str.strip()
    product_id_df["brand_name"] = product_id_df["brand_name"].astype(str).str.strip()

    # page_id 매칭
    page_map = product_id_df.drop_duplicates(subset=["product_name"]).set_index("product_name")["page_id"].to_dict()
    brand_map = product_id_df.drop_duplicates(subset=["product_name"]).set_index("product_name")["brand_name"].to_dict()

    img_df["page_id"] = img_df["product_name"].map(page_map)

    if "brand_name" not in img_df.columns or img_df["brand_name"].isnull().all():
        img_df["brand_name"] = img_df["product_name"].map(brand_map)

    missing_page_id_cnt = img_df["page_id"].isnull().sum()
    if missing_page_id_cnt > 0:
        print(f"   ⚠️ [2] raw_data_table에 product_name이 없어 매칭 실패한 이미지: {missing_page_id_cnt:,}건")

    img_db_df = img_df.dropna(subset=["page_id"]).copy()
    img_db_df["page_id"] = img_db_df["page_id"].astype(int)

    # 미수집/기본 컬럼 정제
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

    # 타겟 컬럼 순서 맞추기
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

    # 복합키 (page_id, image_sequence) 중복 제거
    before_len = len(img_db_df)
    img_db_df = img_db_df.drop_duplicates(
        subset=["page_id", "image_sequence"], keep="first"
    )
    after_len = len(img_db_df)

    if before_len != after_len:
        print(f"   ⚠️ [3] 중복 이미지 순서 데이터 정제됨: {before_len - after_len:,}건")

    # DB 적재
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


if __name__ == "__main__":
    run_full_pipeline()