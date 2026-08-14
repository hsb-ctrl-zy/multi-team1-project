import json
import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

# -------------------------------------------------------------------
# 0. 환경 변수 및 DB 엔진 설정
# -------------------------------------------------------------------
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine

import os
from pathlib import Path
from sqlalchemy import create_engine

# -------------------------------------------------------------------
# 0. geo.env 파일 위치를 파이썬 스크립트 기준으로 직접 지정
# -------------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
ENV_PATH = CURRENT_DIR / "geo.env"

# geo.env 파일을 직접 읽어서 환경변수에 세팅 (인코딩/경로 문제 완전 차단)
with open(ENV_PATH, "r", encoding="utf-8-sig") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, val = line.split("=", 1)
            os.environ[key.strip()] = val.strip()

# 환경변수 값 가져오기
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

# DB 접속 엔진 생성
ENGINE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(ENGINE_URL)


# -------------------------------------------------------------------
# 1. 파일별 전처리 헬퍼 함수
# -------------------------------------------------------------------
def process_product_csv(file_path: Path) -> pd.DataFrame:
    """끝이 product인 CSV 파일 처리"""
    print(f"📦 [Product CSV] Reading: {file_path.name}")
    target_cols = ["쇼핑몰명", "대분류", "중분류", "소분류", "상품명"]

    try:
        df = pd.read_csv(file_path, usecols=target_cols)
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, usecols=target_cols, encoding="cp949")

    cat_cols = ["대분류", "중분류", "소분류"]

    def join_categories(row):
        cats = [
            str(val).strip()
            for val in row
            if pd.notna(val) and str(val).strip()
        ]
        return " > ".join(cats)

    product_cat = df[cat_cols].apply(join_categories, axis=1)

    result_df = pd.DataFrame(
        {
            "brand_name": df["쇼핑몰명"].astype(str).str.strip(),
            "product_name": df["상품명"].astype(str).str.strip(),
            "product_cat": product_cat,
        }
    )
    print(f"   └ 완료 ({len(result_df):,}행)")
    return result_df


def process_image_csv(file_path: Path) -> pd.DataFrame:
    """끝이 image인 CSV 파일 처리"""
    print(f"🖼️ [Image CSV] Reading: {file_path.name}")

    # 원천 CSV에 '브랜드' 혹은 '쇼핑몰명' 컬럼이 포함되어 있을 수 있으므로 범용 처리
    try:
        df = pd.read_csv(file_path)
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="cp949")

    # 원천 데이터의 '브랜드' 또는 '쇼핑몰명' 컬럼을 brand_name으로 매핑
    if "브랜드" in df.columns:
        df = df.rename(columns={"브랜드": "brand_name"})
    elif "쇼핑몰명" in df.columns:
        df = df.rename(columns={"쇼핑몰명": "brand_name"})

    # 필수 컬럼 구성
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
    product_dfs = []
    image_dfs = []
    combined_jsonl_dict = {}

    # ---------------------------------------------------------------
    # PHASE 1: 파일 수집 및 수집 데이터 병합
    # ---------------------------------------------------------------
    # PHASE 1 탐색 루프
    print("\n📁 [PHASE 1] 원천 파일 탐색 및 파싱 중...")
    for file_path in dir_path.rglob("*"):
        if not file_path.is_file():
            continue

        # 1. complete 폴더 및 하위 파일 전체 스킵
        if "complete" in file_path.parts:
            continue

        # 2. 파이썬 스크립트, .env(설정 파일), 주피터 노트북, 숨김 파일 스킵
        if (
            file_path.suffix in [".py", ".env", ".ipynb"]
            or file_path.name.endswith(".env")
            or file_path.name.startswith(".")
        ):
            continue

        # 3. 데이터 파일 탐색 계속 진행...
        file_stem = file_path.stem
        parts = file_stem.split("_")
        file_type = parts[-1]

        if file_type == "product" and file_path.suffix == ".csv":
            product_dfs.append(process_product_csv(file_path))
        elif file_type == "image" and file_path.suffix == ".csv":
            image_dfs.append(process_image_csv(file_path))
        elif file_type == "jl" and file_path.suffix in [".jsonl", ".jl"]:
            combined_jsonl_dict.update(process_jl_jsonl(file_path))

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

    # JSONL 매칭
    p_df["json_ld_contents"] = p_df["product_name"].map(combined_jsonl_dict)
    p_df["has_json_ld"] = p_df["json_ld_contents"].notna()

    # JSON 형변환
    p_df["json_ld_contents"] = p_df["json_ld_contents"].apply(
        convert_to_json_str
    )

    # 고정/기본 컬럼 설정
    if "brand_type" not in p_df.columns:
        p_df["brand_type"] = "소상공인"

    # text_contents 동적 처리 (채워진 상태로 오면 유지, 없으면 빈값)
    if (
        "text_contents" not in p_df.columns
        or p_df["text_contents"].isnull().all()
    ):
        print("   ℹ️ text_contents 컬럼이 비어있어 기본값('')으로 채웁니다.")
        p_df["text_contents"] = ""
    else:
        print("   ℹ️ 기존 수집된 text_contents 데이터를 유지합니다.")
        p_df["text_contents"] = p_df["text_contents"].fillna("")

    # 데이터 모니터링 로그
    p_brands = p_df["brand_name"].unique().tolist()
    matched_json_count = p_df["has_json_ld"].sum()
    print(f"   📊 Product 총 행 수: {len(p_df):,}개")
    print(
        f"   🎯 JSON-LD 매칭 성공 수: {matched_json_count:,}개 / {len(p_df):,}개"
    )
    print(
        f"   🏷️ 감지된 브랜드 ({len(p_brands)}개): {', '.join(map(str, p_brands[:5]))}{' 외 ...' if len(p_brands) > 5 else ''}"
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

    # DB에서 최신 page_id 가져오기 (brand_name + product_name 매칭)
    query = "SELECT page_id, brand_name, product_name FROM raw_data_table"
    product_id_df = pd.read_sql(query, con=engine)

    img_df = raw_image_df.copy()

    # image_df에 brand_name이 없는 경우 product_name으로 참조 매칭
    if "brand_name" not in img_df.columns:
        brand_map = p_df[["product_name", "brand_name"]].drop_duplicates()
        img_df = pd.merge(
            img_df, brand_map, on="product_name", how="left"
        )

    # brand_name + product_name 조인으로 정확한 page_id 1:1 매칭
    img_db_df = pd.merge(
        img_df, product_id_df, on=["brand_name", "product_name"], how="inner"
    )

    # 미수집 컬럼 동적 체크 및 채우기
    if (
        "image_text" not in img_db_df.columns
        or img_db_df["image_text"].isnull().all()
    ):
        img_db_df["image_text"] = ""
    else:
        img_db_df["image_text"] = img_db_df["image_text"].fillna("")

    if "has_alt" not in img_db_df.columns:
        img_db_df["has_alt"] = False

    if "alt_contents" not in img_db_df.columns:
        img_db_df["alt_contents"] = None

    # 이미지 타겟 컬럼
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

    # 복합키(page_id, image_sequence) 중복 사전에 제거
    before_len = len(img_db_df)
    img_db_df = img_db_df.drop_duplicates(
        subset=["page_id", "image_sequence"], keep="first"
    )
    after_len = len(img_db_df)

    if before_len != after_len:
        print(f"   ⚠️ 중복 이미지 순서 데이터 {before_len - after_len}건 정제됨")

    # 이미지 로깅
    img_brands = img_db_df["brand_name"].unique().tolist()
    print(f"   📊 Image 총 행 수: {len(img_db_df):,}개")
    print(
        f"   🏷️ 이미지 데이터 브랜드 ({len(img_brands)}개): {', '.join(map(str, img_brands[:5]))}{' 외 ...' if len(img_brands) > 5 else ''}"
    )

    # image_data_table DB 적재
    try:
        img_db_df.to_sql(
            name="image_data_table",
            con=engine,
            if_exists="append",
            index=False,
            chunksize=1000,
        )
        print("   🚀 image_data_table 적재 완료!")
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