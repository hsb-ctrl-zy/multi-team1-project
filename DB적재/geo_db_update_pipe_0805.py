import json
import os
from collections import defaultdict
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

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
# 1. 파일별 전처리 헬퍼 함수 (소상공인 맞춤형)
# -------------------------------------------------------------------
def process_product_csv(file_path: Path) -> pd.DataFrame:
    """Product CSV 처리 함수 (cat 컬럼 -> product_cat으로 매핑)"""
    print(f"📦 [Product CSV] Reading: {file_path.name}")
    
    target_cols = ["쇼핑몰명", "cat", "CAT", "상품명", "상품번호"]

    try:
        df = pd.read_csv(file_path, usecols=lambda c: c in target_cols)
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, usecols=lambda c: c in target_cols, encoding="cp949")

    cat_col = None
    for c in df.columns:
        if c.lower() == "cat":
            cat_col = c
            break

    product_cat_series = df[cat_col].fillna("").astype(str).str.strip() if cat_col else ""

    result_df = pd.DataFrame(
        {
            "brand_name": df["쇼핑몰명"].astype(str).str.strip() if "쇼핑몰명" in df.columns else "",
            "product_name": df["상품명"].astype(str).str.strip() if "상품명" in df.columns else "",
            "product_id_raw": df["상품번호"].astype(str).str.strip() if "상품번호" in df.columns else "",
            "product_cat": product_cat_series,
            "text_contents": "",
            "image_text": "",
        }
    )
    print(f"   └ 완료 ({len(result_df):,}행)")
    return result_df


def process_text_csv(file_path: Path) -> pd.DataFrame:
    """Text CSV 처리 함수 (DB 매칭용 브랜드/상품명 및 text_contents만 추출)"""
    print(f"📝 [Text CSV] Reading: {file_path.name}")
    target_cols = ["쇼핑몰명", "상품명", "본문텍스트"]

    try:
        df = pd.read_csv(file_path, usecols=lambda c: c in target_cols)
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, usecols=lambda c: c in target_cols, encoding="cp949")

    text_contents = (
        df["본문텍스트"].fillna("").astype(str).str.strip()
        if "본문텍스트" in df.columns
        else ""
    )

    result_df = pd.DataFrame(
        {
            "brand_name": df["쇼핑몰명"].astype(str).str.strip() if "쇼핑몰명" in df.columns else "",
            "product_name": df["상품명"].astype(str).str.strip() if "상품명" in df.columns else "",
            "text_contents": text_contents,
        }
    )
    print(f"   └ 완료 ({len(result_df):,}행)")
    return result_df


def process_vl_csv(file_path: Path) -> pd.DataFrame:
    """VL CSV 파일 처리"""
    print(f"🔍 [VL CSV] Reading: {file_path.name}")
    target_cols = ["쇼핑몰명", "상품명", "Text_contents"]

    try:
        df = pd.read_csv(file_path, usecols=lambda c: c in target_cols)
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, usecols=lambda c: c in target_cols, encoding="cp949")

    if 'Text_contents' in df.columns:
        df['Text_contents'] = df['Text_contents'].fillna('')

    result_df = pd.DataFrame(
        {
            "brand_name": df["쇼핑몰명"].astype(str).str.strip() if "쇼핑몰명" in df.columns else "",
            "product_name": df["상품명"].astype(str).str.strip() if "상품명" in df.columns else "",
            "vl_image_text": df["Text_contents"].fillna("").astype(str).str.strip() if "Text_contents" in df.columns else "",
        }
    )
    result_df = result_df.drop_duplicates(subset=["brand_name", "product_name"], keep="first")
    print(f"   └ 완료 ({len(result_df):,}행)")
    return result_df


def process_image_csv(file_path: Path) -> pd.DataFrame:
    """Image CSV 파일 처리 (alt속성 유무에 상관없이 안전 파싱)"""
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
    if "image_url" in df.columns:
        df["image_url"] = df["image_url"].astype(str).str.strip()

    if "brand_name" in df.columns:
        df["brand_name"] = df["brand_name"].astype(str).str.strip()

    print(f"   └ 완료 ({len(df):,}행)")
    return df


def process_jl_jsonl(file_path: Path) -> dict:
    """JSONL 파일 처리 (소상공인 타겟 키: product_jsonld)"""
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

            p_name = item.get("상품명")
            p_jsonld = item.get("product_jsonld")

            if p_name and p_jsonld is not None:
                jsonl_dict[str(p_name).strip()] = p_jsonld

    print(f"   └ 완료 (매핑 항목: {len(jsonl_dict):,}개)")
    return jsonl_dict


def convert_to_json_str(val):
    if val is None:
        return None
    if isinstance(val, (dict, list)):
        try:
            return json.dumps(val, ensure_ascii=False)
        except Exception:
            return str(val)
    try:
        if pd.isna(val):
            return None
    except Exception:
        pass
    return str(val)


# -------------------------------------------------------------------
# DB 처리 전용 함수 구역
# -------------------------------------------------------------------
def update_existing_raw_data(standalone_text_dfs: list, standalone_vl_dfs: list):
    """기존 DB 데이터에 text_contents 또는 image_text를 UPDATE하는 함수"""
    
    # 1. Text 컬럼 UPDATE
    if standalone_text_dfs:
        combined_text = pd.concat(standalone_text_dfs, ignore_index=True)
        combined_text = combined_text[combined_text["text_contents"] != ""]
        if not combined_text.empty:
            print(f"\n🔄 [본문 UPDATE] text_contents DB 업데이트 중... ({len(combined_text):,}건)")
            update_sql = text("""
                UPDATE raw_data_table 
                SET text_contents = :text_contents 
                WHERE brand_name = :brand_name AND product_name = :product_name
            """)
            records = combined_text[["brand_name", "product_name", "text_contents"]].to_dict(orient="records")
            with engine.begin() as conn:
                conn.execute(update_sql, records)
            print("   ✅ text_contents 업데이트 완료!")

    # 2. VL (image_text) 컬럼 UPDATE
    if standalone_vl_dfs:
        combined_vl = pd.concat(standalone_vl_dfs, ignore_index=True)
        combined_vl = combined_vl[combined_vl["vl_image_text"] != ""]
        if not combined_vl.empty:
            print(f"\n🔄 [VL UPDATE] image_text DB 업데이트 중... ({len(combined_vl):,}건)")
            update_sql = text("""
                UPDATE raw_data_table 
                SET image_text = :vl_image_text 
                WHERE brand_name = :brand_name AND product_name = :product_name
            """)
            records = combined_vl[["brand_name", "product_name", "vl_image_text"]].to_dict(orient="records")
            with engine.begin() as conn:
                conn.execute(update_sql, records)
            print("   ✅ image_text 업데이트 완료!")


def insert_new_image_data(image_dfs: list):
    """신규 상품용 이미지 데이터 DB INSERT 적재 함수"""
    raw_image_df = pd.concat(image_dfs, ignore_index=True) if image_dfs else pd.DataFrame()

    if raw_image_df.empty:
        return

    print("\n🖼️ [신규 Image 적재] image_data_table INSERT 적재 시작...")
    print(f"   📊 이미지 원본 수집 데이터: 총 {len(raw_image_df):,}건")

    query = "SELECT page_id, brand_name, product_name FROM raw_data_table"
    product_id_df = pd.read_sql(query, con=engine)

    img_df = raw_image_df.copy()

    alt_rename_map = {}
    if "alt속성존재여부" in img_df.columns:
        alt_rename_map["alt속성존재여부"] = "has_alt"
    if "alt속성값" in img_df.columns:
        alt_rename_map["alt속성값"] = "alt_contents"

    if alt_rename_map:
        img_df = img_df.rename(columns=alt_rename_map)

    img_df["product_name"] = img_df["product_name"].astype(str).str.strip()
    product_id_df["product_name"] = product_id_df["product_name"].astype(str).str.strip()

    if "brand_name" in img_df.columns:
        img_df["brand_name"] = img_df["brand_name"].astype(str).str.strip()
    product_id_df["brand_name"] = product_id_df["brand_name"].astype(str).str.strip()

    # page_id 및 brand_name 매칭 (기존 방식 유지)
    page_map = product_id_df.drop_duplicates(subset=["product_name"]).set_index("product_name")["page_id"].to_dict()
    brand_map = product_id_df.drop_duplicates(subset=["product_name"]).set_index("product_name")["brand_name"].to_dict()

    img_df["page_id"] = img_df["product_name"].map(page_map)

    if "brand_name" not in img_df.columns or img_df["brand_name"].isnull().all():
        img_df["brand_name"] = img_df["product_name"].map(brand_map)

    img_db_df = img_df.dropna(subset=["page_id"]).copy()
    img_db_df["page_id"] = img_db_df["page_id"].astype(int)

    print(f"   🔍 page_id 매칭 성공 건수: {len(img_db_df):,}건 / {len(img_df):,}건")

    # alt 속성 존재 여부 유연 처리
    if "has_alt" in img_db_df.columns:
        img_db_df["has_alt"] = img_db_df["has_alt"].astype(str).str.upper().isin(["TRUE", "1", "Y", "O", "유", "YES"])
    else:
        img_db_df["has_alt"] = False

    # alt 속성값 유연 처리
    if "alt_contents" in img_db_df.columns:
        img_db_df["alt_contents"] = img_db_df["alt_contents"].fillna("").astype(str).replace("", None)
    else:
        img_db_df["alt_contents"] = None

    img_target_cols = [
        "page_id",
        "brand_name",
        "image_sequence",
        "image_url",
        "has_alt",
        "alt_contents",
    ]
    img_db_df = img_db_df[[c for c in img_target_cols if c in img_db_df.columns]]
    img_db_df = img_db_df.drop_duplicates(subset=["page_id", "image_sequence"], keep="first")

    # [수정사항] DB 기존 image_data_table 중복 조회 및 필터링
    try:
        existing_img_df = pd.read_sql("SELECT page_id, image_sequence FROM image_data_table", con=engine)
        if not existing_img_df.empty:
            existing_tuples = set(zip(existing_img_df["page_id"], existing_img_df["image_sequence"]))
            before_filter = len(img_db_df)
            img_db_df = img_db_df[
                ~img_db_df.apply(lambda r: (r["page_id"], r["image_sequence"]) in existing_tuples, axis=1)
            ]
            filtered_count = before_filter - len(img_db_df)
            if filtered_count > 0:
                print(f"   ⚠️ DB 기존 이미지 중복 제외: {filtered_count:,}건 제외됨")
    except Exception as e:
        print(f"   ℹ️ 기존 이미지 데이터 조회 스킵 (최초 적재 또는 오류): {e}")

    if img_db_df.empty:
        print("   ⚠️ 적재할 신규 이미지 데이터가 없습니다 (모두 중복이거나 매칭 실패).")
        return

    try:
        img_db_df.to_sql(
            name="image_data_table",
            con=engine,
            if_exists="append",
            index=False,
            chunksize=1000,
        )
        print(f"   ✅ image_data_table 신규 적재 완료! ({len(img_db_df):,}건)")
    except Exception as e:
        print(f"   ❌ image_data_table 적재 실패: {e}")


def update_existing_alt_data(image_dfs: list):
    """기존 image_data_table DB 데이터의 alt_contents 및 has_alt를 UPDATE하는 함수"""
    if not image_dfs:
        return

    combined_img = pd.concat(image_dfs, ignore_index=True)
    
    query = "SELECT page_id, product_name FROM raw_data_table"
    product_id_df = pd.read_sql(query, con=engine)
    
    combined_img["product_name"] = combined_img["product_name"].astype(str).str.strip()
    product_id_df["product_name"] = product_id_df["product_name"].astype(str).str.strip()
    
    page_map = product_id_df.drop_duplicates(subset=["product_name"]).set_index("product_name")["page_id"].to_dict()
    combined_img["page_id"] = combined_img["product_name"].map(page_map)
    
    combined_img = combined_img.dropna(subset=["page_id"]).copy()
    combined_img["page_id"] = combined_img["page_id"].astype(int)
    
    # 1. has_alt 처리
    if "alt속성존재여부" in combined_img.columns:
        combined_img["has_alt"] = combined_img["alt속성존재여부"].astype(str).str.upper().isin(["TRUE", "1", "Y", "O", "유", "YES"])
    elif "has_alt" not in combined_img.columns:
        combined_img["has_alt"] = False

    # 2. alt_contents 결측치(NaN/빈문자열) -> None(SQL NULL) 완벽 변환
    if "alt속성값" in combined_img.columns:
        combined_img["alt_contents"] = combined_img["alt속성값"]
    elif "alt_contents" not in combined_img.columns:
        combined_img["alt_contents"] = None

    # pandas NaN, float('nan'), 빈 문자열을 전부 None(NULL)으로 교체
    combined_img["alt_contents"] = combined_img["alt_contents"].where(pd.notna(combined_img["alt_contents"]), None)
    combined_img["alt_contents"] = combined_img["alt_contents"].astype(object).replace("", None)

    print(f"\n🔄 [alt 속성 UPDATE] image_data_table DB 업데이트 중... ({len(combined_img):,}건)")
    
    update_sql = text("""
        UPDATE image_data_table 
        SET has_alt = :has_alt,
            alt_contents = :alt_contents 
        WHERE page_id = :page_id AND image_sequence = :image_sequence
    """)
    
    records = combined_img[["page_id", "image_sequence", "has_alt", "alt_contents"]].to_dict(orient="records")
    
    # records 리스트 내에 혹시 남아있을 지 모를 NaN을 None으로 최종 검증
    for r in records:
        if pd.isna(r["alt_contents"]):
            r["alt_contents"] = None

    with engine.begin() as conn:
        conn.execute(update_sql, records)
        
    print("   ✅ alt 속성(has_alt, alt_contents) 업데이트 완료!")


# -------------------------------------------------------------------
# 2. 메인 실행 파이프라인
# -------------------------------------------------------------------
def run_full_pipeline(target_directory: Path = CURRENT_DIR):
    print("=" * 65)
    print("🚀 전체 데이터 수집 및 DB 적재/업데이트 파이프라인 시작 (소상공인 전용)")
    print("=" * 65)

    dir_path = Path(target_directory)
    tagged_files = defaultdict(dict)
    combined_jsonl_dict = {}

    # ---------------------------------------------------------------
    # PHASE 1: 파일 수집 및 분류
    # ---------------------------------------------------------------
    print("\n📁 [PHASE 1] 원천 파일 탐색 및 파싱 중...")
    for file_path in dir_path.rglob("*"):
        if not file_path.is_file() or "complete" in file_path.parts:
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
        tag_prefix = "_".join(parts[:-1])
        ext = file_path.suffix.lower()

        if file_type == "product" and ext == ".csv":
            tagged_files[tag_prefix]["product"] = file_path
        elif file_type == "text" and ext == ".csv":
            tagged_files[tag_prefix]["text"] = file_path
        elif file_type == "image" and ext == ".csv":
            tagged_files[tag_prefix]["image"] = file_path
        elif file_type == "vl" and ext == ".csv":
            tagged_files[tag_prefix]["vl"] = file_path
        elif file_type == "jl" and ext in [".jsonl", ".jl"]:
            combined_jsonl_dict.update(process_jl_jsonl(file_path))

    # ---------------------------------------------------------------
    # PHASE 2: 신규(INSERT) vs 기존(UPDATE) 구분 및 이미지 분류
    # ---------------------------------------------------------------
    new_product_dfs = []
    standalone_text_dfs = []
    standalone_vl_dfs = []
    
    insert_image_dfs = []  # 신규 상품용 이미지
    update_image_dfs = []  # 기존 상품 alt 업데이트용 이미지

    for tag, files in tagged_files.items():
        has_product = "product" in files

        # 🔵 CASE A: product 파일이 있는 경우 (신규 INSERT 대상)
        if has_product:
            p_df = process_product_csv(files["product"])

            if "text" in files:
                t_df = process_text_csv(files["text"])
                merged_t = pd.merge(p_df, t_df[["brand_name", "product_name", "text_contents"]], on=["brand_name", "product_name"], how="left", suffixes=("", "_y"))
                if "text_contents_y" in merged_t.columns:
                    p_df["text_contents"] = merged_t["text_contents_y"].fillna("")

            if "vl" in files:
                vl_df = process_vl_csv(files["vl"])
                merged_v = pd.merge(p_df, vl_df, on=["brand_name", "product_name"], how="left")
                if "vl_image_text" in merged_v.columns:
                    p_df["image_text"] = merged_v["vl_image_text"].fillna("").astype(str)

            new_product_dfs.append(p_df)

            # 신규 상품의 이미지는 INSERT 목록으로 추가
            if "image" in files:
                insert_image_dfs.append(process_image_csv(files["image"]))

        # 🟡 CASE B: product 파일이 없는 경우 (기존 DB UPDATE 대상)
        else:
            if "text" in files:
                standalone_text_dfs.append(process_text_csv(files["text"]))
            if "vl" in files:
                standalone_vl_dfs.append(process_vl_csv(files["vl"]))
            
            # 기존 상품의 이미지(단독 존재)는 UPDATE 목록으로 추가
            if "image" in files:
                update_image_dfs.append(process_image_csv(files["image"]))

    # ---------------------------------------------------------------
    # PHASE 2-1: 기존 DB UPDATE 처리 (본문, VL)
    # ---------------------------------------------------------------
    if standalone_text_dfs or standalone_vl_dfs:
        update_existing_raw_data(standalone_text_dfs, standalone_vl_dfs)

    # ---------------------------------------------------------------
    # PHASE 2-2: 신규 raw_data_table 적재
    # ---------------------------------------------------------------
    if new_product_dfs:
        print("\n📦 [PHASE 2] 신규 Product 데이터 전처리 & raw_data_table 적재...")
        raw_product_df = pd.concat(new_product_dfs, ignore_index=True)
        p_df = raw_product_df.copy()

        p_df["product_name"] = p_df["product_name"].astype(str).str.strip()

        p_before_len = len(p_df)
        p_df = p_df.drop_duplicates(subset=["product_name"], keep="first")
        p_after_len = len(p_df)
        if p_before_len != p_after_len:
            print(f"   ⚠️ [중복 상품 정제] {p_before_len - p_after_len:,}건의 중복 상품 제거됨")

        p_df["json_ld_contents"] = p_df["product_name"].map(combined_jsonl_dict)
        p_df["has_json_ld"] = p_df["json_ld_contents"].notna()
        p_df["json_ld_contents"] = p_df["json_ld_contents"].apply(convert_to_json_str)

        if "brand_type" not in p_df.columns:
            p_df["brand_type"] = "소상공인"

        p_df["text_contents"] = p_df["text_contents"].fillna("").astype(str)
        p_df["image_text"] = p_df["image_text"].fillna("").astype(str)

        p_brands = p_df["brand_name"].unique().tolist()
        matched_json_count = p_df["has_json_ld"].sum()
        text_filled_count = (p_df["text_contents"] != "").sum()
        image_text_filled_count = (p_df["image_text"] != "").sum()

        print(f"   📊 Product 총 적재 행 수: {len(p_df):,}개")
        print(f"   🎯 JSON-LD 매칭 성공 수: {matched_json_count:,}개 / {len(p_df):,}개")
        print(f"   📝 본문 텍스트 수집 수: {text_filled_count:,}개 / {len(p_df):,}개")
        print(f"   🖼️ 이미지 텍스트(VL) 수집 수: {image_text_filled_count:,}개 / {len(p_df):,}개")
        print(f"   🏷️ 감지된 브랜드 ({len(p_brands)}개): {', '.join(map(str, p_brands))}")

        try:
            raw_target_cols = [
                "brand_name",
                "product_name",
                "product_cat",
                "json_ld_contents",
                "has_json_ld",
                "brand_type",
                "text_contents",
                "image_text",
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
    # PHASE 3: image_data_table 적재 (INSERT) 및 수정 (UPDATE) 분기
    # ---------------------------------------------------------------
    # 1. 신규 상품용 이미지 INSERT
    if insert_image_dfs:
        insert_new_image_data(insert_image_dfs)

    # 2. 기존 상품용 alt 속성 UPDATE
    if update_image_dfs:
        update_existing_alt_data(update_image_dfs)

    if not insert_image_dfs and not update_image_dfs:
        print("\n⚠️ 처리할 이미지 데이터 파일이 없습니다.")

    print("\n" + "=" * 65)
    print("🎉 모든 데이터 적재 및 UPDATE 파이프라인 완료!")
    print("=" * 65)


if __name__ == "__main__":
    run_full_pipeline()