____________________________________________________________________________________________________________________________________________

[ ver1 > premodel2.py ]

반환 데이터 형식: 단순 float / int 숫자 값

결과 분석 상세도: 최종 점수만 확인 가능

하이브리드 검색 방식: BM25 + SBERT (점수 정규화)

JSON-LD 평가 항목: 3개 (파싱, 밀도, 의류속성)

이미지 Alt 평가: 기본적인 문장성 검사

통합 평가 함수: 없음 (메인문에서 직접 계산)

____________________________________________________________________________________________________________________________________________

[ ver2 > premodel2_ver2.py ]

반환 데이터 형식: @dataclass 기반의 구조화된 객체

결과 분석 상세도: 세부 감점 요인, 파싱 결과, 개별 쿼리/이미지 분석 가능

하이브리드 검색 방식: Lexical Overlap(명사 매칭) + SBERT + 임계값(0.5, 0.6)

JSON-LD 평가 항목: 4개 (파싱, 밀도, 의류속성 + 신뢰도/엔티티 연결성)

이미지 Alt 평가: 규칙 강화 (최대 점수 0.75, ETM/EC 어뷰징 감지)

통합 평가 함수: evaluate_page() 단일 통합 함수 지원 (100점 만점)
  ver2 > premodel2_test.ipynb
  premodel2_ver2.py 활용 테스트 파일

____________________________________________________________________________________________________________________________________________

[ ver2 > premodel2_ver201.py ]

입력 데이터 타입 처리 범위: 입력값이 문자열(str)은 물론 이미 파싱된 list나 dict 형태로 들어와도 유연하게 처리할 수 있도록 방어 로직 강화,
                          입력 데이터 최상위가 배열/리스트 형태(isinstance(data, list))일 경우, 첫 번째 요소(data[0])를 자동으로 추출하여 전달

JSON-LD 평가 항목 배점 수정 - 1. 파싱 및 유형 평가 (parsing_score) 부분:
ver2 → Graph 없이 @type: "Product" 단독으로만 세팅되어 있을 때 parsing_score로 0.7점을 부여
ver201 → 소스 2:Product 단독 존재 시의 기준을 완화하여 점수를 0.8점으로 상향 조정

____________________________________________________________________________________________________________________________________________

[ ver2 > premodel2_ver202.py ]

하이브리드 검색 Threshold:
ver201 → 코사인 유사도 0.5 미만 시 0점 처리 + 평균 점수 0.6 미만 시 final_score 0점 처리
ver202 → Threshold 조건 전면 제거

Dataclass 구조 (HybridSearchResult):
ver201 → avg_cosine_sim_thresh, is_passed_threshold 필드 존재
ver202 → Threshold 관련 필드 2개 삭제

JSON-LD 입력 구조 처리:
ver201 → 단순 1차원 리스트(data[0]) 정도만 고려
ver202 → 재귀적 평탄화(_flatten_json_items) 함수 추가로 다중 중첩 리스트 완전 지원

JSON-LD Threshold:
ver201 → raw_score 0.3 미만 시 0점 처리
ver202 → Threshold 조건 제거

디버깅 및 로깅점수:
ver201 → 계산만 수행 (로그 없음)
ver202 → JSON-LD 파싱 내용 및 점수 산출 콘솔 출력(print) 로직 추가  

____________________________________________________________________________________________________________________________________________

[ ver3 > premodel2_ver210.py ]

하이브리드 검색 방식:
ver202 → 전체 질문 리스트를 전달받아 바로 연산
ver210 → 2단계 사전 필터링 도입 (카테고리 1차 → 키워드 2차)

calculate_hybrid_search 입력값:
ver202 → (user_queries, page_text)
ver210 → (user_queries, product_cat, product_name, page_text)

HybridSearchResult Dataclass:
ver202 → 세부 필터 통과 수 관련 필드 없음
ver210 → total_queries_count, cat_filtered_count, must_have_filtered_count 추가

_clean_text 인코딩 방어:
ver202 → 공백 정규화 위주
ver210 → UTF-8 예외 바이트 제거 및 제어 문자 제거 추가

결과 가독성 메서드:
ver202 → formatted_score (속성)
ver210 → formatted_score + print_summary() 메서드 추가  

____________________________________________________________________________________________________________________________________________

[ ver3 > premodel2_ver211.py ]

인코딩 설정 추가: Komoran 사용 시 발생할 수 있는 인코딩 문제를 방지하기 위해 JVM 구동 전 환경 변수(os.environ['JAVA_TOOL_OPTIONS'] = '-Dfile.encoding=UTF-8')를 설정

_clean_text 수정: Null 바이트(\x00, \u0000)를 명시적으로 먼저 제어하고, 보다 안정적인 범주의 제어문자 정규식을 적용

text_ratio 수정:
ver210 → text_ratio >= 0.2일 때는 text_ratio 반환, text_ratio < 0.2일 때는 0점 반환
ver211 → text_ratio < 0.1일 때 0점 반환, 0.1 <= text_ratio < 0.4일 때 (text_ratio-0.1)/0.3 공식으로 선형 점수 부여, text_ratio >= 0.4일 때 1점 반환

하이브리드 검색 속도 개선:
ver210 → 루프 내부에서 각 쿼리마다 embedding_model.encode([query_str])를 개별 실행
ver211 → 유효한 쿼리들을 모아 embedding_model.encode(query_strs)로 한 번에 배치 처리

Komoran 분석 전 문자열 바이트 정제 로직 추가

json-ld 평가 방식 수정:
ver210 → 최상위 객체(Product)의 키값만 직접 비교(attr_keys = [...])
ver211 → 재귀 탐색 함수를 도입: 하위 노드에 숨겨진 속성(material, color, size, category)까지 정확히 수집

ver210 → has_graph 조건이 trust_count 계산 시 무조건 포함되어 중복 가점이 발생하는 문제
ver211 → has_graph 조건을 제거하고 sameAs, shippingDetails, hasMerchantReturnPolicy 실체 항목 점수만 정상 산출하도록 정정

이미지 alt 태그 스터핑 조건 강화:
동일한 명사가 2회 이상 반복 등장(Counter(nouns) >= 2)하는 조건을 스터핑 감지 조건으로 새로 추가
ver210 → 명사가 4개 이상이면 무조건 is_stuffing = True 처리 후 0점 처리
ver211 → 단순 스팸성 스터핑과 단순 단어 나열형 문구를 분리
조사/어미 없이 명사만 4개 이상 나열된 단어 나열형 문구의 경우, 스터핑(0점) 대신 소폭 감점(s3 = 0.08)을 부여하도록 완화

____________________________________________________________________________________________________________________________________________

[ ver4 > premodel2_ver300.py ]

형태소 분석기 변경:
~ver211 → Komoran
ver300 → Kiwi

C++ 기반의 Cpython 모듈인 Kiwi 분석기를 채택하여 Java 의존성을 제거하고 헬퍼 메서드를 통한 깔끔하고 신속한 한국어 파싱 체계를 구축

____________________________________________________________________________________________________________________________________________

[ ver4 > premodel2_ver310.py ]

text_ratio 평가 기준 수정:
ver300 → 0.4 이상 만점
ver310 → 0.5 이상 만점 + 글자 수 가중치(W(N)) 적용  

키워드 스터핑:
ver300 → 감지 시 점수 0점 처리
ver310 → 감점 누적점수 적용 (본문 미존재 시 0점)

이미지 Alt가 빈 값일 때:
ver300 → 이미지가 없으면 1.0점 처리
ver310 → 이미지가 없거나 Alt가 비어 있으면 0점 처리

이미지 Alt - 배점:
ver300 → S1(0.25) / S2(0.25) / S3(0.25)
ver310 → S1(0.20) / S2(0.30) / S3(0.50) (배점 확장)
S1은 alt 속성 존재 여부 - 이걸로 큰 배점을 주는 것이 잘못되었다는 생각

이미지 Alt - 파라미터:
ver300 → is_text_image 사용
ver310 → 이미지를 결합하여 텍스트를 추출했기 때문에 사용할 수 없어짐: is_text_image 제거

____________________________________________________________________________________________________________________________________________


[ ver4 > premodel2_ver320.py ]

Dataclass - KeywordStuffingResult:
ver310 → final_score, raw_score, is_stuffing 등 기본 지표 보유 
ver320 → title_raw_score, body_raw_score 필드 추가
(상품명과 본문의 스터핑 원점수를 구분)  

요약 출력 - print_summary():
ver310 → 키워드 스터핑 세부 점수 출력 시 감점 요소만 표시 
ver320 → 원점수: 상품명(X) | 본문(Y) 항목 추가 출력
(원점수 모니터링 가독성 향상)

키워드 스터핑 - 평가 로직:
ver310 → calculate_keyword_stuffing(page_text) 단일 메서드로 본문만 평가
ver320 → calculate_title_stuffing() 신설, calculate_keyword_stuffing(title_text, page_text)로 확장 (7:3 가중치 적용)
(상품명 키워드 남발/반복 탐지 기능 추가)

이미지 Alt - 무효 패턴 검사:
ver310 → 확장자(jpg 등), 단순 관리코드(img_01 등) 검사
ver320 → 기존 패턴 + 하이픈/언더바 뒤 영문+숫자 난수 코드(-sl3iq, _a1b2c3 등) 패턴 추가
(무의미한 자동 생성 난수 alt 태그 감지율 향상)

Total 평가 - 호출부
ver310 → calculate_keyword_stuffing(page_text) 호출
ver320 → calculate_keyword_stuffing(title_text=product_name, page_text=page_text) 호출
(상품명(product_name) 매개변수 연동)

____________________________________________________________________________________________________________________________________________

[ ver4 > test_with_db_to_df.ipynb ]

DB에서 데이터 50~100건으로 사전 모델 짧게 돌려보는 테스트 파일
ver300부터 320까지 사용한 코드

____________________________________________________________________________________________________________________________________________

[ ver4 > diagnosis_withdb.ipynb ]

DB에서 전체 데이터를 받아 사전 모델을 돌리는 파이프라인
5303개 페이지 돌리는 데 2시간 남짓 소요됨
