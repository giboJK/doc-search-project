import pandas as pd
import numpy as np
import sklearn
import os
import sys

DATA_PATH = "../data/tech_docs2.csv"


def load_data(path: str) -> pd.DataFrame:
    """
    CSV 파일을 불러와 DataFrame으로 반환한다.

    매개변수:
        path (str): CSV 파일 경로
    반환:
        pd.DataFrame: 불러온 데이터
    """
    print("load_data() 함수 실행")
    # 1) 파일 존재 여부 확인
    if not os.path.exists(path):
        print(f"파일을 찾을 수 없습니다: {path}")
        sys.exit(1)

    # 2) CSV 읽기 (한글·특수문자 대비 인코딩 지정)
    data_frame = pd.read_csv(path, encoding="utf-8-sig")

    # 3) 로드 완료 메시지 출력
    rows, cols = data_frame.shape
    print(f"데이터 로드 완료: {rows}행 × {cols}열\n")

    return data_frame

def explore_structure(data_frame: pd.DataFrame):
    """
    DataFrame의 기본 구조(행/열 수, 컬럼·자료형, 상위 5행)를 출력한다.
    """
    print("explore_structure() 함수 실행")
    # 1) 행 수 / 열 수
    print("=" * 20)
    print("행/열 수")
    rows, cols = data_frame.shape
    print(f"행 수: {rows}, 열 수: {cols}\n")

    # 2) 컬럼 이름 + 자료형
    print("=" * 20)
    print("컬럼별 자료형")
    print(data_frame.dtypes)
    print("\n")

    # 3) 상위 5행 미리보기
    print("=" * 20)
    print("상위 5행 미리보기")
    print(data_frame.head(5))
    print("\n")

    # 4) info 출력
    print("=" * 20)
    print("info 출력")
    print(data_frame.info())
    print("\n")


def show_category_distribution(data_frame: pd.DataFrame) -> dict:
    """
    카테고리별 문서 수·비율(%)과 평균 단어 수를 계산·출력하고 딕셔너리로 반환한다.
    """
    print("show_category_distribution() 함수 실행")
    total = len(data_frame)                                   # 전체 문서 수
    category_counts = data_frame["category"].value_counts()   # 카테고리별 문서 수 (많은 순)

    # 1) 카테고리별 문서 수 + 비율(%)
    print("=" * 20)
    print("[카테고리 분포] 문서 수 & 비율")
    for category, count in category_counts.items():
        ratio = count / total * 100
        print(f"{category}: {count}개 ({ratio:.1f}%)")
    print("\n")

    # 2) 반복문 + 딕셔너리로 카테고리별 '평균 단어 수' 계산
    print("=" * 20)
    print("[카테고리 분포] 평균 단어 수")
    avg_word_counts = {}                                  # 결과 담을 딕셔너리
    for category in data_frame["category"].unique():      # 고유 카테고리 목록 하나씩 (반복문)
        subset = data_frame[data_frame["category"] == category]   # 이 카테고리 행만 필터링
        word_counts = []                                  # 문서별 단어 수 모을 리스트

        for content in subset["content"]:
            if pd.isnull(content):                    # 내용이 결측치인 문서는 평균 계산에서 제외
                continue
            word_counts.append(len(content.split()))  # 공백 기준으로 나눈 총 단어의 수

        # 이 카테고리의 문서 내용이 모두 결측치이면 평균을 낼 수 없음
        if not word_counts:
            avg_word_counts[category] = 0.0
            print(f"{category}: 내용이 모두 결측치여서 평균을 계산할 수 없습니다")
            continue

        avg = sum(word_counts) / len(word_counts)     # 전체 단어수의 평균값
        avg_word_counts[category] = avg               # 딕셔너리에 저장
        print(f"{category}: 평균 {avg:.1f} 단어")       # 소수점 1자리까지 출력
    print("\n")

    # 3) 결과를 딕셔너리로 반환
    result = {}
    for category in category_counts.index:
        result[category] = {
            "count": int(category_counts[category]),
            "ratio_percent": round(category_counts[category] / total * 100, 1),
            "avg_word_count": round(avg_word_counts[category], 1),
        }
    return result

def check_missing(data_frame: pd.DataFrame) -> dict:
    """
    컬럼별 결측치 수·비율(%)과 심각도를 파악하고 딕셔너리로 반환한다.
    """
    print("check_missing() 함수 실행")
    total = len(data_frame)                      # 전체 행 수
    missing_counts = data_frame.isnull().sum()   # 컬럼별 결측치 수 (Series)

    print("=" * 20)
    print("[결측치] 현황")
    result = {}          # 결과 딕셔너리
    missing_cols = []    # 결측치가 있는 컬럼 모음
    clean_cols = []      # 결측치가 없는 컬럼 모음

    for col in data_frame.columns:       # 컬럼 하나씩
        missing_count = missing_counts[col]      # 이 컬럼 결측치 수
        ratio = missing_count / total * 100      # 비율(%)

        if missing_count > 0:
            # 비율 기준 심각도 판단
            if ratio < 5:
                level = "낮음"
            elif ratio < 20:
                level = "주의"
            else:
                level = "높음"
            print(f"{col}: {missing_count}개 ({ratio:.1f}%) → 심각도: {level}")
            missing_cols.append(col)
        else:
            clean_cols.append(col)

        result[col] = {"count": int(missing_count), "ratio_percent": round(ratio, 1)}

    # 결측치가 있는 컬럼이 하나도 없을 때
    if not missing_cols:
        print("결측치가 있는 컬럼: 없음")

    # 결측치가 없는 컬럼 목록
    print(f"결측치가 없는 컬럼: {', '.join(clean_cols)}")
    print("\n")

    return result

def numpy_doc_stats(data_frame: pd.DataFrame) -> dict:
    """
    NumPy 함수로 문서 길이(단어 수)의 통계량 5가지(평균, 표준편차, 중앙값, 최솟값, 최댓값)를
    계산·출력하고 딕셔너리로 반환한다.
    결측치가 있는 행은 배열 생성 전에 제거한다.
    조건 필터링으로 50단어 미만 문서를 찾아 출력한다.
    pandas describe()로 계산한 결과와 수치를 비교해 일치하는지 확인하는 출력을 포함한다.
    """
    print("numpy_doc_stats() 함수 실행")

    # 1) 배열을 만들기 전에, 결측치가 있는 행을 먼저 제거
    clean_df = data_frame.dropna(axis=0, how="any")
    removed = len(data_frame) - len(clean_df)
    if removed > 0:
        print(f"결측치로 제외된 행 수: {removed}\n")
    else:
        print("결측치로 제외된 행 수: 없음\n")

    # 제거 후 데이터가 없으면 처리 중단
    if clean_df.empty:
        print("모든 행이 결측치로 제거되어 통계를 계산할 수 없습니다.\n")
        return {}

    # 2) content 각 행을 단어 수로 바꿔 리스트에 모은 뒤, NumPy 배열로 변환
    word_count_list = []                              # 문서별 단어 수 모을 리스트
    for content in clean_df["content"]:
        word_count_list.append(len(content.split()))  # 공백 기준으로 나눈 총 단어의 수
    lengths = np.array(word_count_list)               # 리스트 → NumPy 배열

    # 3) NumPy 함수로 5가지 통계량 계산
    mean_count = float(np.mean(lengths))
    std_count = float(np.std(lengths, ddof=1))   # ddof=1: pandas와 같은 표본표준편차
    median_count = float(np.median(lengths))
    min_count = int(np.min(lengths))
    max_count = int(np.max(lengths))

    print("=" * 20)
    print("[문서 길이 통계량] NumPy 계산")
    print(f"평균: {mean_count:.1f} 단어")
    print(f"표준편차: {std_count:.1f}")
    print(f"중앙값: {median_count:.1f}")
    print(f"최솟값: {min_count}")
    print(f"최댓값: {max_count}")
    print("\n")

    # 4) 조건 필터링으로 50단어 미만 문서 찾아 출력
    print("=" * 20)
    print("[50단어 미만 문서]")
    is_short = lengths < 50              # 조건에 맞는 위치만 True인 배열
    short_lengths = lengths[is_short]    # 조건 필터링으로 단어 수만 추출
    short_docs = clean_df[is_short]      # 같은 조건으로 문서 행도 추출
    print(f"총 {len(short_lengths)}개")
    if len(short_lengths) == 0:
        print("50단어 미만 문서가 없습니다.")
    else:
        for doc_id, title, length in zip(short_docs["doc_id"], short_docs["title"], short_lengths):
            print(f"{doc_id} | {title} | {length}단어")
    print("\n")

    # 5) pandas describe() 결과와 수치를 비교해 일치하는지 확인
    word_counts_series = pd.Series(word_count_list)   # 같은 단어 수를 pandas Series로
    described = word_counts_series.describe()

    print("=" * 20)
    print("[pandas describe()와 비교]")
    print(described)
    print("")

    # describe()의 각 항목과 NumPy 계산값을 하나씩 대조
    comparisons = [
        ("평균", mean_count, float(described["mean"])),
        ("표준편차", std_count, float(described["std"])),
        ("중앙값", median_count, float(described["50%"])),
        ("최솟값", min_count, float(described["min"])),
        ("최댓값", max_count, float(described["max"])),
    ]

    all_match = True
    for name, numpy_value, pandas_value in comparisons:
        matched = np.isclose(numpy_value, pandas_value)   # 부동소수점 오차를 감안한 비교
        if not matched:
            all_match = False
        mark = "일치" if matched else "불일치"
        print(f"{name}: NumPy {numpy_value:.4f} / pandas {pandas_value:.4f} → {mark}")

    if all_match:
        print("→ 5가지 통계량이 pandas describe() 결과와 모두 일치합니다.")
    else:
        print("→ 일치하지 않는 통계량이 있습니다.")
    print("\n")

    # 6) 결과를 딕셔너리로 반환
    result = {
        "mean": round(mean_count, 1),
        "std": round(std_count, 1),
        "median": round(median_count, 1),
        "min": min_count,
        "max": max_count,
        "removed_rows": removed,
        "total_after_clean": len(clean_df),
        "short_docs_count": int(len(short_lengths)),
        "matches_pandas": all_match,
    }
    return result

data_frame = load_data(DATA_PATH)
explore_structure(data_frame)
show_category_distribution(data_frame)
check_missing(data_frame)
numpy_doc_stats(data_frame)