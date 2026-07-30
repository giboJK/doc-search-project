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

data_frame = load_data(DATA_PATH)
explore_structure(data_frame)
show_category_distribution(data_frame)
check_missing(data_frame)