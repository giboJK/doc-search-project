import pandas as pd
import numpy as np
import sklearn
import os
import sys

DATA_PATH = "../data/tech_docs.csv"


def load_data(path: str) -> pd.DataFrame:
    """
    CSV 파일을 불러와 DataFrame으로 반환한다.

    매개변수:
        path (str): CSV 파일 경로
    반환:
        pd.DataFrame: 불러온 데이터
    """

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



data_frame = load_data(DATA_PATH)
explore_structure(data_frame)
