import pandas as pd
import numpy as np
import os
import re
import sys

# 파일 경로는 상수로 한 곳에서 관리한다.
DATA_PATH = "../data/tech_docs.csv"

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

def clean_text(text: str) -> str:
    """
    문자열 하나를 정제한다. (소문자 변환 → 특수문자 제거 → 중복 공백 정리)
    매개변수:
        text (str): 정제할 원본 문자열
    반환:
        str: 정제된 문자열
    """
    # 1) 소문자로 변환
    text = text.lower()

    # 2) 영문·숫자·공백만 남기고 특수문자는 공백으로 치환
    #    (바로 지우면 앞뒤 단어가 붙어버리므로 공백으로 바꾼다)
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # 3) 중복 공백을 하나로 정리하고 앞뒤 공백 제거
    text = re.sub(r"\s+", " ", text).strip()

    return text

def preprocess(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    content 컬럼을 정제해 content_clean 컬럼을 새로 만든 DataFrame을 반환한다.
    content가 결측치인 행은 정제 전에 제거한다.

    매개변수:
        data_frame (pd.DataFrame): 원본 데이터
    반환:
        pd.DataFrame: content_clean 컬럼이 추가된 데이터
    """
    print("preprocess() 함수 실행")

    # 1) content에 결측치가 있는 행을 먼저 제거
    #    (원본을 건드리지 않도록 복사본을 만들어 사용한다)
    clean_df = data_frame.dropna(subset=["content"]).copy()
    removed = len(data_frame) - len(clean_df)
    if removed > 0:
        print(f"content 결측치로 제외된 행 수: {removed}")
    else:
        print("content 결측치로 제외된 행 수: 없음")

    # 2) content 컬럼 전체에 clean_text()를 적용해 content_clean 컬럼 생성
    clean_df["content_clean"] = clean_df["content"].apply(clean_text)

    print(f"전처리 완료: {len(clean_df)}행 (content_clean 컬럼 추가)\n")

    return clean_df



def cosine_similarity_numpy(vec1: np.ndarray, vec2: np.ndarray) -> float:
    # 벡터의 내적 계산
    dot_product = np.dot(vec1, vec2)
    # 벡터의 노름 계산
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    # 코사인 유사도 계산
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0
    return dot_product / (norm_vec1 * norm_vec2)

def main():
    print("=" * 20)
    print("week2 텍스트 전처리")
    print(f"데이터 파일: {DATA_PATH}")
    print("=" * 20)
    print("\n")

    # [1] 데이터 불러오기
    print("[1] 데이터 불러오기")
    data_frame = load_data(DATA_PATH)

    # [2] 텍스트 전처리
    print("[2] 텍스트 전처리")
    clean_df = preprocess(data_frame)

    # 확인: 원본 content와 정제된 content_clean 비교 (상위 3행)
    print("=" * 20)
    print("[content / content_clean] 상위 3행")
    print(clean_df[["content", "content_clean"]].head(3))
    print("\n")

    # [3] NumPy로 문서 길이 통계량 계산
    print("[3] NumPy로 문서 길이 통계량 계산")
    print(cosine_similarity_numpy(np.array([1, 2, 3]), np.array([1, 2, 3])))
    print(cosine_similarity_numpy(np.array([1, 0, 0]), np.array([0, 0, 1])))


    print("=" * 20)
    print("전체 실행 완료")
    print("=" * 20)


if __name__ == "__main__":
    main()
