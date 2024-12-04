import streamlit as st
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Streamlit UI 설정
st.title("소비자심리지수(CSI) 회귀분석")
st.sidebar.header("분석 옵션")


# 1. 데이터 로드
@st.cache_data
def load_data():
    file_path = "소비자동향조사_filtered_df.csv"  # CSV 파일 경로
    return pd.read_csv(file_path)


data = load_data()

# 2. 종속변수 선택
selected_csi_codes = [
    '의류비 지출전망CSI', '외식비 지출전망CSI', '여행비 지출전망CSI',
    '교육비 지출전망CSI', '의료·보건비 지출전망CSI', '교양·오락·문화생활비 지출전망CSI',
    '교통비 및 통신비 지출전망CSI', '주거비 지출전망CSI'
]
target_csi = st.sidebar.selectbox("종속 변수 선택", selected_csi_codes)

# 3. CSI분류코드 선택
income_ranges = ['전체', '100만원미만', '100-200만원', '200-300만원', '300-400만원', '400-500만원', '500만원이상']
income_range = st.sidebar.selectbox("소득 범위 선택", income_ranges)

# 4. 독립변수 선택
mandatory_var = '현재가계부채CSI'
optional_vars = ['가계부채전망CSI', '물가수준전망(1년후)CSI', '가계수입전망CSI', '금리수준전망CSI']
independent_vars = st.sidebar.multiselect(
    "독립 변수 선택 (필수: 현재가계부채CSI 포함)",
    optional_vars,
    default=[optional_vars[0]]
)

# 필수 독립변수 추가 (중복 방지)
if mandatory_var not in independent_vars:
    independent_vars.insert(0, mandatory_var)


# 5. 데이터 전처리 함수
def preprocess_data(data, target_csi, income_range, independent_vars):
    # 필요한 종속변수와 독립변수 선택
    filtered = data[data['CSI코드'].isin([target_csi] + independent_vars)]

    # 데이터를 날짜별로 변환
    reshaped = filtered.melt(
        id_vars=['CSI코드', 'CSI분류코드'],
        var_name='Date',
        value_name='Value'
    )
    reshaped = reshaped.pivot_table(
        index=['Date', 'CSI분류코드'],
        columns='CSI코드',
        values='Value'
    ).reset_index()

    # 날짜 형식 변환
    reshaped['Date'] = pd.to_datetime(reshaped['Date'], format='%b-%y', errors='coerce')
    reshaped = reshaped.dropna(subset=['Date'])  # 변환 실패한 날짜 제거

    # 소득 범위 필터링
    reshaped = reshaped[reshaped['CSI분류코드'] == income_range].dropna()

    # 날짜 기준으로 정렬
    reshaped = reshaped.sort_values(by='Date').reset_index(drop=True)

    return reshaped


# 6. 분석 실행 버튼
if st.sidebar.button("분석 실행"):
    if len(independent_vars) < 2:  # 최소 독립변수 수 체크
        st.error("현재가계부채CSI 외에 최소 하나의 독립 변수를 선택해야 합니다!")
    else:
        # 데이터 전처리
        processed_data = preprocess_data(data, target_csi, income_range, independent_vars)

        # 회귀 분석 실행
        X = processed_data[independent_vars]
        y = processed_data[target_csi]
        X = sm.add_constant(X)  # 상수항 추가
        model = sm.OLS(y, X).fit()

        # 회귀분석 결과 출력
        st.subheader(f"회귀분석 결과 ({income_range})")
        st.text(model.summary())

        # 예측값 추가 및 시각화
        processed_data['Predicted'] = model.predict(X)

        # 데이터 확인
        st.write("분석된 데이터 샘플:")


        # 그래프 생성
        plt.figure(figsize=(12, 6))
        plt.plot(processed_data['Date'], y, label="Actual", marker="o", linestyle="-")
        plt.plot(processed_data['Date'], processed_data['Predicted'], label="Predicted", linestyle="--")
        plt.title(f"{income_range} - Actual vs Predicted {target_csi}")
        plt.xlabel("Date")
        plt.ylabel("CSI Value")
        plt.legend()
        plt.grid()
        st.pyplot(plt)
