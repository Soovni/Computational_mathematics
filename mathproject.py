import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import scipy.stats as stats
import numpy as np


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

        # 잔차 계산
        processed_data['Residuals'] = model.resid

        # 데이터 확인
        st.write("분석된 데이터 샘플:")
        st.write(processed_data.head())

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

        # 피어슨 상관계수 계산 및 출력
        st.subheader("피어슨 상관계수 분석 (종속변수 포함)")
        correlation_matrix = processed_data[[target_csi] + independent_vars].corr(method='pearson')
        st.write("상관계수 매트릭스:")
        st.write(correlation_matrix)

        # 다중공선성 확인 (VIF)
        st.subheader("다중공선성 확인 (VIF)")
        vif_data = pd.DataFrame()
        vif_data["변수"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        st.write(vif_data)

        max_vif = vif_data[vif_data["변수"] != "const"]["VIF"].max()

        if max_vif > 10:

            st.warning("VIF 값이 10을 초과하는 변수가 있습니다. 다중공선성을 의심할 수 있습니다.")

        else:
            st.success("VIF 값이 모두 10 이하로, 다중공선성 문제가 없습니다.")



        # 잔차 정규성 검토
        st.subheader("잔차 정규성 검토")
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # 히스토그램
        axs[0].hist(processed_data['Residuals'], bins=20, edgecolor='black', alpha=0.7)
        axs[0].set_title("Residuals Histogram")
        axs[0].set_xlabel("Residuals")
        axs[0].set_ylabel("Frequency")

        # Q-Q Plot
        sm.qqplot(processed_data['Residuals'], line='s', ax=axs[1])
        axs[1].set_title("Q-Q Plot of Residuals")

        st.pyplot(fig)

        # Jarque-Bera 테스트
        jb_stat, jb_pval = stats.jarque_bera(processed_data['Residuals'])
        st.write("Jarque-Bera Test:")
        st.write(f"Test Statistic: {jb_stat:.3f}, P-value: {jb_pval:.3f}")
        if jb_pval < 0.05:
            st.warning("잔차가 정규성을 따르지 않을 가능성이 높습니다.")

        # Durbin-Watson 통계량 출력
        dw_stat = sm.stats.stattools.durbin_watson(model.resid)
        st.subheader("Durbin-Watson 통계량")
        st.write(f"Durbin-Watson Statistic: {dw_stat:.3f}")
        if dw_stat < 1.5 or dw_stat > 2.5:
            st.warning("Durbin-Watson 값이 1.5~2.5를 벗어났습니다. 자기상관 문제가 있을 수 있습니다.")



