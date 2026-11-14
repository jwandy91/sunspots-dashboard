from pathlib import Path

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet


# 실행 환경 확인
is_notebook = "__file__" not in globals()
base_dir = Path.cwd() if is_notebook else Path(__file__).resolve().parent

if not is_notebook:
    st.set_page_config(page_title="🌞 Sunspot Forecast", layout="wide")
    st.title("🌞 Prophet Forecast with Preprocessed Sunspot Data")


# ----------------------------------
# [1] 데이터 불러오기 및 전처리
# ----------------------------------
# 전처리된 CSV를 먼저 찾고, 없으면 원본 CSV를 찾습니다.
candidate_paths = [
    base_dir / "data" / "sunspots_for_prophet.csv",
    base_dir / "sunspots_for_prophet.csv",
    base_dir / "data" / "sunspots.csv",
    base_dir / "sunspots.csv",
]

csv_source = next(
    (path for path in candidate_paths if path.is_file()),
    None
)

if csv_source is None:
    if is_notebook:
        from google.colab import files
        import io

        print("sunspots_for_prophet.csv 또는 sunspots.csv를 선택해주세요.")
        uploaded = files.upload()

        if not uploaded:
            raise ValueError("CSV 파일을 선택해주세요.")

        csv_source = io.BytesIO(next(iter(uploaded.values())))
    else:
        st.info("분석에 사용할 태양흑점 CSV 파일을 업로드해주세요.")
        csv_source = st.file_uploader(
            "sunspots_for_prophet.csv 또는 sunspots.csv",
            type=["csv"]
        )

        if csv_source is None:
            st.stop()

raw_df = pd.read_csv(csv_source)
raw_df.columns = raw_df.columns.str.strip()

if {"ds", "y"}.issubset(raw_df.columns):
    # 이미 전처리된 데이터
    df = raw_df[["ds", "y"]].copy()
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")

elif {"YEAR", "SUNACTIVITY"}.issubset(raw_df.columns):
    # 원본 데이터에서 Prophet용 데이터 생성
    years = pd.to_numeric(raw_df["YEAR"], errors="coerce").astype("Int64")

    df = pd.DataFrame({
        "ds": pd.to_datetime(
            years.astype("string"),
            format="%Y",
            errors="coerce"
        ),
        "y": raw_df["SUNACTIVITY"]
    })

else:
    message = (
        "CSV에 ds와 y 열 또는 YEAR와 SUNACTIVITY 열이 필요합니다."
    )
    if is_notebook:
        raise ValueError(message)
    st.error(message)
    st.stop()

df["y"] = pd.to_numeric(df["y"], errors="coerce")
df = df.dropna(subset=["ds", "y"])

# 학습 기간을 1900~2008년으로 제한
df = (
    df.loc[df["ds"].between("1900-01-01", "2008-01-01")]
    .sort_values("ds")
    .reset_index(drop=True)
)

if df["ds"].nunique() < 2:
    message = "1900~2008년 기간에 학습 가능한 날짜가 2개 이상 필요합니다."
    if is_notebook:
        raise ValueError(message)
    st.error(message)
    st.stop()

if is_notebook:
    print("데이터 미리보기")
    print(df.head())
else:
    st.subheader("📄 데이터 미리보기")
    st.dataframe(df.head())

    st.download_button(
        "전처리된 CSV 다운로드",
        data=df.to_csv(index=False, date_format="%Y-%m-%d"),
        file_name="sunspots_for_prophet.csv",
        mime="text/csv"
    )


# 코랩에서는 그래프 출력, Streamlit에서는 앱에 표시
def show_figure(fig):
    if is_notebook:
        plt.show()
    else:
        st.pyplot(fig)
    plt.close(fig)


# ----------------------------------
# [2] Prophet 모델 정의 및 학습
# ----------------------------------
model = Prophet(
    yearly_seasonality=False,
    weekly_seasonality=False,
    daily_seasonality=False,
    changepoint_prior_scale=0.05,
    seasonality_mode="additive"
)

# period는 일 단위이므로 11년을 일수로 환산
model.add_seasonality(
    name="sunspot_cycle",
    period=11 * 365.25,
    fourier_order=5
)

model.fit(df)


# ----------------------------------
# [3] 향후 30년간 연 단위 예측
# ----------------------------------
future = model.make_future_dataframe(
    periods=30,
    freq="YS",
    include_history=True
)

forecast = model.predict(future)


# ----------------------------------
# [4] Prophet 기본 시각화
# ----------------------------------
if not is_notebook:
    st.subheader("📈 Prophet Forecast Plot")

fig1 = model.plot(forecast)
fig1.suptitle("Sunspot Activity Forecast", fontsize=16)
show_figure(fig1)

if not is_notebook:
    st.subheader("📊 Forecast Components")

fig2 = model.plot_components(forecast)
fig2.suptitle("Forecast Components", fontsize=16)
show_figure(fig2)


# ----------------------------------
# [5] 실제값 vs 예측값 + 예측구간
# ----------------------------------
if not is_notebook:
    st.subheader("📉 Actual vs Predicted with Prediction Intervals")

fig3, ax = plt.subplots(figsize=(14, 6))

ax.fill_between(
    forecast["ds"],
    forecast["yhat_lower"].to_numpy(),
    forecast["yhat_upper"].to_numpy(),
    color="steelblue",
    alpha=0.2,
    label="Prediction interval"
)

ax.plot(
    df["ds"],
    df["y"],
    color="black",
    label="Actual"
)

ax.plot(
    forecast["ds"],
    forecast["yhat"],
    color="steelblue",
    label="Predicted"
)

ax.axvline(
    df["ds"].max(),
    color="gray",
    linestyle="--",
    label="Last observation"
)

ax.set(
    title="Sunspots: Actual vs. Predicted",
    xlabel="Year",
    ylabel="Sunspot Activity"
)
ax.legend()
ax.grid(True, alpha=0.3)
fig3.tight_layout()
show_figure(fig3)


# ----------------------------------
# [6] 잔차 분석 시각화
# ----------------------------------
if not is_notebook:
    st.subheader("📉 Residual Analysis (예측 오차 분석)")

merged = df.merge(
    forecast[["ds", "yhat"]],
    on="ds",
    how="inner"
)
merged["residual"] = merged["y"] - merged["yhat"]

fig4, ax2 = plt.subplots(figsize=(14, 4))

ax2.plot(
    merged["ds"],
    merged["residual"],
    color="teal",
    label="Residual"
)

ax2.axhline(
    0,
    color="black",
    linestyle="--",
    linewidth=1,
    label="Zero line"
)

ax2.set(
    title="Residual Analysis (Actual - Predicted)",
    xlabel="Year",
    ylabel="Residual"
)
ax2.legend()
ax2.grid(True, alpha=0.3)
fig4.tight_layout()
show_figure(fig4)


# ----------------------------------
# [7] 잔차 통계 요약
# ----------------------------------
summary = merged["residual"].describe().to_frame()

if is_notebook:
    print("\n잔차 통계 요약")
    print(summary)
else:
    st.subheader("📌 Residual Summary Statistics")
    st.dataframe(summary)
