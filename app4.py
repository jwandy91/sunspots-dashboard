from pathlib import Path

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet


# 코랩 셀과 파이썬 파일 실행 환경 구분
is_notebook = "__file__" not in globals()
base_dir = Path.cwd() if is_notebook else Path(__file__).resolve().parent

if not is_notebook:
    st.set_page_config(page_title="🌞 Sunspot Forecast", layout="wide")
    st.title("🌞 Prophet Forecast with Preprocessed Sunspot Data")


# ----------------------------------
# [1] 데이터 불러오기
# ----------------------------------
csv_path = base_dir / "data" / "sunspots_for_prophet.csv"

if not csv_path.is_file():
    csv_path = base_dir / "sunspots_for_prophet.csv"

if not csv_path.is_file():
    message = (
        "sunspots_for_prophet.csv 파일을 찾을 수 없습니다. "
        "현재 작업 폴더 또는 data 폴더에 파일을 업로드해주세요."
    )
    if is_notebook:
        raise FileNotFoundError(message)
    st.error(message)
    st.stop()

df = pd.read_csv(csv_path, parse_dates=["ds"])
df = (
    df[["ds", "y"]]
    .dropna()
    .sort_values("ds")
    .reset_index(drop=True)
)

if is_notebook:
    print("데이터 미리보기")
    print(df.head())
else:
    st.subheader("📄 데이터 미리보기")
    st.dataframe(df.head())


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

# period의 단위는 일입니다.
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
    df["ds"], df["y"],
    color="black",
    label="Actual"
)

ax.plot(
    forecast["ds"], forecast["yhat"],
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
    merged["ds"], merged["residual"],
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
