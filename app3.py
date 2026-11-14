from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df["YEAR_INT"] = df["YEAR"].astype(int)
    df["DATE"] = pd.to_datetime(df["YEAR_INT"].astype(str), format="%Y")
    return df.set_index("DATE").sort_index()


def plot_advanced_sunspot_visualizations(
    df, sunactivity_col="SUNACTIVITY", hist_bins=30,
    trend_degree=1, point_size=10, point_alpha=0.5
):
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Sunspots Data Advanced Visualization", fontsize=18)

    activity = df[sunactivity_col].dropna()
    data = activity.to_numpy(dtype=float)
    years = activity.index.year.to_numpy()

    # (a) 선택한 연도 범위의 시계열
    ax = axs[0, 0]
    ax.plot(df.index, df[sunactivity_col], color="blue")
    ax.set(title="Sunspot Activity Over Time", xlabel="Year", ylabel="Sunspot Count")
    ax.grid(True)

    # (b) 히스토그램과 밀도 곡선
    ax = axs[0, 1]
    if len(data) > 0:
        ax.hist(
            data, bins=hist_bins, density=True,
            alpha=0.6, color="gray", label="Histogram"
        )
        if data.min() < data.max():
            xs = np.linspace(data.min(), data.max(), 200)
            density = gaussian_kde(data)
            ax.plot(xs, density(xs), color="red", linewidth=2, label="Density")
        ax.legend()

    ax.set(title="Distribution of Sunspot Activity", xlabel="Sunspot Count", ylabel="Density")
    ax.grid(True)

    # (c) 선택한 기간 중 1900~2000년에 해당하는 데이터의 상자 그림
    ax = axs[1, 0]
    activity_20th = activity.loc["1900":"2000"]
    if not activity_20th.empty:
        ax.boxplot(activity_20th, vert=False)
    else:
        ax.text(
            0.5, 0.5, "No data for 1900-2000 in the selected range",
            ha="center", va="center", transform=ax.transAxes
        )
    ax.set(title="Boxplot of Sunspot Activity (1900-2000)", xlabel="Sunspot Count")

    # (d) 산점도와 선택한 차수의 추세선
    ax = axs[1, 1]
    if len(data) > 0:
        ax.scatter(years, data, s=point_size, alpha=point_alpha, label="Data Points")

        # 예를 들어 3차 추세선에는 서로 다른 연도가 4개 이상 필요
        if len(np.unique(years)) > trend_degree:
            trend = np.polynomial.Polynomial.fit(years, data, trend_degree)
            x_trend = np.linspace(years.min(), years.max(), 100)
            ax.plot(x_trend, trend(x_trend), color="red", linewidth=2, label="Trend Line")
        ax.legend()

    ax.set(title="Trend of Sunspot Activity", xlabel="Year", ylabel="Sunspot Count")
    ax.grid(True)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


# 메인 앱
st.title("🌞 태양흑점 데이터 분석 대시보드 🌞")
st.write("사이드바에서 연도 범위와 그래프 설정을 조절할 수 있습니다.")

# app3.py가 있는 폴더를 기준으로 CSV를 찾음
app_dir = Path(__file__).resolve().parent
csv_path = app_dir / "data" / "sunspots.csv"

if not csv_path.is_file():
    csv_path = app_dir / "sunspots.csv"

if not csv_path.is_file():
    st.error("sunspots.csv 파일을 찾을 수 없습니다.")
    st.info("GitHub 저장소에서 app3.py 옆이나 data 폴더 안에 sunspots.csv를 올려주세요.")
    st.stop()

df = load_data(csv_path)

if df.empty:
    st.warning("CSV 파일에 데이터가 없습니다.")
    st.stop()

# 사이드바 설정
st.sidebar.header("시각화 설정")

min_year = int(df["YEAR_INT"].min())
max_year = int(df["YEAR_INT"].max())

year_range = st.sidebar.slider(
    "연도 범위", min_year, max_year, (min_year, max_year)
)
hist_bins = st.sidebar.slider("히스토그램 구간 수", 5, 100, 30)
trend_degree = st.sidebar.slider("추세선 차수", 1, 5, 1)
point_size = st.sidebar.slider("산점도 점 크기", 1, 100, 10)
point_alpha = st.sidebar.slider("산점도 투명도", 0.0, 1.0, 0.5, step=0.05)

# 선택한 기간의 데이터로 그래프 그리기
filtered_df = df.loc[df["YEAR_INT"].between(year_range[0], year_range[1])]

if filtered_df.empty:
    st.warning("선택한 기간에 데이터가 없습니다.")
else:
    st.subheader("태양흑점 데이터 종합 시각화")
    fig = plot_advanced_sunspot_visualizations(
        filtered_df,
        hist_bins=hist_bins,
        trend_degree=trend_degree,
        point_size=point_size,
        point_alpha=point_alpha
    )
    st.pyplot(fig)
    plt.close(fig)
