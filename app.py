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


def plot_advanced_sunspot_visualizations(df, sunactivity_col="SUNACTIVITY"):
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Sunspots Data Advanced Visualization", fontsize=18)
    activity = df[sunactivity_col].dropna()

    # (a) 전체 시계열 라인 차트
    ax = axs[0, 0]
    ax.plot(df.index, df[sunactivity_col], color="steelblue", linewidth=1)
    ax.set(title="Sunspot Activity Over Time", xlabel="Year", ylabel="Sunspot Count")
    ax.grid(True)

    # (b) 히스토그램과 커널 밀도 곡선
    ax = axs[0, 1]
    data = activity.to_numpy(dtype=float)
    if len(data) > 0:
        ax.hist(data, bins=30, density=True, alpha=0.6, color="gray", label="Histogram")
        if len(data) > 1 and data.min() < data.max():
            xs = np.linspace(data.min(), data.max(), 200)
            density = gaussian_kde(data)
            ax.plot(xs, density(xs), color="red", linewidth=2, label="Density")
        ax.legend()
    ax.set(title="Distribution of Sunspot Activity", xlabel="Sunspot Count", ylabel="Density")
    ax.grid(True)

    # (c) 1900년부터 2000년까지의 상자 그림
    ax = axs[1, 0]
    activity_20th = activity.loc["1900":"2000"]
    if not activity_20th.empty:
        ax.boxplot(activity_20th, vert=False)
    ax.set(title="Boxplot of Sunspot Activity (1900-2000)", xlabel="Sunspot Count")

    # (d) 연도별 산점도와 회귀선
    ax = axs[1, 1]
    years = activity.index.year.to_numpy()
    if len(years) > 0:
        ax.scatter(years, data, s=10, alpha=0.5, label="Data Points")
        if len(np.unique(years)) > 1:
            trend = np.poly1d(np.polyfit(years, data, 1))
            ax.plot(years, trend(years), color="red", linewidth=2, label="Trend Line")
        ax.legend()
    ax.set(title="Trend of Sunspot Activity", xlabel="Year", ylabel="Sunspot Count")
    ax.grid(True)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


# 메인 앱
st.title("🌞 태양흑점 데이터 분석 대시보드 🌞")
st.write("이 대시보드는 태양흑점 데이터를 다양한 시각화 방법으로 보여줍니다.")

# app.py가 있는 폴더를 기준으로 CSV 위치 확인
app_dir = Path(__file__).resolve().parent
file_path = app_dir / "data" / "sunspots.csv"

if not file_path.is_file():
    file_path = app_dir / "sunspots.csv"

if not file_path.is_file():
    st.error("sunspots.csv를 찾을 수 없습니다.")
    st.info("GitHub 저장소의 data 폴더 또는 app.py와 같은 폴더에 sunspots.csv를 업로드하세요.")
    st.stop()

df = load_data(file_path)

if df.empty:
    st.warning("데이터가 없습니다.")
else:
    st.subheader("태양흑점 데이터 종합 시각화")
    fig = plot_advanced_sunspot_visualizations(df)
    st.pyplot(fig)
    plt.close(fig)
