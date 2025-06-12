# 파일: app.py
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import matplotlib.font_manager as fm
import platform
import os
import subprocess
import pandas as pd
from PIL import Image
from utils import load_and_preprocess_data, train_model, predict_test
from visuals import (
    plot_survival_by_sex,
    plot_survival_by_class,
    plot_age_distribution,
)

# 한글 폰트 설정 함수
def set_korean_font():
    system = platform.system()
    if system == 'Darwin':
        plt.rcParams['font.family'] = 'AppleGothic'
    elif system == 'Windows':
        plt.rcParams['font.family'] = 'Malgun Gothic'
    else:
        font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
        if not os.path.exists(font_path):
            try:
                subprocess.run(["sudo", "apt", "update"], check=True)
                subprocess.run(["sudo", "apt", "install", "-y", "fonts-nanum"], check=True)
                fm._rebuild()
            except Exception as e:
                print("폰트 설치 오류:", e)
        plt.rcParams['font.family'] = 'NanumGothic'
    plt.rcParams['axes.unicode_minus'] = False

# 페이지 설정
st.set_page_config(page_title="🚢 타이타닉 생존자 대시보드", layout="wide")
st.title("🚢 타이타닉 생존자 대시보드")

# 메뉴 구성
# 🔍 탐색 메뉴 (사이드바)
st.sidebar.markdown("<h3 style='font-size:22px; margin-bottom:10px;'>🔍 탐색 메뉴</h3>", unsafe_allow_html=True)

menu = st.sidebar.radio("", (
    "🏠 홈",
    "📊 데이터 탐색",
    "📍 탑승항 분석",
    "📈 상관관계 히트맵",
    "🤖 예측 모델",
    "📁 결과 다운로드"
))

# 데이터 로딩
train, test, gender, X_train, y_train, X_test, test_ids = load_and_preprocess_data()

# -------------------------
# 🏠 홈: 소개 + 설명 + 이미지
# -------------------------
if menu == "🏠 홈":
    st.header("📖 프로젝트 소개")

    st.markdown("""
<h3>💡 프로젝트 목적</h3>
<p>
본 웹 애플리케이션은 1912년 침몰한 <strong>RMS 타이타닉</strong>호의 승객 데이터를 기반으로<br>
승객의 <strong>생존 여부를 분석하고 예측</strong>하는 데이터 분석 프로젝트입니다.
</p>

<h3>📌 주요 기능</h3>
<ul>
  <li>탐색적 데이터 분석 (EDA): 성별, 객실 등급, 나이 등 변수별 생존 분포 및 생존률 확인</li>
  <li>머신러닝 모델 학습: 랜덤 포레스트 분류기를 이용한 생존 예측</li>
  <li>결과 시각화 및 다운로드: 예측 결과를 표와 그래프로 확인하고 CSV로 저장 가능</li>
</ul>

<h3>🚀 사용 방법</h3>
<ol>
  <li>좌측 메뉴에서 원하는 분석 메뉴를 선택하세요.</li>
  <li><code>예측 모델</code> 메뉴에서는 필터를 이용해 조건별 승객을 선택하고, 해당 조건에 따른 생존 여부를 예측할 수 있습니다.</li>
  <li><code>결과 다운로드</code> 메뉴에서 예측 결과를 CSV로 받을 수 있습니다.</li>
</ol>
""", unsafe_allow_html=True)

    # 이미지 표시
    image = Image.open("titanic.jpeg")
    st.image(image, caption="🚢 RMS 타이타닉 (1912년 실제 촬영)", use_container_width=True)

# -------------------------
# 📊 데이터 탐색
# -------------------------
elif menu == "📊 데이터 탐색":
    st.header("📊 데이터 탐색")

    # ▶️ 기본 분포 시각화
    col1, col2, col3 = st.columns(3)
    with col1:
        st.plotly_chart(plot_survival_by_sex(train), use_container_width=True)
    with col2:
        st.plotly_chart(plot_survival_by_class(train), use_container_width=True)
    with col3:
        st.plotly_chart(plot_age_distribution(train), use_container_width=True)

    # ▶️ 생존률 기반 추가 시각화
    st.subheader("📊 생존률 분석")

    # 결측치 처리 및 나이 그룹 생성
    train['Age'] = train['Age'].fillna(train['Age'].median())
    train['AgeGroup'] = pd.cut(train['Age'], bins=[0, 12, 20, 40, 60, 80],
                               labels=["0-12", "13-20", "21-40", "41-60", "61+"])

    # 생존률 계산
    sex_surv = train.groupby('Sex')['Survived'].mean().reset_index()
    sex_surv['Sex'] = sex_surv['Sex'].map({'male': '남성', 'female': '여성'})

    pclass_surv = train.groupby('Pclass')['Survived'].mean().reset_index()
    agegroup_surv = train.groupby('AgeGroup')['Survived'].mean().reset_index()

    # ▶️ 생존률 그래프
    col4, col5, col6 = st.columns(3)
    with col4:
        fig4 = px.bar(sex_surv, x='Sex', y='Survived',
                      title='성별 생존률',
                      labels={'Survived': '생존률'},
                      color='Sex',
                      color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig4, use_container_width=True)

    with col5:
        fig5 = px.bar(pclass_surv, x='Pclass', y='Survived',
                      title='객실 등급 생존률',
                      labels={'Survived': '생존률'},
                      color='Pclass',
                      color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig5, use_container_width=True)

    with col6:
        fig6 = px.bar(agegroup_surv, x='AgeGroup', y='Survived',
                      title='연령대별 생존률',
                      labels={'Survived': '생존률'},
                      color='AgeGroup',
                      color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig6, use_container_width=True)
# -------------------------
# 📍 탑승항 분석
# -------------------------
elif menu == "📍 탑승항 분석":
    st.header("📍 탑승항(Embarked)에 따른 생존자 분포")
    set_korean_font()
    fig = plt.figure(figsize=(8, 5))
    sns.countplot(data=train, x='Embarked', hue='Survived')
    plt.xlabel('탑승항 (C = Cherbourg, Q = Queenstown, S = Southampton)')
    plt.ylabel('탑승자 수')
    plt.title('탑승항에 따른 생존/사망 분포')
    st.pyplot(fig)

# -------------------------
# 📈 상관관계 히트맵
# -------------------------
elif menu == "📈 상관관계 히트맵":
    st.header("📈 수치형 변수 간 상관관계 히트맵")
    set_korean_font()
    corr_df = train.copy()
    corr_df['Sex'] = corr_df['Sex'].map({'male': 0, 'female': 1})
    corr_df['Embarked'] = corr_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    corr_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Survived']
    corr_matrix = corr_df[corr_cols].corr()
    fig = plt.figure(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="Blues")
    plt.title("수치형 변수 간 상관관계")
    st.pyplot(fig)

# -------------------------
# 🤖 예측 모델
# -------------------------
# 🤖 생존자 예측 (인터랙티브)
# -------------------------
elif menu == "🤖 예측 모델":
    st.header("🤖 생존자 예측")

    # 모델 학습 및 예측
    model = train_model(X_train, y_train)
    preds = predict_test(model, X_test)

    # 결과 저장
    result_df = test[['PassengerId']].copy()
    result_df['Survived'] = preds
    result_df = pd.merge(result_df, test, on='PassengerId')

    # 예측 생존 비율
    survival_rate = result_df['Survived'].mean() * 100
    st.markdown(f"### ✅ 전체 예측 생존 비율: **{survival_rate:.2f}%**")

    # 🔎 필터 선택
    st.subheader("🔍 조건별로 결과 확인")
    pclass_filter = st.multiselect("객실 등급 (Pclass)", [1, 2, 3], default=[1, 2, 3])
    sex_filter = st.multiselect("성별", ['male', 'female'], default=['male', 'female'])

        # 나이 범위 선택 (정수만 허용: 1, 2, 3, ..., N)
    min_age = int(result_df['Age'].min())
    max_age = int(result_df['Age'].max())

    age_range = st.slider(
        "나이 범위 선택",
        min_value=min_age,
        max_value=max_age,
        value=(min_age, max_age),
        step=1
    )

    filtered_df = result_df[
        result_df['Pclass'].isin(pclass_filter) &
        result_df['Sex'].isin(sex_filter) &
        result_df['Age'].between(age_range[0], age_range[1])
    ]

    st.markdown(f"**필터링된 승객 수: {len(filtered_df)}명**")
    st.dataframe(filtered_df[['PassengerId', 'Pclass', 'Sex', 'Age', 'Survived']])

    # 📈 생존율 재계산
    if len(filtered_df) > 0:
        sub_rate = filtered_df['Survived'].mean() * 100
        st.markdown(f"📈 **선택한 조건의 생존 비율: {sub_rate:.2f}%**")

# -------------------------
# 📁 결과 다운로드
# -------------------------
elif menu == "📁 결과 다운로드":
    st.header("📁 예측 결과 다운로드")
    model = train_model(X_train, y_train)
    preds = predict_test(model, X_test)
    result_df = test[['PassengerId']].copy()
    result_df['Survived'] = preds
    st.download_button(
        label="📥 CSV 파일 다운로드",
        data=result_df.to_csv(index=False).encode('utf-8'),
        file_name='titanic_prediction.csv',
        mime='text/csv'
    )
