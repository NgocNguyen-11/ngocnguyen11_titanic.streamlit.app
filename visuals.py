# 파일: visuals.py

import plotly.express as px
import pandas as pd

def plot_survival_by_sex(df):
    grouped = df.groupby(['Sex', 'Survived']).size().reset_index(name='Count')
    grouped['Sex'] = grouped['Sex'].map({"male": "남성", "female": "여성"})
    grouped['Survived'] = grouped['Survived'].map({0: "사망", 1: "생존"})
    fig = px.bar(grouped, x='Sex', y='Count', color='Survived', barmode='group',
                 labels={'Sex': '성별', 'Count': '인원 수'},
                 title='성별에 따른 생존자 분포')
    return fig

def plot_survival_by_class(df):
    grouped = df.groupby(['Pclass', 'Survived']).size().reset_index(name='Count')
    grouped['Survived'] = grouped['Survived'].map({0: "사망", 1: "생존"})
    fig = px.bar(grouped, x='Pclass', y='Count', color='Survived', barmode='group',
                 labels={'Pclass': '객실 등급', 'Count': '인원 수'},
                 title='객실 등급에 따른 생존자 분포')
    return fig

def plot_age_distribution(df):
    df['Survived_label'] = df['Survived'].map({0: "사망", 1: "생존"})
    fig = px.histogram(
        df,
        x='Age',
        color='Survived_label',
        nbins=30,
        labels={'Age': '나이', 'Survived_label': '생존 상태'},
        title='나이에 따른 생존자 분포'
    )
    return fig

