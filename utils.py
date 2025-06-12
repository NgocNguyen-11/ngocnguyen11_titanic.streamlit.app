# 파일: utils.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data():
    # CSV 파일 불러오기
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    gender = pd.read_csv("gender_submission.csv")

    test_passenger_ids = test['PassengerId']

    # 학습/테스트 구분 플래그 추가
    train['TrainFlag'] = 1
    test['TrainFlag'] = 0
    full_df = pd.concat([train, test], sort=False)

    # 성별 인코딩
    full_df['Sex'] = LabelEncoder().fit_transform(full_df['Sex'])

    # 나이, 요금 결측치 중앙값으로 대체
    for col in ['Age', 'Fare']:
        imputer = SimpleImputer(strategy='median')
        full_df[col] = imputer.fit_transform(full_df[[col]])

    # 탑승항 결측치 처리 및 인코딩
    full_df['Embarked'] = full_df['Embarked'].fillna('S')
    full_df['Embarked'] = LabelEncoder().fit_transform(full_df['Embarked'])

    # 모델 입력 특성 지정
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

    train_df = full_df[full_df['TrainFlag'] == 1].copy()
    test_df = full_df[full_df['TrainFlag'] == 0].copy()

    X_train = train_df[features]
    y_train = train_df['Survived']
    X_test = test_df[features]

    return train, test, gender, X_train, y_train, X_test, test_passenger_ids

def train_model(X, y):
    # 랜덤 포레스트 모델 학습
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def predict_test(model, X_test):
    # 테스트 데이터 예측
    return model.predict(X_test)
