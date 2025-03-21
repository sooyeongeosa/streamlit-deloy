import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 데이터 불러오기
df = pd.read_csv("data/diabetes.csv")

# Step1: 결측치 확인 및 이상치 제거
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

df = remove_outliers(df, 'BMI')

# Step2: 특성과 타겟 분리
X = df.drop(columns=['Outcome'])
y = df['Outcome']

# Step3: 데이터 정규화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step4: 데이터셋 분할
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step5: 모델 학습
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step6: 예측 및 성능 평가
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Streamlit UI 디자인
st.set_page_config(page_title="Diabetes Prediction Dashboard", layout="wide")

# Tab-based navigation instead of radio buttons
tabs = st.sidebar.tabs(["Home", "EDA", "Model Performance"])
# 사이드바 메뉴
st.sidebar.title("Navigation")
# menu = st.sidebar.radio("Go to",["Home","EDA","Model Performance"])


# Home screen
def home():
    st.title("Diabetes Prediction Dashboard")
    st.markdown("""
    - **Pregnancies**: 임신 횟수
    - **Glucose**: 포도당 수치
    - **BloodPressure**: 혈압
    - **SkinThickness**: 피부 두께
    - **Insulin**: 인슐린 수치
    - **BMI**: 체질량지수
    - **DiabetesPedigreeFunction**: 가족력 지수
    - **Age**: 나이
    - **Outcome**: 당뇨 여부 (0: 정상, 1: 당뇨병)
    """)

# Exploratory Data Analysis (EDA)
def eda():
    st.title("Exploratory Data Analysis")
    chart_tabs = st.tabs(["Histogram", "Boxplot", "Heatmap"])

    with chart_tabs[0]:
        st.subheader("Feature Distribution")
        # Plotly Histograms for feature distribution
        columns = ["Glucose", "BloodPressure", "BMI", "Age"]
        for col in columns:
            fig = px.histogram(df, x=col, nbins=20, title=f"{col} Distribution", marginal="box")
            fig.update_layout(width=600, height=400)  # Adjust size here
            st.plotly_chart(fig)

    with chart_tabs[1]:
        st.subheader("BMI Boxplot by Outcome")
        # Plotly Boxplot for BMI by Outcome
        fig = px.box(df, x="Outcome", y="BMI", title="BMI Boxplot by Outcome")
        fig.update_layout(width=600, height=400)  # Adjust size here
        st.plotly_chart(fig)

    with chart_tabs[2]:
        st.subheader("Feature Correlation Heatmap")
        # Plotly Heatmap for feature correlation
        corr = df.corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale='Viridis',
            zmin=-1, zmax=1
        ))
        fig.update_layout(title="Correlation Heatmap", xaxis_title="Features", yaxis_title="Features",
                          width=600, height=400)  # Adjust size here
        st.plotly_chart(fig)

# Model Performance
def model_performance():
    st.title("Model Performance")
    st.write(f'### Model Accuracy: {accuracy:.2f}')
    
    # Classification report as a string
    st.subheader("Classification Report")
    st.text(classification_rep)
    
    # Confusion Matrix
    st.subheader("Confusion Matrix")
    fig = go.Figure(data=go.Heatmap(
        z=conf_matrix,
        x=["Normal", "Diabetes"],
        y=["Normal", "Diabetes"],
        colorscale='Blues',
        zmin=0, zmax=np.max(conf_matrix),
        showscale=True
    ))
    fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual", 
                      width=600, height=400)  # Adjust size here
    st.plotly_chart(fig)

# 탭 메뉴에 따른 화면 전환
if tabs[0] == "Home":
    home()

elif tabs[1] == "EDA":
    eda()

elif tabs[2] == "Model Performance":
    model_performance()
