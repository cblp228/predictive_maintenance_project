import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def main():
    st.title("Анализ данных и модель")
    
    # Загрузка данных
    data = load_data()
    
    if data is not None:
        # Предобработка
        processed_data = preprocess_data(data)
        
        # Разделение данных
        X_train, X_test, y_train, y_test = split_data(processed_data)
        
        # Обучение модели
        model, scaler = train_model(X_train, y_train)
        
        # Оценка
        evaluate_model(model, X_test, y_test)
        
        # Интерфейс предсказания
        prediction_interface(model, scaler)

def load_data():
    uploaded_file = st.file_uploader("Загрузите датасет (CSV)", type="csv")
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    else:
        try:
            return pd.read_csv("data/predictive_maintenance.csv")
        except:
            st.warning("Используйте пример данных или загрузите файл")
            return None

def preprocess_data(data):
    # Удаление столбцов
    data = data.drop(columns=['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
    
    # Кодирование категориальных данных
    data['Type'] = LabelEncoder().fit_transform(data['Type'])
    
    # Масштабирование
    scaler = StandardScaler()
    num_cols = ['Air temperature [K]', 'Process temperature [K]', 
                'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    data[num_cols] = scaler.fit_transform(data[num_cols])
    
    return data

def split_data(data):
    X = data.drop('Machine failure', axis=1)
    y = data['Machine failure']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, StandardScaler().fit(X_train)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    st.subheader("Результаты оценки")
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', ax=ax)
    st.pyplot(fig)
    
    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))

def prediction_interface(model, scaler):
    st.header("Предсказание отказа")
    with st.form("prediction_form"):
        air_temp = st.number_input("Температура воздуха [K]", min_value=295.0, max_value=305.0)
        process_temp = st.number_input("Температура процесса [K]", min_value=305.0, max_value=315.0)
        rotational_speed = st.number_input("Скорость вращения [rpm]", min_value=1100, max_value=3000)
        torque = st.number_input("Крутящий момент [Nm]", min_value=3.0, max_value=80.0)
        tool_wear = st.number_input("Износ инструмента [min]", min_value=0, max_value=250)
        product_type = st.selectbox("Тип продукта", ["L", "M", "H"])
        
        if st.form_submit_button("Предсказать"):
            # Преобразование введенных данных
            input_data = pd.DataFrame({
                'Type': [product_type],
                'Air temperature [K]': [air_temp],
                'Process temperature [K]': [process_temp],
                'Rotational speed [rpm]': [rotational_speed],
                'Torque [Nm]': [torque],
                'Tool wear [min]': [tool_wear]
            })
            input_data['Type'] = LabelEncoder().fit_transform(input_data['Type'])
            
            # Предсказание
            prediction = model.predict(input_data)
            probability = model.predict_proba(input_data)[0][1]
            
            st.success(f"Результат: {'Отказ' if prediction[0] == 1 else 'Без отказов'}")
            st.info(f"Вероятность отказа: {probability:.2%}")

if __name__ == "__main__":
    main()