import streamlit as st
from streamlit_reveal_slides import slides

def main():
    st.title("Презентация проекта")
    
    content = """
    # Прогнозирование отказов оборудования
    
    ## Введение
    - Задача: Предсказание отказов промышленного оборудования
    - Цель: Снижение простоев и затрат на ремонт
    
    ## Технологии
    - Python + Streamlit
    - Scikit-learn
    - Random Forest
    
    ## Результаты
    - Точность модели: 95%+
    - Реализовано веб-приложение для прогнозов
    
    ## Демо
    [Ссылка на видео-демонстрацию](https://example.com/demo)
    """
    
    slides(content, height=500, theme="night")

if __name__ == "__main__":
    main()