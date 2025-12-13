import pandas as pd
import os
import sys
from src.cleaner import DataCleaner
from src.statistics import DataStats
from src.visualizer import DataVisualizer
from src.machine_learning import DataPredictor

# Настройки отображения: показывать все колонки и делать широкую строку
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Глобальная переменная для хранения загруженного датасета
current_df = None

def load_data():
    global current_df
    path = input("Введите путь к CSV файлу (например, data.csv): ")
    # Убираем кавычки, если пользователь скопировал путь как "C:\path\to\file"
    path = path.strip('"').strip("'")
    
    if not os.path.exists(path):
        print(f"Ошибка: Файл '{path}' не найден.")
        return

    try:
        current_df = pd.read_csv(path)
        print(f"\nУспешно загружено! Размер таблицы: {current_df.shape}")
        print("Первые 5 строк:")
        print(current_df.head())
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")

def show_current_data():
    if current_df is None:
        print("Сначала загрузите данные!")
        return
    print("\n--- Текущие данные ---")
    print(current_df)
    print(f"\nРазмер: {current_df.shape}")

def clean_data():
    global current_df
    if current_df is None:
        print("Сначала загрузите данные!")
        return

    # Интерактивный режим: не выходим, пока юзер сам не скажет
    while True:
        print("\n--- Меню очистки данных ---")
        print("1. Удалить дубликаты")
        print("2. Удалить строки с пропусками (NaN)")
        print("3. Заполнить пропуски средним значением")
        print("4. Исправить форматирование (текст -> числа)")
        print("5. Удалить выбросы (метод IQR)")
        print("6. Показать текущую таблицу")
        print("0. Назад в главное меню")
        
        choice = input("Выберите действие: ")
        
        cleaner = DataCleaner(current_df)
        
        if choice == "1":
            current_df = cleaner.remove_duplicates()
        elif choice == "2":
            current_df = cleaner.remove_missing_values()
        elif choice == "3":
            current_df = cleaner.fill_missing_values()
        elif choice == "4":
            current_df = cleaner.convert_to_numeric()
        elif choice == "5":
            current_df = cleaner.remove_outliers()
        elif choice == "6":
            print(current_df)
        elif choice == "0":
            break
        else:
            print("Неверный выбор.")

def show_statistics():
    global current_df
    if current_df is None:
        print("Сначала загрузите данные!")
        return
    
    # Создаем объект статистики
    stats_module = DataStats(current_df)
    
    print("\n--- Статистика ---")
    print("1. Общая статистика (Среднее, Медиана, Мода, Дисперсия)")
    print("2. Матрица корреляции")
    print("0. Назад")
    
    choice = input("Выберите действие: ")
    
    if choice == "1":
        print("\n[Основные показатели]")
        # .T транспонирует таблицу (строки становятся столбцами) для удобства чтения
        print(stats_module.get_basic_stats().T)
    elif choice == "2":
        print("\n[Матрица корреляции]")
        print(stats_module.get_correlation())
    elif choice == "0":
        return
    else:
        print("Неверный выбор.")

def show_plots():
    global current_df
    if current_df is None:
        print("Сначала загрузите данные!")
        return

    # Pandas находит все колонки с числами, как бы они ни назывались
    numeric_cols = current_df.select_dtypes(include=['number']).columns.tolist()
    
    if not numeric_cols:
        print("В этом файле нет числовых колонок для построения графиков.")
        return

    viz = DataVisualizer(current_df)
    
    print("\n--- Визуализация ---")
    print(f"Доступные колонки: {', '.join(numeric_cols)}")
    print("1. Гистограмма")
    print("2. Box Plot (Ящик с усами)")
    print("3. Violin Plot (Скрипичная диаграмма)")
    print("4. Scatter Plot (Точечный график)")
    print("0. Назад")
    
    choice = input("Выберите тип графика: ")
    
    if choice == "1":
        col = input("Введите название колонки из списка выше: ")
        if col in numeric_cols:
            viz.plot_histogram(col)
        else:
            print("Ошибка: такой колонки нет.")
            
    elif choice == "2":
        col = input("Введите название колонки из списка выше: ")
        if col in numeric_cols:
            viz.plot_boxplot(col)
        else:
            print("Ошибка: такой колонки нет.")

    elif choice == "3":
        col = input("Введите название колонки из списка выше: ")
        if col in numeric_cols:
            viz.plot_violin(col)
        else:
            print("Ошибка: такой колонки нет.")
    
    elif choice == "4":
        col_x = input("Введите колонку для оси X: ")
        col_y = input("Введите колонку для оси Y: ")
        if col_x in numeric_cols and col_y in numeric_cols:
            viz.plot_scatter(col_x, col_y)
        else:
            print("Ошибка: неверные названия колонок.")
            
    elif choice == "0":
        return
    else:
        print("Неверный выбор.")

def run_ml():
    global current_df
    if current_df is None:
        print("Сначала загрузите данные!")
        return

    numeric_cols = current_df.select_dtypes(include=['number']).columns.tolist()
    if len(numeric_cols) < 2:
        print("Мало данных для прогноза.")
        return

    print("\n--- Машинное обучение ---")
    print(f"Доступные колонки: {', '.join(numeric_cols)}")
    
    # Выбор цели
    target = input("Что предсказываем (Y)? Введите одну колонку: ")
    if target not in numeric_cols:
        print("Ошибка: такой колонки нет.")
        return

    # Выбор параметров (можно несколько)
    print("На основе чего предсказываем (X)? Введите колонки через запятую (например: Age, Pclass)")
    features_input = input("Параметры: ")
    # Превращаем строку "Age, Pclass" в список ['Age', 'Pclass'] и чистим пробелы
    features = [f.strip() for f in features_input.split(",")]
    
    # Проверка, что все введенные колонки существуют
    valid_features = []
    for f in features:
        if f in numeric_cols and f != target:
            valid_features.append(f)
        elif f == target:
            print("Нельзя использовать целевую колонку для предсказания самой себя.")
    
    if not valid_features:
        print("Не выбрано ни одной корректной колонки для X.")
        return

    # Выбор модели
    print("\nВыберите модель:")
    print("1. Линейная регрессия (для простых связей)")
    print("2. Дерево решений (Decision Tree)")
    print("3. Случайный лес (Random Forest - мощная модель)")
    
    m_choice = input("Ваш выбор: ")
    model_type = "linear"
    if m_choice == "2":
        model_type = "tree"
    elif m_choice == "3":
        model_type = "forest"

    # Запуск
    predictor = DataPredictor(current_df)
    predictor.predict(target, valid_features, model_type)

def main_menu():
    while True:
        print("\n=== УНИВЕРСАЛЬНАЯ СИСТЕМА СТАТИСТИКИ ===")
        print("1. Загрузить данные")
        print("2. Очистить данные")
        print("3. Показать статистику")
        print("4. Построить графики")
        print("5. Прогнозирование")
        print("6. Показать таблицу")
        print("0. Выход")
        
        choice = input("Выберите действие: ")
        
        if choice == "1":
            load_data()
        elif choice == "2":
            clean_data()
        elif choice == "3":
            show_statistics()
        elif choice == "4":
            show_plots()
        elif choice == "5":
            run_ml()
        elif choice == "6":
            show_current_data()
        elif choice == "0":
            print("Выход...")
            break
        else:
            print("Неверная команда.")

if __name__ == "__main__":
    main_menu()
