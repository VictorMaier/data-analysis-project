import pandas as pd
import os
import sys
from src.cleaner import DataCleaner
from src.statistics import DataStats
from src.visualizer import DataVisualizer

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

def clean_data():
    global current_df
    if current_df is None:
        print("Сначала загрузите данные!")
        return

    print("\n--- Меню очистки данных ---")
    print("1. Удалить дубликаты")
    print("2. Удалить строки с пропусками (NaN)")
    print("3. Заполнить пропуски средним значением")
    print("0. Назад")
    
    choice = input("Выберите действие: ")
    
    # Создаем объект очистителя, передавая ему текущий датасет
    cleaner = DataCleaner(current_df)
    
    if choice == "1":
        current_df = cleaner.remove_duplicates()
    elif choice == "2":
        current_df = cleaner.remove_missing_values()
    elif choice == "3":
        current_df = cleaner.fill_missing_values()
    elif choice == "0":
        return
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
    print("3. Scatter Plot (Точечный график)")
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
    if current_df is None:
        print("Сначала загрузите данные!")
        return
    print("--- Машинное обучение ---")
    print("ML пока не реализован.")

def main_menu():
    while True:
        print("\n=== УНИВЕРСАЛЬНАЯ СИСТЕМА СТАТИСТИКИ ===")
        print("1. Загрузить данные")
        print("2. Очистить данные")
        print("3. Показать статистику")
        print("4. Построить графики")
        print("5. Прогнозирование")
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
        elif choice == "0":
            print("Выход...")
            break
        else:
            print("Неверная команда.")

if __name__ == "__main__":
    main_menu()
