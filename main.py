import pandas as pd
import os
import sys
from src.cleaner import DataCleaner
from src.statistics import DataStats
from src.visualizer import DataVisualizer
from src.machine_learning import DataPredictor

# Настройки отображения
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Глобальная переменная для хранения загруженного датасета
current_df = None
# Глобальный клинер, чтобы помнить историю очистки
current_cleaner = None

def load_data():
    global current_df, current_cleaner
    path = input("Введите путь к CSV файлу (например, data.csv): ")
    # Убираем кавычки, если пользователь скопировал путь как "C:\path\to\file"
    path = path.strip('"').strip("'")
    
    if not os.path.exists(path):
        print(f"Ошибка: Файл '{path}' не найден.")
        return

    try:
        current_df = pd.read_csv(path)
        # Создаем клинер один раз при загрузке
        current_cleaner = DataCleaner(current_df)
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
    global current_df, current_cleaner
    if current_df is None:
        print("Сначала загрузите данные!")
        return

    # Обновляем df внутри клинера, если он менялся снаружи (на всякий случай)
    current_cleaner.df = current_df

    while True:
        print("\n--- Меню очистки данных ---")
        print("1. Удалить дубликаты")
        print("2. Удалить строки с пропусками (NaN)")
        print("3. Заполнить пропуски средним значением")
        print("4. Исправить форматирование (текст -> числа)")
        print("5. Исправить форматирование (текст -> даты)")
        print("6. Удалить выбросы (Z-score)")
        print("7. Показать сводку по очистке")
        print("8. Показать текущую таблицу")
        print("0. Назад в главное меню")
        
        choice = input("Выберите действие: ")
        
        if choice == "1":
            current_df = current_cleaner.remove_duplicates()
        elif choice == "2":
            current_df = current_cleaner.remove_missing_values()
        elif choice == "3":
            current_df = current_cleaner.fill_missing_values()
        elif choice == "4":
            current_df = current_cleaner.convert_to_numeric()
        elif choice == "5":
            current_df = current_cleaner.convert_to_datetime()
        elif choice == "6":
            try:
                val = input("Введите порог Z-оценки (по умолчанию 3): ")
                # Если нажали Enter, ставим 3
                threshold = float(val) if val.strip() else 3.0
                current_df = current_cleaner.remove_outliers(threshold)
            except ValueError:
                print("Ошибка: нужно ввести число.")
        elif choice == "7":
            current_cleaner.print_summary()
        elif choice == "8":
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
    print("1. Общая статистика")
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
    print("2. Диаграмма плотности")
    print("3. Box Plot (Медиана/IQR)")
    print("4. Box Plot (Среднее/Std)")
    print("5. Violin Plot")
    print("6. Scatter Plot")
    print("0. Назад")
    
    choice = input("Выберите тип графика: ")
    
    if choice in ["1", "2", "3", "4", "5"]:
        col = input("Введите название колонки из списка выше: ")
        if col not in numeric_cols:
            print("Ошибка: такой колонки нет.")
            return

        if choice == "1":
            viz.plot_histogram(col)
        elif choice == "2":
            viz.plot_density(col)
        elif choice == "3":
            viz.plot_boxplot(col)
        elif choice == "4":
            viz.plot_boxplot_mean_std(col)
        elif choice == "5":
            viz.plot_violin(col)

    elif choice == "6":
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

    # Ищем только числовые колонки
    numeric_cols = current_df.select_dtypes(include=['number']).columns.tolist()
    if len(numeric_cols) < 2:
        print("Для прогноза нужно минимум 2 числовые колонки.")
        return

    print("\n--- Машинное обучение (Прогноз) ---")
    print(f"Доступные колонки: {', '.join(numeric_cols)}")
    
    # Спрашиваем, что предсказывать (Y)
    target = input("Что предсказываем (Y)? Введите одну колонку: ")
    if target not in numeric_cols:
        print("Ошибка: такой колонки нет.")
        return

    # Спрашиваем, на основе чего предсказывать (X)
    print("На основе чего предсказываем (X)? Введите колонки через запятую (например: Age, Pclass)")
    features_input = input("Параметры: ")
    features = [f.strip() for f in features_input.split(",")]
    
    valid_features = []
    for f in features:
        if f in numeric_cols and f != target:
            valid_features.append(f)
        elif f == target:
            print("Нельзя использовать целевую колонку для предсказания самой себя.")
    
    if not valid_features:
        print("Не выбрано ни одной корректной колонки для X.")
        return

    print("\nВыберите модель:")
    print("1. Линейная регрессия")
    print("2. Дерево решений")
    print("3. Случайный лес")
    
    m_choice = input("Ваш выбор: ")
    model_type = "linear"
    if m_choice == "2":
        model_type = "tree"
    elif m_choice == "3":
        model_type = "forest"

    # Запускаем
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
