import pandas as pd
import os
import sys
from src.cleaner import DataCleaner

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
    if current_df is None:
        print("Сначала загрузите данные!")
        return
    print("--- Статистика ---")
    print(current_df.describe()) 

def show_plots():
    if current_df is None:
        print("Сначала загрузите данные!")
        return
    print("--- Графики ---")
    print("Графики пока не реализованы.")

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
