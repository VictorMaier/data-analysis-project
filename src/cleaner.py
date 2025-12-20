import pandas as pd
import numpy as np
import warnings

class DataCleaner:
    def __init__(self, df):
        self.df = df
        self.history = []

    def remove_duplicates(self):
        """Удаляет полные дубликаты строк."""
        before = len(self.df)
        self.df = self.df.drop_duplicates()
        after = len(self.df)
        count = before - after
        msg = f"[Очистка] Удалено дубликатов: {count}"
        print(msg)
        self.history.append(msg)
        return self.df

    def remove_missing_values(self):
        """Удаляет строки, где есть хотя бы одно пустое значение (NaN)."""
        before = len(self.df)
        self.df = self.df.dropna()
        after = len(self.df)
        count = before - after
        msg = f"[Очистка] Удалено строк с пустыми значениями: {count}"
        print(msg)
        self.history.append(msg)
        return self.df

    def fill_missing_values(self):
        """Заполняет пустые числа средним значением (альтернатива удалению)."""
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if self.df[col].isnull().sum() > 0:
                mean_val = self.df[col].mean()
                self.df[col] = self.df[col].fillna(mean_val)
                msg = f"[Очистка] В колонке '{col}' пропуски заменены на {mean_val:.2f}"
                print(msg)
                self.history.append(msg)
        return self.df

    def convert_to_numeric(self):
        """Пытается превратить строки в числа (исправление форматирования)."""
        for col in self.df.select_dtypes(include=['object']).columns:
            try:
                converted = pd.to_numeric(self.df[col], errors='coerce')
                if converted.notna().sum() > 0:
                    self.df[col] = converted
                    msg = f"[Очистка] Колонка '{col}' преобразована в числа."
                    print(msg)
                    self.history.append(msg)
            except:
                pass
        return self.df

    def convert_to_datetime(self):
        """Пытается превратить строки в даты."""
        for col in self.df.select_dtypes(include=['object']).columns:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    converted = pd.to_datetime(self.df[col], errors='coerce')
                
                if converted.notna().mean() > 0.5:
                    self.df[col] = converted
                    msg = f"[Очистка] Колонка '{col}' преобразована в даты."
                    print(msg)
                    self.history.append(msg)
            except:
                pass
        return self.df

    def remove_outliers(self, threshold=3.0):
        """Удаляет выбросы используя Z-score (порог вводится пользователем)."""
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        if len(numeric_cols) == 0:
            return self.df
        
        before = len(self.df)
        
        for col in numeric_cols:
            mean = self.df[col].mean()
            std = self.df[col].std()
            if std == 0:
                continue
            z_scores = ((self.df[col] - mean) / std).abs()
            self.df = self.df[z_scores < threshold]
            
        after = len(self.df)
        count = before - after
        msg = f"[Очистка] Удалено строк с выбросами (Z-score > {threshold}): {count}"
        print(msg)
        self.history.append(msg)
        return self.df

    def print_summary(self):
        """Выводит сводку удаленных данных."""
        print("\n--- Сводка действий по очистке ---")
        if not self.history:
            print("История пуста.")
        else:
            for item in self.history:
                print(f"- {item}")
