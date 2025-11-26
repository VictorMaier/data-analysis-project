import pandas as pd

class DataCleaner:
    def __init__(self, df):
        self.df = df

    def remove_duplicates(self):
        """Удаляет полные дубликаты строк."""
        before = len(self.df)
        self.df = self.df.drop_duplicates()
        after = len(self.df)
        count = before - after
        print(f"[Очистка] Удалено дубликатов: {count}")
        return self.df

    def remove_missing_values(self):
        """Удаляет строки, где есть хотя бы одно пустое значение (NaN)."""
        before = len(self.df)
        self.df = self.df.dropna()
        after = len(self.df)
        count = before - after
        print(f"[Очистка] Удалено строк с пустыми значениями: {count}")
        return self.df

    def fill_missing_values(self):
        """Заполняет пустые числа средним значением (альтернатива удалению)."""
        # Берем только числовые колонки
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if self.df[col].isnull().sum() > 0:
                mean_val = self.df[col].mean()
                self.df[col] = self.df[col].fillna(mean_val)
                print(f"[Очистка] В колонке '{col}' пропуски заменены на {mean_val:.2f}")
        return self.df
