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

    def convert_to_numeric(self):
        """Пытается превратить строки в числа (исправление форматирования)."""
        # Проходим по колонкам, которые определились как object (текст)
        for col in self.df.select_dtypes(include=['object']).columns:
            try:
                # errors='coerce' превратит то, что не смог, в NaN, но не упадет
                converted = pd.to_numeric(self.df[col], errors='coerce')
                # Если колонка стала числовой (все значения распознались), сохраняем
                if converted.notna().sum() > 0:
                    self.df[col] = converted
                    print(f"[Очистка] Колонка '{col}' преобразована в числа.")
            except:
                pass
        return self.df

    def remove_outliers(self):
        """Удаляет выбросы используя межквартильный размах (IQR)."""
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        if len(numeric_cols) == 0:
            return self.df
        
        before = len(self.df)
        
        # Для каждой числовой колонки считаем границы
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Оставляем только те строки, которые внутри границ
            self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
            
        after = len(self.df)
        print(f"[Очистка] Удалено строк с выбросами: {before - after}")
        return self.df
