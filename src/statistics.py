import pandas as pd

class DataStats:
    def __init__(self, df):
        self.df = df
        # Работаем только с числовыми колонками для статистики
        self.numeric_df = self.df.select_dtypes(include=['number'])

    def get_basic_stats(self):
        """Возвращает расширенную статистику (включая асимметрию и эксцесс)."""
        if self.numeric_df.empty:
            return "Нет числовых данных для анализа."

        stats = {}
        for col in self.numeric_df.columns:
            series = self.numeric_df[col]
            
            # Вычисляем IQR (Интерквартильный размах)
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1

            # Среднее абсолютное отклонение (MAD)
            # В новых версиях pandas удалили .mad(), считаем вручную:
            mad = (series - series.mean()).abs().mean()

            stats[col] = {
                'Среднее': series.mean(),
                'Медиана': series.median(),
                'Мода': series.mode()[0] if not series.mode().empty else "N/A",
                'Минимум': series.min(),
                'Максимум': series.max(),
                'Стд. отклонение': series.std(),
                'Дисперсия': series.var(),
                'MAD (Ср. абс. откл)': mad,
                'IQR (Интерквартильный)': IQR,
                'Skew (Асимметрия)': series.skew(),
                'Kurtosis (Эксцесс)': series.kurt()
            }
        return pd.DataFrame(stats)

    def get_correlation(self):
        """Возвращает матрицу корреляции Пирсона"""
        if self.numeric_df.empty:
            return "Нет данных для корреляции."
        # corr() считает корреляцию Пирсона по умолчанию
        return self.numeric_df.corr()
