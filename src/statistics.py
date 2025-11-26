import pandas as pd

class DataStats:
    def __init__(self, df):
        self.df = df
        # Работаем только с числовыми колонками для статистики
        self.numeric_df = self.df.select_dtypes(include=['number'])

    def get_basic_stats(self):
        """Возвращает расширенную статистику (Мода, Медиана, Дисперсия и т.д.)"""
        if self.numeric_df.empty:
            return "Нет числовых данных для анализа."

        stats = {}
        for col in self.numeric_df.columns:
            series = self.numeric_df[col]
            stats[col] = {
                'Среднее (Mean)': series.mean(),
                'Медиана (Median)': series.median(),
                'Мода (Mode)': series.mode()[0] if not series.mode().empty else "N/A",
                'Минимум': series.min(),
                'Максимум': series.max(),
                'Стд. отклонение (Std)': series.std(),
                'Дисперсия (Variance)': series.var()
            }
        return pd.DataFrame(stats)

    def get_correlation(self):
        """Возвращает матрицу корреляции Пирсона"""
        if self.numeric_df.empty:
            return "Нет данных для корреляции."
        # corr() считает корреляцию Пирсона по умолчанию
        return self.numeric_df.corr()
