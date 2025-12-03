import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class DataVisualizer:
    def __init__(self, df):
        self.df = df
        # Устанавливаем красивый стиль графиков
        sns.set_theme(style="whitegrid")

    def plot_histogram(self, column):
        """Строит гистограмму (распределение) для выбранной колонки."""
        plt.figure(figsize=(10, 6))
        # kde=True рисует плавную линию тренда
        sns.histplot(self.df[column], kde=True, color='skyblue')
        plt.title(f'Распределение: {column}')
        plt.xlabel(column)
        plt.ylabel('Частота')
        print("График открыт в новом окне. Закройте его, чтобы продолжить.")
        plt.show()

    def plot_boxplot(self, column):
        """Строит 'Ящик с усами' (Box Plot) для поиска выбросов."""
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=self.df[column], color='lightgreen')
        plt.title(f'Box Plot: {column}')
        print("График открыт в новом окне. Закройте его, чтобы продолжить.")
        plt.show()

    def plot_scatter(self, col_x, col_y):
        """Строит график зависимости одной переменной от другой."""
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.df[col_x], y=self.df[col_y])
        plt.title(f'Зависимость {col_y} от {col_x}')
        plt.xlabel(col_x)
        plt.ylabel(col_y)
        print("График открыт в новом окне. Закройте его, чтобы продолжить.")
        plt.show()
