import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class DataVisualizer:
    def __init__(self, df):
        self.df = df
        sns.set_theme(style="whitegrid")

    def plot_histogram(self, column):
        plt.figure(figsize=(10, 6))
        sns.histplot(self.df[column], kde=True, color='skyblue')
        plt.title(f'Гистограмма: {column}')
        plt.xlabel(column)
        plt.ylabel('Частота')
        print("График открыт.")
        plt.show()

    def plot_boxplot(self, column):
        """Классический BoxPlot (ящик с усами) на основе IQR."""
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=self.df[column], color='lightgreen')
        plt.title(f'Box Plot (IQR): {column}')
        plt.xlabel('Значение (точки за усами - выбросы)')
        print("График открыт.")
        plt.show()

    def plot_violin(self, column):
        """Скрипичная диаграмма (второй тип диаграммы размаха)."""
        plt.figure(figsize=(10, 6))
        sns.violinplot(x=self.df[column], color='orange')
        plt.title(f'Violin Plot: {column}')
        print("График открыт.")
        plt.show()

    def plot_scatter(self, col_x, col_y):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.df[col_x], y=self.df[col_y])
        plt.title(f'Зависимость {col_y} от {col_x}')
        plt.xlabel(col_x)
        plt.ylabel(col_y)
        print("График открыт.")
        plt.show()
