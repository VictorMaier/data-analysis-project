import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
        plt.title(f'Гистограмма: {column}')
        plt.xlabel(column)
        plt.ylabel('Частота')
        print("График открыт.")
        plt.show()

    def plot_density(self, column):
        """Строит диаграмму плотности с линиями среднего, медианы и моды."""
        data = self.df[column].dropna()
        mean_val = data.mean()
        median_val = data.median()
        # Мода может вернуть несколько значений, берем первое
        mode_val = data.mode()[0] if not data.mode().empty else mean_val

        plt.figure(figsize=(10, 6))
        # Рисуем саму плотность
        sns.kdeplot(data, fill=True, color='purple', alpha=0.3)
        
        # Рисуем вертикальные линии
        plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Среднее: {mean_val:.2f}')
        plt.axvline(median_val, color='green', linestyle='-', linewidth=2, label=f'Медиана: {median_val:.2f}')
        plt.axvline(mode_val, color='blue', linestyle=':', linewidth=2, label=f'Мода: {mode_val:.2f}')

        plt.title(f'Диаграмма плотности: {column}')
        plt.xlabel(column)
        plt.ylabel('Плотность')
        plt.legend() # Показываем легенду линий
        print("График открыт.")
        plt.show()

    def plot_boxplot(self, column):
        """Строит 'Ящик с усами' (Box Plot) на основе Медианы и квартилей."""
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=self.df[column], color='lightgreen')
        plt.title(f'Box Plot (IQR/Median): {column}')
        print("График открыт.")
        plt.show()

    def plot_boxplot_mean_std(self, column):
        """Строит Box Plot на основе Среднего и Стандартного отклонения."""
        data = self.df[column].dropna()
        mean = data.mean()
        std = data.std()
        min_val = data.min()
        max_val = data.max()

        plt.figure(figsize=(10, 6))
        ax = plt.gca()

        # Рисуем ящик (Среднее +/- Стд. отклонение) вручную
        # (x, y), width, height
        rect = patches.Rectangle((mean - std, -0.2), 2 * std, 0.4, 
                                 linewidth=1, edgecolor='blue', facecolor='lightblue', alpha=0.5)
        ax.add_patch(rect)
        
        # Линия среднего
        plt.plot([mean, mean], [-0.2, 0.2], color='red', linewidth=2, label='Среднее')
        # Усы (до мин и макс)
        plt.plot([min_val, mean - std], [0, 0], color='black', linestyle='--')
        plt.plot([mean + std, max_val], [0, 0], color='black', linestyle='--')
        # Засечки
        plt.plot([min_val, min_val], [-0.1, 0.1], color='black')
        plt.plot([max_val, max_val], [-0.1, 0.1], color='black')

        plt.yticks([])
        plt.ylim(-0.5, 0.5)
        plt.xlabel(column)
        plt.title(f'Box Plot (Mean +/- Std): {column}')
        plt.legend()
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
        """Строит график зависимости одной переменной от другой."""
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.df[col_x], y=self.df[col_y])
        plt.title(f'Зависимость {col_y} от {col_x}')
        plt.xlabel(col_x)
        plt.ylabel(col_y)
        print("График открыт.")
        plt.show()
