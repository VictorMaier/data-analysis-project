import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

class DataPredictor:
    def __init__(self, df):
        self.df = df

    def predict_one_to_one(self, target_col, feature_col):
        # Простая линейная регрессия: предсказываем target (Y) на основе feature (X).
        # Подготовка данных
        # Удаляем пустые строки, чтобы модель не упала
        data = self.df[[target_col, feature_col]].dropna()
        
        X = data[[feature_col]] # Признак (вход)
        y = data[target_col]    # Цель (выход)

        # Разделение на обучение и тест (80% учим, 20% проверяем)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Обучение модели
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Проверка (предсказание)
        y_pred = model.predict(X_test)

        # Оценка точности
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"\n[Результаты обучения]")
        print(f"Средняя ошибка (MAE): {mae:.2f}")
        print(f"Коэффициент R2 (точность от 0 до 1): {r2:.2f}")
        print(f"Формула: {target_col} = {model.coef_[0]:.2f} * {feature_col} + {model.intercept_:.2f}")

        # Визуализация: Точки + Линия регрессии
        plt.figure(figsize=(10, 6))
        plt.scatter(X_test, y_test, color='blue', label='Реальные данные')
        plt.plot(X_test, y_pred, color='red', linewidth=2, label='Предсказание модели')
        plt.xlabel(feature_col)
        plt.ylabel(target_col)
        plt.legend()
        plt.title(f'Прогноз: {target_col} vs {feature_col}')
        print("График с результатами открыт в новом окне.")
        plt.show()
