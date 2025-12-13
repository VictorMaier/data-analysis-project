import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score

class DataPredictor:
    def __init__(self, df):
        self.df = df

    def predict(self, target_col, feature_cols, model_type="linear"):
        """
        Множественная регрессия.
        target_col: строка (что предсказываем)
        feature_cols: список строк (на основе чего)
        model_type: 'linear', 'tree', 'forest'
        """
        # Подготовка данных
        # Собираем все нужные колонки и чистим от NaN
        cols_needed = [target_col] + feature_cols
        data = self.df[cols_needed].dropna()

        if data.empty:
            print("Ошибка: После удаления пустых строк данных не осталось.")
            return

        X = data[feature_cols] # Входные параметры (может быть много)
        y = data[target_col]   # Цель

        # Разделение
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Выбор модели
        if model_type == "linear":
            model = LinearRegression()
            name = "Линейная регрессия"
        elif model_type == "tree":
            model = DecisionTreeRegressor(random_state=42)
            name = "Дерево решений"
        elif model_type == "forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            name = "Случайный лес"
        else:
            print("Неизвестная модель.")
            return

        # Обучение
        model.fit(X_train, y_train)

        # Прогноз
        y_pred = model.predict(X_test)

        # Оценка
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"\n[Результаты обучения: {name}]")
        print(f"Использованы параметры: {', '.join(feature_cols)}")
        print(f"Средняя ошибка (MAE): {mae:.2f}")
        print(f"Коэффициент R2: {r2:.2f}")

        if model_type == "linear":
            # Для линейной регрессии можно вывести коэффициенты
            coeffs = zip(feature_cols, model.coef_)
            print("Вклад параметров (коэффициенты):")
            for feature, coef in coeffs:
                print(f"  {feature}: {coef:.4f}")

        # График "Реальность vs Прогноз"
        # Так как X многомерный, мы не можем построить простой 2D график.
        # Строим график, где по X - реальные значения, по Y - предсказанные.
        # Идеальный прогноз - это диагональная линия.
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.7, color='blue')
        
        # Линия идеального предсказания
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Идеал')
        
        plt.xlabel(f'Реальные значения ({target_col})')
        plt.ylabel(f'Предсказанные значения ({target_col})')
        plt.title(f'Качество модели: {name}')
        plt.legend()
        print("График сравнения открыт.")
        plt.show()
