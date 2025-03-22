import argparse
import shutil
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import sys
import logging
from logger import SingletonLogger
from clearml import Task, Logger

# Инициализация задачи ClearML
task = Task.init(
    project_name="Spaceship Titanic",
    task_name="CatBoost Model Training",
    tags=["classification", "catboost"]
)

# Инициализация логгера
logger = SingletonLogger().get_logger()
optuna_logger = optuna.logging.get_logger("optuna")
optuna_logger.handlers = logger.handlers


class LoggerWriter:
    """
    Класс для перенаправления вывода в логгер
    """

    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, message):
        if message.strip():  # Игнорируем пустые строки
            self.logger.log(self.level, message.strip())

    def flush(self):
        pass


# Перенаправляем stdout и stderr в логгер
sys.stdout = LoggerWriter(logger, logging.INFO)
sys.stderr = LoggerWriter(logger, logging.ERROR)


class MyClassifierModel:
    """
    Класс для обучения и предсказания с использованием модели CatBoostClassifier.
    """

    def __init__(self):
        """
        Инициализация класса с указанием путей сохранения модели и результатов предсказания
        """
        self.model_path = './data/model/catboost_model.cbm'
        self.results_path = './data/results.csv'

    def _fill_missing_values(self, data):
        """
        Заполняет пропущенные значения в данных.
        """
        numeric_data = data.select_dtypes(include=["int", "float"]).columns
        categorical_data = data.select_dtypes(exclude=["int", "float"]).columns

        # Преобразование числовых значений
        for col in numeric_data:
            data[col] = data[col].fillna(data[col].median())

        # Преобразование категориальных значений
        for col in categorical_data:
            data[col] = data[col].fillna(data[col].mode()[0])

        # Преобразование булевых колонок
        data["CryoSleep"] = data["CryoSleep"].astype(bool).fillna(False)
        data["VIP"] = data["VIP"].astype(bool).fillna(False)

        return data

    def _prepare_data(self, data):
        """
        Подготавливает данные для обучения или предсказания.
        """
        data = self._fill_missing_values(data)

        # Обработка колонки Cabin
        cabin_data = data["Cabin"].str.split("/", expand=True)
        cabin_data.columns = ["Deck", "Num", "Side"]
        cabin_data["Num"] = cabin_data["Num"].fillna(-1).astype(int)
        cabin_data["Deck"] = cabin_data["Deck"].fillna("Unknown")
        cabin_data["Side"] = cabin_data["Side"].fillna("Unknown")

        # Сумма всех трат
        spends_columns = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        data["Spends"] = data[spends_columns].sum(axis=1)
        data["NoSpends"] = (data["Spends"] == 0)

        # Числовые колонки
        numeric_columns = [
            'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', "Spends",
            "NoSpends", "Age", "CryoSleep", "VIP"
        ]
        numerics = data[numeric_columns]

        # Категориальные колонки
        columns_for_dummies = ["HomePlanet", "Destination"]
        dummies = pd.get_dummies(cabin_data[["Deck", "Side"]].join(data[columns_for_dummies]))

        # Объединение числовых и категориальных данных
        result = pd.concat([numerics, dummies], axis=1)
        return result

    def train(self, dataset_path):
        """
        Обучает и сохраняет модель CatBoostClassifier на предоставленных данных
        """
        logger.info("Начало обучения модели")
        train_data = pd.read_csv(dataset_path)
        y = train_data["Transported"]
        X = self._prepare_data(train_data)

        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

        def objective_catboost(trial):
            """
            Целевая функция для оптимизации гиперпараметров CatBoost
            """
            # Параметры для обучения модели с интервалом для перебора в optuna
            params = {
                'loss_function': 'Logloss',
                'iterations': 100,
                'depth': trial.suggest_int('depth', 4, 10),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
                'random_strength': trial.suggest_float('random_strength', 1e-3, 10.0, log=True),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 1e-3, 10.0, log=True),
                'border_count': trial.suggest_int('border_count', 32, 255),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
            }

            # Логируем параметры в ClearML
            task.connect(params, name="CatBoost Hyperparameters")

            # Обучение модели
            model = CatBoostClassifier(**params, verbose=0)
            model.fit(X_train, y_train)

            # Смотрим на аккуратность модели
            preds = model.predict_proba(X_valid)[:, 1]
            auc = roc_auc_score(y_valid, preds)

            # Логируем аккуратность в ClearML
            Logger.current_logger().report_scalar(
                title="Validation Metrics",
                series="ROC-AUC",
                value=auc,
                iteration=trial.number
            )

            return auc

        # Запускаем обучение модели
        study = optuna.create_study(direction='maximize')
        study.optimize(objective_catboost, n_trials=3)

        # Смотрим параметры лучшей обученной модели
        logger.info(f"Лучшие параметры: {study.best_trial.params}")

        # Тренируем модель уже на всех тренировойных данных, указывая лучшие параметры
        best_model = CatBoostClassifier(**study.best_trial.params, verbose=0)
        best_model.fit(X_train, y_train)

        # Сохраняем модель локально
        best_model.save_model(self.model_path)

        # Делаем копию обученной модели и сохраняем на сервет clearMl
        clearml_model_path = self.model_path + ".clearml_copy"
        shutil.copyfile(self.model_path, clearml_model_path)
        task.update_output_model(model_path=clearml_model_path)

        logger.info("Обучение модели завершено")

    def predict(self, dataset_path):
        """
        Выполняет предсказание с использованием обученной модели
        """
        logger.info("Начало предсказания с использованием обученной модели")

        # Читаем файл с тастовыми данными
        test_data = pd.read_csv(dataset_path)

        # Подготавливаем тестовые данные
        X_test = self._prepare_data(test_data)

        # Загружаем обученную модель
        model = CatBoostClassifier()
        model.load_model(self.model_path)

        # Делаем предсказание
        predictions = model.predict(X_test)
        predictions = (predictions > 0.5).astype(int)

        # Сохраняем файл с предсказанием
        output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Transported': predictions})
        output["Transported"] = output["Transported"].astype(bool)
        output.to_csv(self.results_path, index=False)
        logger.info("Предсказание завершено, результаты сохранены")


if __name__ == '__main__':
    # Указываем основной класс для тренировки модели и предсказания
    classifier = MyClassifierModel()

    # Указываем параметры необходимые к передаче
    parser = argparse.ArgumentParser(description="Обучение и предсказание с использованием CatBoost.")
    parser.add_argument("mode", choices=["train", "predict"], help="Режим работы: обучение или предсказание.")
    parser.add_argument("--dataset", required=True, help="Полный путь к датасету для обучения или предсказания.")

    # Считываем параметры
    args = parser.parse_args()

    # Если train - обучаем модель, если predict - делаем предсказание
    if args.mode == "train":
        classifier.train(args.dataset)
    elif args.mode == "predict":
        classifier.predict(args.dataset)
