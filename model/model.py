# import argparse
# import optuna
# import pandas as pd
# from catboost import CatBoostClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import roc_auc_score
# import sys
# import logging
# import warnings
# from logger import SingletonLogger
#
# logger = SingletonLogger().get_logger()
# optuna_logger = optuna.logging.get_logger("optuna")
# optuna_logger.handlers = logger.handlers
#
# # logging.captureWarnings(True)
# # warnings.filterwarnings("always")
#
# class LoggerWriter:
#     def __init__(self, logger, level):
#         self.logger = logger
#         self.level = level
#
#     def write(self, message):
#         if message.strip():  # Игнорируем пустые строки
#             self.logger.log(self.level, message.strip())
#
#     def flush(self):
#         pass  # Не требуется для логгера
#
# # Перенаправляем stdout и stderr в логгер
# sys.stdout = LoggerWriter(logger, logging.INFO)
# sys.stderr = LoggerWriter(logger, logging.ERROR)
#
#
# class My_Classifier_Model:
#
#     # Приватный метод класса для заполнения отсутствующих значений в данных
#     def _fill_missing_values(self, data):
#         numeric_data = [column for column in data.select_dtypes(["int", "float"])]
#         categorical_data = [column for column in data.select_dtypes(exclude=["int", "float"])]
#
#         for col in numeric_data:
#             data[col] = data[col].infer_objects(copy=False).fillna(data[col].median())
#
#         for col in categorical_data:
#             data[col] = data[col].fillna(data[col].value_counts().index[0]).infer_objects(copy=False)
#
#         return data
#
#     # Приватный метод для подготовки данных в удобный формат
#     def _data_preparations(self, data):
#         data = self._fill_missing_values(data)
#
#         cabin_data = data["Cabin"].str.split("/", expand=True)
#         cabin_data.columns = ["Deck", "Num", "Side"]
#         cabin_data["Num"] = cabin_data["Num"].fillna(-1).astype(int)
#         cabin_data["Deck"] = cabin_data["Deck"].fillna("Unknown")
#         cabin_data["Side"] = cabin_data["Side"].fillna("Unknown")
#
#         data["CryoSleep"] = data["CryoSleep"].astype(bool).fillna(False).astype(int)
#         data["VIP"] = data["VIP"].astype(bool).fillna(False).astype(int)
#
#         # Пробую с суммой всех трат
#         spends_columns = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
#         data["Spends"] = data[spends_columns].sum(axis=1)
#
#         data["NoSpends"] = (data["Spends"] == 0)
#         # print(data.head())
#
#         numeric_columns = [
#             'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', "Spends",
#             "NoSpends", "Age", "CryoSleep", "VIP"
#         ]
#         numerics = data.copy()[numeric_columns]
#
#         columns_for_dummies = ["HomePlanet", "Destination"]
#         dummies = pd.get_dummies(cabin_data[["Deck", "Side"]].join(data.copy()[columns_for_dummies]))
#
#         result = pd.concat([numerics, dummies], axis=1)
#
#         # print(len(result.columns))
#
#         return result
#
#     # Функция для обучения модели и её сохранения
#     def train(self, path_to_dataset):
#         logger.info("Начало обучения модели")
#         train_data = pd.read_csv(path_to_dataset)
#         y = train_data["Transported"]
#         X = self._data_preparations(train_data)
#
#         X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
#
#         def objective_catboost(trial):
#             param = {
#                 'loss_function': 'Logloss',
#                 'iterations': 100,
#                 'depth': trial.suggest_int('depth', 4, 10),
#                 'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
#                 'random_strength': trial.suggest_float('random_strength', 1e-3, 10.0, log=True),
#                 'bagging_temperature': trial.suggest_float('bagging_temperature', 1e-3, 10.0, log=True),
#                 'border_count': trial.suggest_int('border_count', 32, 255),
#                 'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
#             }
#
#             model = CatBoostClassifier(**param, verbose=0)
#             model.fit(X_train, y_train)
#
#             preds = model.predict_proba(X_valid)[:, 1]
#             auc = roc_auc_score(y_valid, preds)
#             return auc
#
#
#         study_catboost = optuna.create_study(direction='maximize')
#
#
#         study_catboost.optimize(objective_catboost, n_trials=3)
#
#
#         best_trial_catboost = study_catboost.best_trial.params
#
#
#         print('Best trial for CatBoost:', best_trial_catboost)
#
#
#
#         best_catboost = CatBoostClassifier(**study_catboost.best_trial.params, verbose=0)
#         best_catboost.fit(X_train, y_train)
#
#
#         best_catboost.save_model('./data/model/catboost_model.cbm')
#         logger.info("Обучение модели завершено")
#         return
#
#     # Функция для предсказания при помощи сохраненной модели
#     def predict(self, path_to_dataset):
#         test_data = pd.read_csv(path_to_dataset)
#         X_test = self._data_preparations(test_data)
#         model = CatBoostClassifier()
#         model.load_model('./data/model/catboost_model.cbm')
#         predictions = model.predict(X_test)
#         predictions = (predictions > 0.5).astype(int)
#         output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Transported': predictions})
#         output["Transported"] = output["Transported"].astype(bool)
#         output.to_csv('./data/results.csv', index=False)
#         return
#
#
# if __name__ == '__main__':
#     # Указываем основной класс для тренировки модели и предсказания
#     classifier = My_Classifier_Model()
#
#     # Указываем параметры необходимые к передаче
#     parser = argparse.ArgumentParser()
#     parser.add_argument("mode",
#                         choices=["train", "predict"])
#     parser.add_argument("--dataset",
#                         required=True,
#                         help="Full path to your dataset either for training or predicting.")
#
#     # Считываем параметры
#     args = parser.parse_args()
#
#     # Если train - обучаем модель, если predict - делаем предсказание
#     if args.mode == "train":
#         classifier.train(args.dataset)
#     elif args.mode == "predict":
#         classifier.predict(args.dataset)


import argparse
import shutil

import optuna
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import sys
import logging
import warnings
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

# Перенаправление stdout и stderr в логгер
class LoggerWriter:
    """
    Класс для перенаправления вывода в логгер.
    """
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, message):
        if message.strip():  # Игнорируем пустые строки
            self.logger.log(self.level, message.strip())

    def flush(self):
        pass

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

        for col in numeric_data:
            data[col] = data[col].fillna(data[col].median())

        for col in categorical_data:
            data[col] = data[col].fillna(data[col].mode()[0])

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

        # Преобразование булевых колонок
        data["CryoSleep"] = data["CryoSleep"].astype(bool).fillna(False)
        data["VIP"] = data["VIP"].astype(bool).fillna(False)

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
        Обучает модель CatBoostClassifier на предоставленных данных.
        """
        logger.info("Начало обучения модели")
        train_data = pd.read_csv(dataset_path)
        y = train_data["Transported"]
        X = self._prepare_data(train_data)

        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

        def objective_catboost(trial):
            """
            Целевая функция для оптимизации гиперпараметров CatBoost.
            """
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

            model = CatBoostClassifier(**params, verbose=0)
            model.fit(X_train, y_train)

            preds = model.predict_proba(X_valid)[:, 1]
            auc = roc_auc_score(y_valid, preds)

            # Логируем метрику в ClearML
            Logger.current_logger().report_scalar(
                title="Validation Metrics",
                series="ROC-AUC",
                value=auc,
                iteration=trial.number
            )

            return auc

        study = optuna.create_study(direction='maximize')
        study.optimize(objective_catboost, n_trials=3)

        logger.info(f"Лучшие параметры: {study.best_trial.params}")

        best_model = CatBoostClassifier(**study.best_trial.params, verbose=0)
        best_model.fit(X_train, y_train)
        best_model.save_model(self.model_path)

        clearml_model_path = self.model_path + ".clearml_copy"
        shutil.copyfile(self.model_path, clearml_model_path)
        task.update_output_model(model_path=clearml_model_path)


        logger.info("Обучение модели завершено")

    def predict(self, dataset_path):
        """
        Выполняет предсказание с использованием обученной модели.
        """
        test_data = pd.read_csv(dataset_path)
        X_test = self._prepare_data(test_data)

        model = CatBoostClassifier()
        model.load_model(self.model_path)
        predictions = model.predict(X_test)
        predictions = (predictions > 0.5).astype(int)

        output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Transported': predictions})
        output["Transported"] = output["Transported"].astype(bool)
        output.to_csv(self.results_path, index=False)
        logger.info("Предсказание завершено, результаты сохранены")

if __name__ == '__main__':
    classifier = MyClassifierModel()

    parser = argparse.ArgumentParser(description="Обучение и предсказание с использованием CatBoost.")
    parser.add_argument("mode", choices=["train", "predict"], help="Режим работы: обучение или предсказание.")
    parser.add_argument("--dataset", required=True, help="Полный путь к датасету для обучения или предсказания.")

    args = parser.parse_args()

    if args.mode == "train":
        classifier.train(args.dataset)
    elif args.mode == "predict":
        classifier.predict(args.dataset)