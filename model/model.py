import argparse
import optuna
import pandas as pd
import lightgbm as lgb
from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import sys
import logging
import warnings
from logger import SingletonLogger

logger = SingletonLogger().get_logger()
optuna_logger = optuna.logging.get_logger("optuna")
optuna_logger.handlers = logger.handlers

# logging.captureWarnings(True)
# warnings.filterwarnings("always")

class LoggerWriter:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, message):
        if message.strip():  # Игнорируем пустые строки
            self.logger.log(self.level, message.strip())

    def flush(self):
        pass  # Не требуется для логгера

# Перенаправляем stdout и stderr в логгер
sys.stdout = LoggerWriter(logger, logging.INFO)
sys.stderr = LoggerWriter(logger, logging.ERROR)


class My_Classifier_Model:

    # Приватный метод класса для заполнения отсутствующих значений в данных
    def _fill_missing_values(self, data):
        numeric_data = [column for column in data.select_dtypes(["int", "float"])]
        categorical_data = [column for column in data.select_dtypes(exclude=["int", "float"])]

        for col in numeric_data:
            data[col] = data[col].infer_objects(copy=False).fillna(data[col].median())

        for col in categorical_data:
            data[col] = data[col].fillna(data[col].value_counts().index[0]).infer_objects(copy=False)

        return data

    # Приватный метод для подготовки данных в удобный формат
    def _data_preparations(self, data):
        data = self._fill_missing_values(data)

        cabin_data = data["Cabin"].str.split("/", expand=True)
        cabin_data.columns = ["Deck", "Num", "Side"]
        cabin_data["Num"] = cabin_data["Num"].fillna(-1).astype(int)
        cabin_data["Deck"] = cabin_data["Deck"].fillna("Unknown")
        cabin_data["Side"] = cabin_data["Side"].fillna("Unknown")

        data["CryoSleep"] = data["CryoSleep"].astype(bool).fillna(False).astype(int)
        data["VIP"] = data["VIP"].astype(bool).fillna(False).astype(int)

        # Пробую с суммой всех трат
        spends_columns = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        data["Spends"] = data[spends_columns].sum(axis=1)

        data["NoSpends"] = (data["Spends"] == 0)
        # print(data.head())

        numeric_columns = [
            'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', "Spends",
            "NoSpends", "Age", "CryoSleep", "VIP"
        ]
        numerics = data.copy()[numeric_columns]

        columns_for_dummies = ["HomePlanet", "Destination"]
        dummies = pd.get_dummies(cabin_data[["Deck", "Side"]].join(data.copy()[columns_for_dummies]))

        result = pd.concat([numerics, dummies], axis=1)

        # print(len(result.columns))

        return result

    # Функция для обучения модели и её сохранения
    def train(self, path_to_dataset):
        logger.info("Начало обучения модели")
        train_data = pd.read_csv(path_to_dataset)
        y = train_data["Transported"]
        X = self._data_preparations(train_data)

        # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        #
        # def objective(trial):
        #     param = {
        #         "objective": "binary",
        #         "metric": "",
        #         "boosting_type": trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss']),
        #         # "is_provide_training_metric" : True,
        #         "n_estimators": trial.suggest_int("n_estimators", 100, 1500),
        #         "learning_rate": 0.043,
        #         "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        #         "max_depth": trial.suggest_int("max_depth", 3, 5, step=1),
        #         "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
        #         "subsample": trial.suggest_float("subsample", 0.4, 1.0),
        #         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        #         "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        #         "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 12.0, log=True),
        #         "random_state": trial.suggest_int("random_state", 2, 100, step=2),
        #         "n_jobs": -1,
        #         "verbose": -1
        #     }
        #     model = lgb.LGBMClassifier(**param)
        #     model.fit(X_train, y_train)
        #     predictions = model.predict(X_val)
        #     auc = roc_auc_score(y_val, predictions)
        #     return auc
        #
        # study = optuna.create_study(direction="maximize")
        # study.optimize(objective, n_trials=100, show_progress_bar=True)
        #
        # trial = study.best_trial
        #
        # best_params = trial.params
        # print(best_params)
        # best_params["verbose"] = -1
        # best_model = lgb.LGBMClassifier(**best_params)
        # best_model.fit(X, y)
        #
        # best_model.booster_.save_model("./data/model/trained_model.txt")

        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

        def objective_xgb(trial):
            param = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
                'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),
                'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 9),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 1e-3, 10.0, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            }

            model = XGBClassifier(**param)
            model.fit(X_train, y_train)

            preds = model.predict_proba(X_valid)[:, 1]
            auc = roc_auc_score(y_valid, preds)
            return auc

        def objective_lgbm(trial):
            param = {
                "objective": "binary",
                "metric": "",
                "boosting_type": trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss']),
                # "is_provide_training_metric" : True,
                "n_estimators": trial.suggest_int("n_estimators", 100, 1500),
                "learning_rate": 0.043,
                "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                "max_depth": trial.suggest_int("max_depth", 3, 5, step=1),
                "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
                "subsample": trial.suggest_float("subsample", 0.4, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 12.0, log=True),
                "random_state": trial.suggest_int("random_state", 2, 100, step=2),
                "n_jobs": -1,
                "verbose": -1
            }

            model = lgb.LGBMClassifier(**param)
            model.fit(X_train, y_train)

            preds = model.predict_proba(X_valid)[:, 1]
            auc = roc_auc_score(y_valid, preds)
            return auc

        def objective_catboost(trial):
            param = {
                'loss_function': 'Logloss',
                'iterations': 100,
                'depth': trial.suggest_int('depth', 4, 10),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
                'random_strength': trial.suggest_float('random_strength', 1e-3, 10.0, log=True),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 1e-3, 10.0, log=True),
                'border_count': trial.suggest_int('border_count', 32, 255),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
            }

            model = CatBoostClassifier(**param, verbose=0)
            model.fit(X_train, y_train)

            preds = model.predict_proba(X_valid)[:, 1]
            auc = roc_auc_score(y_valid, preds)
            return auc

        # study_xgb = optuna.create_study(direction='maximize')
        # study_lgbm = optuna.create_study(direction='maximize')
        study_catboost = optuna.create_study(direction='maximize')

        # study_xgb.optimize(objective_xgb, n_trials=5)
        # study_lgbm.optimize(objective_lgbm, n_trials=5)
        study_catboost.optimize(objective_catboost, n_trials=3)

        # best_trial_xgb = study_xgb.best_trial.params
        # best_trial_lgbm = study_lgbm.best_trial.params
        best_trial_catboost = study_catboost.best_trial.params

        # print('Best trial for XGBoost:', best_trial_xgb)
        # print('Best trial for LightGBM:', best_trial_lgbm)
        print('Best trial for CatBoost:', best_trial_catboost)

        # best_xgb = XGBClassifier(**study_xgb.best_trial.params)
        # best_xgb.fit(X_train, y_train)
        # auc_xgb = roc_auc_score(y_valid, best_xgb.predict_proba(X_valid)[:, 1])
        #
        # best_lgbm = lgb.LGBMClassifier(**study_lgbm.best_trial.params)
        # best_lgbm.fit(X_train, y_train)
        # auc_lgbm = roc_auc_score(y_valid, best_lgbm.predict_proba(X_valid)[:, 1])

        best_catboost = CatBoostClassifier(**study_catboost.best_trial.params, verbose=0)
        best_catboost.fit(X_train, y_train)
        # auc_catboost = roc_auc_score(y_valid, best_catboost.predict_proba(X_valid)[:, 1])

        # # Print the best AUC scores for each algorithm
        # print(f'Best AUC for XGBoost: {auc_xgb:.4f}')
        # print(f'Best AUC for LightGBM: {auc_lgbm:.4f}')
        # print(f'Best AUC for CatBoost: {auc_catboost:.4f}')

        best_catboost.save_model('./data/model/catboost_model.cbm')
        logger.info("Обучение модели завершено")
        return

    # Функция для предсказания при помощи сохраненной модели
    def predict(self, path_to_dataset):
        test_data = pd.read_csv(path_to_dataset)
        X_test = self._data_preparations(test_data)
        # model = lgb.Booster(model_file="./data/model/trained_model.txt")
        model = CatBoostClassifier()
        model.load_model('./data/model/catboost_model.cbm')
        # print(model.params)
        predictions = model.predict(X_test)
        predictions = (predictions > 0.5).astype(int)
        output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Transported': predictions})
        output["Transported"] = output["Transported"].astype(bool)
        output.to_csv('./data/results.csv', index=False)
        return


if __name__ == '__main__':
    # Указываем основной класс для тренировки модели и предсказания
    classifier = My_Classifier_Model()

    # Указываем параметры необходимые к передаче
    parser = argparse.ArgumentParser()
    parser.add_argument("mode",
                        choices=["train", "predict"])
    parser.add_argument("--dataset",
                        required=True,
                        help="Full path to your dataset either for training or predicting.")

    # Считываем параметры
    args = parser.parse_args()

    # Если train - обучаем модель, если predict - делаем предсказание
    if args.mode == "train":
        classifier.train(args.dataset)
    elif args.mode == "predict":
        classifier.predict(args.dataset)
