import argparse
import optuna
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


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

        numeric_columns = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "Age"
            , "CryoSleep", "VIP"
                           ]
        numerics = data.copy()[numeric_columns]

        columns_for_dummies = ["HomePlanet", "Destination"]
        dummies = pd.get_dummies(cabin_data[["Deck", "Side"]].join(data.copy()[columns_for_dummies]))

        result = pd.concat([numerics, dummies], axis=1)

        # print(len(result.columns))

        return result

    # Функция для обучения модели и её сохранения
    def train(self, path_to_dataset):
        train_data = pd.read_csv(path_to_dataset)
        y = train_data["Transported"]
        X = self._data_preparations(train_data)

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        def objective(trial):
            param = {
                "objective": "binary",
                "metric": "auc",
                "boosting_type": "gbdt",
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 20, 200, step=10),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
                "subsample": trial.suggest_float("subsample", 0.4, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                "random_state": trial.suggest_int("random_state", 1, 200, step=1),
                "n_jobs": -1,
                "verbose": -1
            }
            model = lgb.LGBMClassifier(**param)
            model.fit(X_train, y_train)
            predictions = model.predict(X_val)
            auc = roc_auc_score(y_val, predictions)
            return auc

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=500, show_progress_bar=True)

        trial = study.best_trial

        best_params = trial.params
        best_params["verbose"] = -1
        best_model = lgb.LGBMClassifier(**best_params)
        best_model.fit(X, y)

        best_model.booster_.save_model("./data/model/trained_model.txt")
        return

    # Функция для предсказания при помощи сохраненной модели
    def predict(self, path_to_dataset):
        test_data = pd.read_csv(path_to_dataset)
        X_test = self._data_preparations(test_data)
        model = lgb.Booster(model_file="./data/model/trained_model.txt")
        print(model.params)
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
