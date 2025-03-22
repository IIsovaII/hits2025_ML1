# Me
### Sofia Kovalenko 972303

---
# What is it

So this project is dedicated to solving a problem on ML as part of studying at the university. 

Competition: https://www.kaggle.com/competitions/tsumladvanced2025. 

The project contains jupyter notebooks in `/notebooks`, where I tried to study the input data of the competition, experimented with the models. 

The final model is wrapped in a CLI application, the model code is in `./model/model.py`. The final solution uses CatBoostClassifier.

---
# How to

## Train Model Using CLI

To train the model, run the following command from the root repository folder:

```
python ./model/model.py train --dataset=/path/to/dataset
```

- Replace `/path/to/dataset` with the actual path to your dataset.
- The model will be trained using the provided dataset and saved for future use in `./data/model/trained_model.txt`.
- You can use `./data/inputs/train.csv` as train dataset

## Get Predictions Using CLI

To generate predictions using the trained model, run the following command from the root repository folder:

```
python ./model/model.py predict --dataset=/path/to/dataset
```

- Replace `/path/to/dataset` with the path to the dataset for which you want predictions.
- The predictions will be generated and saved in `./data/results.csv`.
- You can use `./data/inputs/test.csv` as train dataset

---

# Resources

Here is what I used and studied for this project:
- **Poetry Installation**: [https://python-poetry.org/docs/](https://python-poetry.org/docs/)
- **Poetry Initializer in PyCharm**: [https://www.jetbrains.com/help/pycharm/poetry.html](https://www.jetbrains.com/help/pycharm/poetry.html)
- **How to Write a README.md**: [https://gist.github.com/alinastorm/8a04cdbc36be9c051a66f90ae6d6df35](https://gist.github.com/alinastorm/8a04cdbc36be9c051a66f90ae6d6df35)
- **Optuna Documentation**: [https://optuna.readthedocs.io/en/stable/index.html](https://optuna.readthedocs.io/en/stable/index.html)
- **LightGBM Documentation**: [https://lightgbm.readthedocs.io/en/v3.3.5/index.html](https://lightgbm.readthedocs.io/en/v3.3.5/index.html)
- **Pandas in Python**: [https://docs-python.ru/packages/modul-pandas-analiz-dannykh-python/](https://docs-python.ru/packages/modul-pandas-analiz-dannykh-python/)
- **Info about logging**: https://habr.com/ru/companies/wunderfund/articles/683880/
- **More info about logging**: https://medium.com/@balakrishnamaduru/mastering-logging-in-python-a-singleton-logger-with-dynamic-log-levels-17f3e8bfa2cf
- **Logging in python documentation**: https://docs.python.org/3/library/logging.html
- **Pyplot tutorial**: https://cpp-python-nsu.inp.nsk.su/textbook/sec4/ch8
- **Seaborn Documentation**: https://seaborn.pydata.org/index.html
- **ClearML Documentation**: https://clear.ml/docs/latest/docs/
- **ClearML tutorial**: https://habr.com/ru/articles/691314/
---