import pandas as pd
from main import create_pipeline, import_data
from constants import RANDOM_STATE, TARGET
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.gaussian_process import GaussianProcessClassifier

if __name__ == "__main__":
    X_train, y_train = import_data()

    pipeline = create_pipeline()

    X_train_prepared = pipeline.fit_transform(X_train)

    param_distribs = {
        "n_restarts_optimizer": randint(low=1, high=200),
        "max_iter_predict": randint(low=10, high=200),
        "warm_start": [True, False],
    }

    rnd_search = RandomizedSearchCV(
        GaussianProcessClassifier(),
        param_distributions=param_distribs,
        cv=5,
        random_state=RANDOM_STATE,
        scoring="f1",
        n_jobs=-1,
    )
    rnd_search.fit(X_train_prepared, y_train)
    cvres = rnd_search.cv_results_

    for f1, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(f1, params)
