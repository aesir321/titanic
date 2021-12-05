import pandas as pd
from main import create_pipeline, import_data
from constants import RANDOM_STATE, TARGET
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier

if __name__ == "__main__":
    X_train, y_train = import_data()

    pipeline = create_pipeline()

    X_train_prepared = pipeline.fit_transform(X_train)

    param_distribs_gp = {
        "n_restarts_optimizer": randint(low=1, high=200),
        "max_iter_predict": randint(low=10, high=200),
        "warm_start": [True, False],
    }

    rnd_search = RandomizedSearchCV(
        GaussianProcessClassifier(),
        param_distributions=param_distribs_gp,
        cv=5,
        random_state=RANDOM_STATE,
        scoring="f1",
        n_jobs=-1,
    )

    rnd_search.fit(X_train_prepared, y_train)
    cvres = rnd_search.cv_results_

    for f1, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(f1, params)

    param_distribs_rf = {
        "n_estimators": randint(low=1, high=300),
        # "criterion": ["gini", "entropy"],
        "max_depth": randint(low=1, high=100),
        "min_samples_split": randint(low=2, high=100),
        "max_features": ["auto", "sqrt", "log2"],
        "bootstrap": [True, False],
    }

    rnd_search = RandomizedSearchCV(
        RandomForestClassifier(),
        param_distributions=param_distribs_rf,
        cv=5,
        random_state=RANDOM_STATE,
        scoring="accuracy",
        n_jobs=-1,
    )
    rnd_search.fit(X_train_prepared, y_train)
    cvres = rnd_search.cv_results_

    for f1, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(f1, params)
