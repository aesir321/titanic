# TODO:
# import data
# handle missing values
# create features
# try some models
# find optimal params
# submit best model

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from AttributeAdder import AttributeAdder
from ManualFeatureSelector import ManualFeatureSelector

NUM_ATTRIBS = ["Age", "SibSp", "Parch", "Fare"]
CAT_ATTRIBS = ["Pclass", "Sex", "Embarked", "Cabin"]  # ship_section is created later..
TARGET = "Survived"
RANDOM_STATE = 42


def display_scores(scores, name):
    print(f"-----{name}-----")
    print(scores)
    print(scores.mean())
    print(scores.std())


def test_broad_classifiers(X_train, y_train):
    names = [
        "Nearest Neighbors",
        "Linear SVM",
        "RBF SVM",
        "Gaussian Process",
        "Decision Tree",
        "Random Forest",
        "Neural Net",
        "AdaBoost",
        "Naive Bayes",
        "QDA",
    ]

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025, random_state=RANDOM_STATE),
        SVC(gamma=2, C=1, random_state=RANDOM_STATE),
        GaussianProcessClassifier(1.0 * RBF(1.0), random_state=RANDOM_STATE),
        DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
        RandomForestClassifier(
            max_depth=5, n_estimators=10, max_features=1, random_state=RANDOM_STATE
        ),
        MLPClassifier(alpha=1, max_iter=1000, random_state=RANDOM_STATE),
        AdaBoostClassifier(random_state=RANDOM_STATE),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
    ]

    model_scores = {}
    for name, clf in zip(names, classifiers):
        scores = cross_val_score(clf, X_train, y_train, scoring="f1", cv=5, n_jobs=-1)
        model_scores[name] = scores.mean()
        display_scores(scores, name)


if __name__ == "__main__":
    train = pd.read_csv("./data/train.csv")

    X_train = train.drop(axis=1, columns=TARGET)
    y_train = train[TARGET]

    num_pipeline = Pipeline(
        [
            # how to make category based?
            ("imputer", SimpleImputer(strategy="median")),
            ("std_scaler", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        [
            # adds the ship section, returns a numpy array
            ("attribs_adder", AttributeAdder()),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    full_pipeline = ColumnTransformer(
        [
            ("num_pipeline", num_pipeline, NUM_ATTRIBS),
            ("cat_pipeline", cat_pipeline, CAT_ATTRIBS),
        ]
    )

    X_train_prepared = full_pipeline.fit_transform(X_train)

    test_broad_classifiers(X_train_prepared, y_train)
