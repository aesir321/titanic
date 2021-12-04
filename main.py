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
from sklearn.preprocessing import StandardScaler

from AttributeAdder import AttributeAdder

NUM_ATTRIBS = ["Age", "SibSp", "Parch"]
if __name__ == "__main__":
    train = pd.read_csv("./data/train.csv")

    imputer = SimpleImputer(strategy="median")  # how to make category based?
    attr_adder = AttributeAdder()  # adds the ship section, returns a numpy array

    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("std_scaler", StandardScaler()),
        ]
    )
