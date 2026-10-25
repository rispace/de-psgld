"""Loaders for the two real datasets used in the experiments.

adult-all.csv : Adult (Census Income), 48,842 records, from the UCI repository
                (Becker and Kohavi, 1996).
housing.csv   : California housing, 20,640 districts from the 1990 census
                (Pace and Barry, 1997).
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

ADULT_COLS = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
              "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
              "hours_per_week", "native_country", "income"]


def load_adult(path=DATA_DIR / "adult-all.csv", train_size=36180, seed=0):
    """Adult data: drop records with missing values (45,222 remain), take log(1+x) of the
    capital variables, standardize the numerical features, one-hot encode the categorical
    features (drop first), add an intercept, and make a stratified train/test split.

    Returns Xtr, Xte, ytr, yte, feature names.
    """
    df = pd.read_csv(path, header=None, names=ADULT_COLS, skipinitialspace=True)
    df = df[~(df == "?").any(axis=1)].reset_index(drop=True)
    y = df["income"].str.startswith(">").astype(int).values
    df["native_country"] = np.where(df["native_country"] == "United-States", "US", "non-US")
    for c in ["capital_gain", "capital_loss"]:
        df[c] = np.log1p(df[c])
    num = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
    cat = ["workclass", "marital_status", "occupation", "relationship", "race", "sex",
           "native_country"]
    X = pd.get_dummies(df[num + cat], columns=cat, drop_first=True, dtype=float)
    Xtr, Xte, ytr, yte = train_test_split(X.values, y, train_size=train_size, stratify=y,
                                          random_state=seed)
    sc = StandardScaler().fit(Xtr)
    Xtr = np.c_[np.ones(len(Xtr)), sc.transform(Xtr)]
    Xte = np.c_[np.ones(len(Xte)), sc.transform(Xte)]
    return Xtr, Xte, ytr, yte, ["intercept"] + list(X.columns)


def load_housing(path=DATA_DIR / "housing.csv"):
    """California housing: drop districts with missing values (20,433 remain).

    Returns a DataFrame with the coordinates, the response log(median house value),
    and the feature matrix columns (before standardization).
    """
    df = pd.read_csv(path).dropna().reset_index(drop=True)
    df["ocean_proximity"] = df["ocean_proximity"].replace({"ISLAND": "NEAR OCEAN"})
    feat = pd.DataFrame({
        "median income": df["median_income"],
        "house age": df["housing_median_age"],
        "log rooms/household": np.log(df["total_rooms"] / df["households"]),
        "bedrooms/room": df["total_bedrooms"] / df["total_rooms"],
        "log population/household": np.log(df["population"] / df["households"]),
        "latitude": df["latitude"],
        "longitude": df["longitude"]})
    dum = pd.get_dummies(df["ocean_proximity"], drop_first=True, dtype=float)
    dum.columns = [c.lower() for c in dum.columns]
    X = pd.concat([feat, dum], axis=1)
    return df[["latitude", "longitude"]], X, np.log(df["median_house_value"].values)
