"""Module enable customer churn analysis."""
from copy import deepcopy
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

from config import CAT_COLUMNS, QUANT_COLUMNS
from src.plot_chart import (
    compare_roc_curve,
    plot_boxplot_with_category,
    plot_categorical_proportion,
    plot_classifier_summary_as_image,
    plot_correlations,
    plot_quant_histogram,
    plot_target_proportion,
    show_features_importance,
)


def import_data(pth: str, verbose: bool = False) -> pd.DataFrame:
    """
    Return dataframe for the csv found at pth.

    :param pth: a path to the csv
    :type pth: str
    :return: pandas dataframe with imported data
    :rtype: pd.DataFrame
    """
    df = pd.read_csv(pth, index_col=0)

    df[CAT_COLUMNS] = df[CAT_COLUMNS].astype("category")

    if verbose:
        print(df.head())
        print(df.info())

    return df


def perform_eda(df: pd.DataFrame) -> None:
    """
    Perform EDA on df and save figures to images folder.

    :param df: pandas dataframe
    :type df: pd.DataFrame
    """
    data = deepcopy(df)

    print(data.head())
    print(data.shape)
    print(data.isnull().sum())
    print(data.describe())

    data["Churn"] = (
        data["Attrition_Flag"]
        .apply(lambda val: 0 if val == "Existing Customer" else 1)
        .astype("int64")
    )

    # Get some info about correlation between our target and numerical values

    plot_correlations(data)
    plot_target_proportion(data)  # Univariate, quantitative plot

    plot_quant_histogram(data, column_name=QUANT_COLUMNS[0])  # Univariate, categorical plot

    for cat in CAT_COLUMNS[1:]:  # without Attrition_Flag
        plot_categorical_proportion(data, column_name=cat)  # Bivariate plot

    plot_boxplot_with_category(data, column_name=QUANT_COLUMNS[0])  # Bivariate plot


def perform_feature_engineering(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Enable efficient training on that data. Return splitted dataset.

    :param df: data
    :type df: pd.DataFrame
    :return: X_train, X_test, y_train, y_test
    :rtype: Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
    """
    df.drop(columns="CLIENTNUM", inplace=True)  # not store any valuable information

    df[QUANT_COLUMNS] = StandardScaler().fit_transform(
        df[QUANT_COLUMNS]
    )  # apply for logistic regression

    df[["Education_Level", "Marital_Status", "Income_Category"]] = df[
        ["Education_Level", "Marital_Status", "Income_Category"]
    ].applymap(
        lambda x: np.nan if x == "Unknown" else x
    )  # correct form for one-hot encoding

    df = pd.get_dummies(df[CAT_COLUMNS[1:]]).join(df).drop(columns=df[CAT_COLUMNS[1:]])

    return train_test_split(
        df.drop(columns="Attrition_Flag"),
        df["Attrition_Flag"],
        test_size=0.3,
        random_state=42,
        stratify=df["Attrition_Flag"],
    )


def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    classifier: ClassifierMixin,
    classifier_name: str,
    param_grid: Dict[str, List[Any]],
) -> ClassifierMixin:
    """
    Train, store model results: images + scores, and store models.

    :param X_train: train features
    :type X_train: pd.DataFrame
    :param X_test: test features
    :type X_test: pd.DataFrame
    :param y_train: train labels
    :type y_train: pd.Series
    :param y_test: test labels
    :type y_test: pd.Series
    :param classifier: classifier/model
    :type classifier: ClassifierMixin
    :param classifier_name: name of classifier
    :type classifier_name: str
    :param param_grid: parameteres to be checked for given classifier
    :type param_grid: Dict[str, List[Any]]
    :return: trained model
    :rtype: ClassifierMixin
    """
    classifier = GridSearchCV(
        estimator=classifier, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2
    )
    classifier.fit(X_train, y_train)

    y_train_preds = classifier.best_estimator_.predict(X_train)
    y_test_preds = classifier.best_estimator_.predict(X_test)

    # scores
    print(f"{classifier_name} results")
    print("test results")
    print(classification_report(y_test, y_test_preds))
    print("train results")
    print(classification_report(y_train, y_train_preds))

    # plot
    RocCurveDisplay.from_estimator(estimator=classifier.best_estimator_, X=X_test, y=y_test)
    plt.show()

    # save best model
    joblib.dump(classifier.best_estimator_, f"./models/{classifier_name}.pkl")

    return classifier


if __name__ == "__main__":
    df = import_data("./data/bank_data.csv", verbose=False)
    perform_eda(df)
    X_train, X_test, y_train, y_test = perform_feature_engineering(df)

    param_grid_rfc_model: Dict[str, List[Any]] = {
        "n_estimators": [200, 500],
        "max_features": ["auto", "sqrt"],
        "max_depth": [4, 5, 100],
        "criterion": ["gini", "entropy"],
    }

    param_grid_logistic_model = {"C": [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

    rfc_model = train_model(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        classifier=RandomForestClassifier(random_state=42),
        classifier_name="rfc_model",
        param_grid=param_grid_rfc_model,
    )
    logistic_model = train_model(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        classifier=LogisticRegression(),
        classifier_name="logistic_model",
        param_grid=param_grid_logistic_model,
    )

    rfc_model = joblib.load("./models/rfc_model.pkl")
    logistic_model = joblib.load("./models/logistic_model.pkl")

    compare_roc_curve(list_of_models=[rfc_model, logistic_model], X_test=X_test, y_test=y_test)
    show_features_importance(rfc_model, X_test)
    plot_classifier_summary_as_image(
        rfc_model, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
    )
