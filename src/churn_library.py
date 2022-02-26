"""
Module enable customer churn analysis.

created by Damian Brzoza 26.02.2022
"""
from typing import Any, Dict, List

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, classification_report
from sklearn.model_selection import GridSearchCV

from src.dataset import CustomerDataset
from src.plot import compare_roc_curve, plot_classifier_summary_as_image, show_features_importance


def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    classifier: ClassifierMixin,
    classifier_name: str,
    param_grid: Dict[str, List[Any]],
    verbose: bool = True,
    n_jobs: int = -1,
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
    :param verbose: verbose or not?, defaults to False
    :type verbose: bool, optional
    :param n_jobs: how many jobs to run in parallel in GridSearchCV, defaults -1
    :type n_jobs: int, optional
    :return: trained model
    :rtype: ClassifierMixin
    """
    classifier = GridSearchCV(
        estimator=classifier, param_grid=param_grid, cv=5, n_jobs=n_jobs, verbose=2
    )
    classifier.fit(X_train, y_train)

    y_train_preds = classifier.best_estimator_.predict(X_train)
    y_test_preds = classifier.best_estimator_.predict(X_test)

    # scores
    if verbose:
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

    return classifier.best_estimator_


def main(make_eda: bool = False, load_existing_model: bool = True) -> None:  # pragma: no cover
    """
    Perform all available operation.

    :param make_eda: make_eda or not?, defaults to False
    :type make_eda: bool, optional
    :param load_existing_model: load_existing_model or train new one?, defaults to True
    :type load_existing_model: bool, optional
    """
    customer_dataset = CustomerDataset("./data/bank_data.csv", verbose=False)
    customer_dataset.perform_eda()
    X_train, X_test, y_train, y_test = customer_dataset.perform_feature_engineering()

    if load_existing_model:
        rfc_model = joblib.load("./models/rfc_model.pkl")
        logistic_model = joblib.load("./models/logistic_model.pkl")

    else:
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

    compare_roc_curve(list_of_models=[rfc_model, logistic_model], X_test=X_test, y_test=y_test)
    show_features_importance(rfc_model, X_test)
    plot_classifier_summary_as_image(
        rfc_model, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
    )


if __name__ == "__main__":  # pragma: no cover
    main()
