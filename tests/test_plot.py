"""
Test plot module.

created by Damian Brzoza 26.02.2022
"""

import logging
from typing import List, Tuple

import _pytest
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from sklearn.base import ClassifierMixin

from src.config import CAT_COLUMNS, QUANT_COLUMNS
from src.dataset import CustomerDataset
from src.plot import (
    compare_roc_curve,
    plot_boxplot_with_category,
    plot_categorical_proportion,
    plot_classifier_summary_as_image,
    plot_correlations,
    plot_target_proportion,
    show_features_importance,
)

monkeypatch = _pytest.monkeypatch.MonkeyPatch()  # pytest object that enable monkeypatching
monkeypatch.setattr(target=plt, name="show", value=lambda: None)  # turn off showing plots


@pytest.fixture
def customer_dataset() -> CustomerDataset:
    """
    Customer_dataset fixture.

    :return: CustomerDataset object
    :rtype: CustomerDataset
    """
    return CustomerDataset(pth="data/bank_data.csv", verbose=False)


def test_plot_correlation(customer_dataset: CustomerDataset) -> None:
    """
    Test plotting correlation between variables in dataset.

    :param customer_dataset: dataset
    :type customer_dataset: CustomerDataset
    """
    try:
        plot_correlations(customer_dataset.df)
    except Exception as err:
        logging.error("Testing plot_correlations: FAILED")
        raise err

    logging.info("Testing plot_correlations: SUCCESS")


def test_plot_target_proportion(customer_dataset: CustomerDataset) -> None:
    """
    Test plotting proportion between target values in different groups.

    :param customer_dataset: dataset
    :type customer_dataset: CustomerDataset
    """
    try:
        plot_target_proportion(customer_dataset.df)
    except Exception as err:
        logging.error("Testing target_proportion: FAILED")
        raise err

    logging.info("Testing target_proportion: SUCCESS")


@pytest.mark.parametrize("column_name", CAT_COLUMNS[1:])
def test_plot_categorical_proportion(customer_dataset: CustomerDataset, column_name: str) -> None:
    """
    Test plotting categorical_proportion.

    :param customer_dataset: dataset
    :type customer_dataset: CustomerDataset
    :param column_name: name of the column
    :type column_name: str
    """
    try:
        plot_categorical_proportion(customer_dataset.df, column_name)
    except Exception as err:
        logging.error("Testing plot_categorical_proportion: FAILED")
        raise err

    logging.info(f"Testing plot_categorical_proportion, column_name = {column_name}: SUCCESS")


@pytest.mark.parametrize("column_name", QUANT_COLUMNS)
def test_plot_boxplot_with_category(customer_dataset: CustomerDataset, column_name: str) -> None:
    """
    Test plotting correlation between variables in dataset.

    :param customer_dataset: dataset
    :type customer_dataset: CustomerDataset
    :param column_name: name of the column
    :type column_name: str
    """
    try:
        plot_boxplot_with_category(customer_dataset.df, column_name)
    except Exception as err:
        logging.error("Testing plot_boxplot_with_category: FAILED")
        raise err
    logging.info(f"Testing boxplot_with_category, column_name = {column_name}: SUCCESS")


@pytest.fixture
def list_of_models() -> List[ClassifierMixin]:
    """
    Return loaded models from checkpoints.

    :return: list of classifiers
    :rtype: List[ClassifierMixin]
    """
    rfc_model = joblib.load("./models/rfc_model.pkl")
    logistic_model = joblib.load("./models/logistic_model.pkl")
    return [rfc_model, logistic_model]


@pytest.fixture
def data(
    customer_dataset: CustomerDataset,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Return splitted dataset from given dataset.

    :param customer_dataset: dataset
    :type customer_dataset: CustomerDataset
    :return: X_train, X_test, y_train, y_test
    :rtype: Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
    """
    X_train, X_test, y_train, y_test = customer_dataset.perform_feature_engineering()
    return X_train, X_test, y_train, y_test


def test_compare_roc_curve(
    list_of_models: List[ClassifierMixin],
    data: Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
) -> None:
    """
    Test plotting comparison between ROC curves.

    :param list_of_models: list of classifiers
    :type list_of_models: List[ClassifierMixin]
    :param data: X_train, X_test, y_train, y_test
    :type data: Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
    """
    try:
        _, X_test, _, y_test = data
        compare_roc_curve(list_of_models, X_test=X_test, y_test=y_test)
    except Exception as err:
        logging.error("Testing compare_roc_curve: FAILED")
        raise err

    logging.info("Testing compare_roc_curve: SUCCESS")


def test_show_features_importance(
    list_of_models: List[ClassifierMixin],
    data: Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
) -> None:
    """
    Test plotting features importance.

    :param list_of_models: list of classifiers
    :type list_of_models: List[ClassifierMixin]
    :param data: X_train, X_test, y_train, y_test
    :type data: Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
    """
    try:
        _, X_test, _, _ = data
        show_features_importance(list_of_models[0], X_test)

    except Exception as err:
        logging.error("Testing show_features_importance: FAILED")
        raise err

    logging.info("Testing show_features_importance: SUCCESS")


def test_plot_classifier_summary_as_image(
    list_of_models: List[ClassifierMixin],
    data: Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
) -> None:
    """
    Test plotting classifier summary as image.

    :param list_of_models: list of classifiers
    :type list_of_models: List[ClassifierMixin]
    :param data: X_train, X_test, y_train, y_test
    :type data: Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
    """
    try:
        X_train, X_test, y_train, y_test = data
        for model in list_of_models:
            plot_classifier_summary_as_image(
                model, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
            )

    except Exception as err:
        logging.error("Testing plot_classifier_summary_as_image: FAILED")
        raise err

    logging.info("Testing plot_classifier_summary_as_image: SUCCESS")
