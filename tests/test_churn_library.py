"""
Test churn_library module.

created by Damian Brzoza 26.02.2022
"""

import logging
from typing import Any, Dict, List, Tuple

import _pytest
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier

from src.churn_library import train_model
from src.dataset import CustomerDataset


@pytest.fixture
def customer_dataset_splitted() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Customer_dataset fixture.

    :return: X_train, X_test, y_train, y_test
    :rtype: Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
    """
    customer_dataset = CustomerDataset(pth="data/bank_data.csv", verbose=False)
    return customer_dataset.perform_feature_engineering()


def test_train_models(
    customer_dataset_splitted: Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
    capfd: _pytest.capture.CaptureFixture,
    monkeypatch: _pytest.monkeypatch.MonkeyPatch,
) -> None:
    """
    Test training model process.

    :param customer_dataset_splitted: splitted dataset
    :type customer_dataset_splitted: Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
    :param capfd: enable capture stdout
    :type capfd: _pytest.capture.CaptureFixture
    :param monkeypatch: enable monkeypatching plt.show method
    :type monkeypatch: _pytest.monkeypatch.MonkeyPatch
    :raises err: AssertionError when assert failed
    """
    monkeypatch.setattr(target=plt, name="show", value=lambda: None)
    X_train, X_test, y_train, y_test = customer_dataset_splitted
    param_grid_rfc_model: Dict[str, List[Any]] = {
        "n_estimators": [200, 500],
    }

    rfc_model = train_model(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        classifier=RandomForestClassifier(random_state=42),
        classifier_name="rfc_model",
        param_grid=param_grid_rfc_model,
        verbose=True,
        n_jobs=1,
    )

    try:
        out, _ = capfd.readouterr()
        assert out != ""

    except AssertionError as err:
        logging.error("Testing train model: Nothing was printed in console!")
        raise err

    try:
        assert isinstance(rfc_model, ClassifierMixin)

    except AssertionError as err:
        logging.error("Testing train model: train_model output should be ClassifierMixin!")
        raise err

    logging.info("Testing performing eda: SUCCESS")
