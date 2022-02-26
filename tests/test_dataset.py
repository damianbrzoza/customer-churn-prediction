"""
Test dataset module.

created by Damian Brzoza 26.02.2022
"""

import logging

import _pytest
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from src.dataset import CustomerDataset


def test_init(capfd: _pytest.capture.CaptureFixture) -> None:
    """
    Test data import.

    :param capfd: enable capture stdout
    :type capfd: _pytest.capture.CaptureFixture
    :raises err: AssertionError
    """
    try:
        dataset = CustomerDataset(pth="data/bank_data.csv", verbose=True)
    except FileNotFoundError as err:
        logging.error("Testing CustomerDataset init: The file wasn't found")
        raise err

    try:
        assert dataset.df.shape[0] > 0
        assert dataset.df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing CustomerDataset init: The file doesn't appear to have rows and columns"
        )
        raise err

    try:
        out, _ = capfd.readouterr()
        assert out != ""
    except AssertionError as err:
        logging.error("Testing CustomerDataset init: Nothing was printed in console!")
        raise err

    logging.info("Testing CustomerDataset init: SUCCESS")


@pytest.fixture
def customer_dataset() -> CustomerDataset:
    """
    Customer_dataset fixture.

    :return: CustomerDataset object
    :rtype: CustomerDataset
    """
    return CustomerDataset(pth="data/bank_data.csv", verbose=False)


def test_perform_eda(
    customer_dataset: CustomerDataset,
    capfd: _pytest.capture.CaptureFixture,
    monkeypatch: _pytest.monkeypatch.MonkeyPatch,
) -> None:
    """
    Test performing EDA.

    :param customer_dataset: dataset
    :type customer_dataset: CustomerDataset
    :param capfd: enable capture stdout
    :type capfd: _pytest.capture.CaptureFixture
    :param monkeypatch: enable monkeypatching plt.show method
    :type monkeypatch: _pytest.monkeypatch.MonkeyPatch
    :raises err: AssertionError when assert failed
    """
    monkeypatch.setattr(target=plt, name="show", value=lambda: None)

    customer_dataset.perform_eda()

    try:
        out, _ = capfd.readouterr()
        assert out != ""

    except AssertionError as err:
        logging.error("Testing performing eda: No output was printed in console")
        raise err

    logging.info("Testing performing eda: SUCCESS")


def test_feature_engineering(customer_dataset: CustomerDataset) -> None:
    """
    Test feature engineering.

    :param customer_dataset: dataset
    :type customer_dataset: CustomerDataset
    """
    X_train, X_test, y_train, y_test = customer_dataset.perform_feature_engineering()

    try:
        for data in [X_train, X_test, y_train, y_test]:
            assert data.shape[0] > 0

    except AssertionError as err:
        logging.error("Testing splitting data: The file doesn't appear to have rows")
        raise err

    try:
        for dataframe in [X_train, X_test]:
            assert isinstance(dataframe, pd.DataFrame)
        for series in [y_train, y_test]:
            assert isinstance(series, pd.Series)

    except AssertionError as err:
        logging.error("Testing splitting data: Output has incorrect types")
        raise err

    logging.info("Testing splitting data: SUCCESS")
