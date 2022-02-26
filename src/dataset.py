"""
Module enable processing data for customer churn analysis.

created by Damian Brzoza 26.02.2022
"""
from copy import deepcopy
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import CAT_COLUMNS, QUANT_COLUMNS
from src.plot import (
    plot_boxplot_with_category,
    plot_categorical_proportion,
    plot_correlations,
    plot_quant_histogram,
    plot_target_proportion,
)


class CustomerDataset:
    """Store df and enable some operation on it."""

    def __init__(self, pth: str, verbose: bool = False):
        """
        Return dataframe for the csv found at pth.

        :param pth: a path to the csv
        :type pth: str
        :param verbose: verbose or not?, defaults to False
        :type verbose: bool, optional
        """
        self.df = pd.read_csv(pth, index_col=0)
        self.df[CAT_COLUMNS] = self.df[CAT_COLUMNS].astype("category")

        if verbose:
            print(self.df.head())
            print(self.df.info())

    def perform_eda(self) -> None:
        """Perform EDA on Dataset and save figures to images folder."""
        data = deepcopy(self.df)

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
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Enable efficient training on that data. Return splitted dataset.

        :return: X_train, X_test, y_train, y_testo
        :rtype: Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        """
        df = deepcopy(self.df)
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


if __name__ == "__main__":  # pragma: no cover
    customer_dataset = CustomerDataset("./data/bank_data.csv", verbose=False)
    customer_dataset.perform_eda()
    X_train, X_test, y_train, y_test = customer_dataset.perform_feature_engineering()
