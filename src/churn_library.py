"""Module enable customer churn analysis."""
import pandas as pd

from config import CAT_COLUMNS, QUANT_COLUMNS
from src.plot_chart import (
    plot_boxplot_with_category,
    plot_categorical_proportion,
    plot_correlations,
    plot_quant_histogram,
    plot_target_proportion,
)


def import_data(pth: str, verbose=False) -> pd.DataFrame:
    """
    Returns dataframe for the csv found at pth.
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
    print(df.head())
    print(df.shape)
    print(df.isnull().sum())
    print(df.describe())

    df["Churn"] = (
        df["Attrition_Flag"]
        .apply(lambda val: 0 if val == "Existing Customer" else 1)
        .astype("int64")
    )

    # Get some info about correlation between our target and numerical values

    plot_correlations(df)
    plot_target_proportion(df)  # Univariate, quantitative plot

    plot_quant_histogram(df, column_name=QUANT_COLUMNS[0])  # Univariate, categorical plot

    for cat in CAT_COLUMNS[1:]:  # without Attrition_Flag
        plot_categorical_proportion(df, column_name=cat)  # Bivariate plot

    plot_boxplot_with_category(df, column_name=QUANT_COLUMNS[0])  # Bivariate plot


def perform_feature_engineering(df, response):
    """
    input:
        df: pandas dataframe
        response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
        X_train: X training data
        X_test: X testing data
        y_train: y training data
        y_test: y testing data
    """

    return train_test_split(X, y, test_size=0.3, random_state=42)


def classification_report_image(
    y_train, y_test, y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf
):
    """
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
        y_train: training response values
        y_test:  test response values
        y_train_preds_lr: training predictions from logistic regression
        y_train_preds_rf: training predictions from random forest
        y_test_preds_lr: test predictions from logistic regression
        y_test_preds_rf: test predictions from random forest

    output:
        None
    """
    pass


def feature_importance_plot(model, X_data, output_pth):
    """
    creates and stores the feature importances in pth
    input:
        model: model object containing feature_importances_
        X_data: pandas dataframe of X values
        output_pth: path to store the figure

    output:
        None
    """
    pass


def train_models(X_train, X_test, y_train, y_test):
    """
    train, store model results: images + scores, and store models
    input:
        X_train: X training data
        X_test: X testing data
        y_train: y training data
        y_test: y testing data
    output:
        None
    """
    pass


if __name__ == "__main__":
    df = import_data("./data/bank_data.csv", verbose=False)
    # perform_eda(df)
