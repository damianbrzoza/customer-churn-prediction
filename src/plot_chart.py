import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from heatmap import corrplot


def plot_correlations(df: pd.DataFrame) -> None:
    """
    Plot summary correlations between our target and numerical columns.

    :param df: data
    :type df: pd.DataFrame
    """

    plt.figure(figsize=(10, 10))
    corrplot(df.corr())
    plt.tight_layout()
    plt.gcf().subplots_adjust(bottom=0.15, left=0.2)
    plt.show()


def plot_target_proportion(df: pd.DataFrame) -> None:
    """
    Show proportion between target classes.

    :param df: data
    :type df: pd.DataFrame
    """
    plt.figure(figsize=(20, 10))
    df["Attrition_Flag"].hist()
    plt.show()


def plot_categorical_proportion(df: pd.DataFrame, column_name: str) -> None:
    """
    Try to find out some factors that make attrition more likely.

    :param df: data
    :type df: pd.DataFrame
    :param column_name: name of the column to process
    :type column_name: str
    """

    # fmt: off
    categorical = (
        df
        [['Attrition_Flag', column_name, 'CLIENTNUM']]
        .groupby([column_name, 'Attrition_Flag'])
        .count()
        .reset_index(level=1)
        .pivot_table('CLIENTNUM', column_name, 'Attrition_Flag')
    )

    proportions = (
        categorical
        .apply(
            lambda row: {'Attrited Customer': row['Attrited Customer']/sum(row),
                        'Existing Customer': row['Existing Customer']/sum(row)},
            axis=1)
        .apply(pd.Series)
    )

    # fmt: on

    proportions.plot(kind="barh", stacked=True, colormap="tab10", figsize=(10, 6))

    plt.axvline(
        x=np.mean(proportions["Attrited Customer"]), color="b", linestyle="--", label="mean"
    )

    plt.legend(loc="lower left", ncol=3)
    plt.ylabel(f"Attrited Customer/{column_name}")
    plt.xlabel("Proportion")

    plt.show()


def plot_boxplot_with_category(df: pd.DataFrame, column_name: str) -> None:
    """
    Show boxplot for given column with split by our target category.

    :param df: data
    :type df: pd.DataFrame
    :param column_name: name of the column to process
    :type column_name: str
    """
    sns.boxplot(
        x=column_name,
        y="Attrition_Flag",
        data=df[[column_name, "Attrition_Flag"]],
    )
    plt.show()


def plot_quant_histogram(df: pd.DataFrame, column_name: str) -> None:
    """
    Show histogram for given column.

    :param df: data
    :type df: pd.DataFrame
    :param column_name: name of the column to process
    :type column_name: str
    """
    sns.histplot(data=df, x=column_name, bins=10)
    plt.show()
