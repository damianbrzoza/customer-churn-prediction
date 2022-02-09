"""Some helper functions to plotting."""
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from heatmap import corrplot
from sklearn.base import ClassifierMixin
from sklearn.metrics import RocCurveDisplay, classification_report


def plot_correlations(df: pd.DataFrame) -> None:
    """
    Plot summary correlations between our target and numerical columns.

    :param df: data
    :type df: pd.DataFrame
    """
    plt.figure(figsize=(10, 10), num="Correlations")
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
    plt.figure(figsize=(20, 10), num="target_proportion")
    plt.title("target_proportion")
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
        .pivot_table(values='CLIENTNUM', index=column_name, columns='Attrition_Flag')
    )

    # fmt: on

    proportions = categorical.apply(
        lambda row: {
            "Attrited Customer": row["Attrited Customer"] / sum(row),
            "Existing Customer": row["Existing Customer"] / sum(row),
        },
        axis=1,
    ).apply(pd.Series)

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


def compare_roc_curve(
    list_of_models: List[ClassifierMixin], X_test: pd.DataFrame, y_test: pd.Series
) -> None:
    """
    Compare roc curve for given models.

    :param list_of_models: list of classifiers
    :type list_of_models: List[ClassifierMixin]
    :param X_test: test features
    :type X_test: pd.DataFrame
    :param y_test: test labels
    :type y_test: pd.Series
    """
    first_plot = RocCurveDisplay.from_estimator(
        estimator=list_of_models[0], X=X_test, y=y_test, alpha=0.8
    )
    if len(list_of_models) > 1:
        for model in list_of_models[1:]:
            RocCurveDisplay.from_estimator(
                estimator=model, X=X_test, y=y_test, ax=first_plot.ax_, alpha=0.8
            )

    plt.show()


def show_features_importance(model: ClassifierMixin, X_test: pd.DataFrame) -> None:
    """
    Show features importance for given tree-based model.

    :param model: tree based model
    :type model: ClassifierMixin
    :param X_test: test features
    :type X_test: pd.DataFrame
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test[:100])
    shap.summary_plot(shap_values, X_test[:100], plot_type="bar")
    plt.show()

    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_test.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5), num="Feature Importance")

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel("Importance")

    # Add bars
    plt.bar(range(X_test.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_test.shape[1]), names, rotation=90)
    plt.show()


def plot_classifier_summary_as_image(
    model: ClassifierMixin,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    """
    Make classifier report in image form.

    :param model: classifier
    :type model: ClassifierMixin
    :param X_train: train features
    :type X_train: pd.DataFrame
    :param X_test: test features
    :type X_test: pd.DataFrame
    :param y_train: train labels
    :type y_train: pd.Series
    :param y_test: test labels
    :type y_test: pd.Series
    """
    plt.rc("figure", figsize=(5, 5))
    y_train_preds = model.predict(X_train)
    y_test_preds = model.predict(X_test)
    plt.text(
        0.01,
        1.25,
        s=str("Logistic Regression Train"),
        fontdict={"fontsize": 10},
        fontproperties="monospace",
    )
    plt.text(
        0.01,
        0.05,
        s=str(classification_report(y_train, y_train_preds)),
        fontdict={"fontsize": 10},
        fontproperties="monospace",
    )  # approach improved by OP -> monospace!
    plt.text(
        0.01,
        0.6,
        s=str("Logistic Regression Test"),
        fontdict={"fontsize": 10},
        fontproperties="monospace",
    )
    plt.text(
        0.01,
        0.7,
        s=str(classification_report(y_test, y_test_preds)),
        fontdict={"fontsize": 10},
        fontproperties="monospace",
    )  # approach improved by OP -> monospace!
    plt.axis("off")
    plt.show()
