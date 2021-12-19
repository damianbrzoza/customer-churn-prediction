"""Module enable customer churn analysis."""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from heatmap import corrplot

def import_data(pth: str, verbose = False) -> pd.DataFrame:
    """
    Returns dataframe for the csv found at pth.

    :param pth: a path to the csv
    :type pth: str
    :return: pandas dataframe
    :rtype: pd.DataFrame
    """

    df = pd.read_csv(pth, index_col=0)

    cat_columns = [
    'Attrition_Flag',
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'                
    ] 

    df[cat_columns] = df[cat_columns].astype('category')

    if verbose:
        print(df.head())   
        print(df.info())    
    
    return df

def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:  
        df: pandas dataframe

    output:
        None
    '''
    df.head()
    df.shape
    df.isnull().sum()
    df.describe()

    cat_columns = [
    'Attrition_Flag',
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'                
    ]

    quant_columns = [
        'Customer_Age',
        'Dependent_count', 
        'Months_on_book',
        'Total_Relationship_Count', 
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon', 
        'Credit_Limit', 
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy', 
        'Total_Amt_Chng_Q4_Q1', 
        'Total_Trans_Amt',
        'Total_Trans_Ct', 
        'Total_Ct_Chng_Q4_Q1', 
        'Avg_Utilization_Ratio'
    ]

    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1).astype('int64')

    # plt.figure(figsize=(20,10)) 
    # df['Customer_Age'].hist()

    # plt.figure(figsize=(20,10)) 
    # df.Marital_Status.value_counts('normalize').plot(kind='bar');

    # plt.figure(figsize=(20,10)) 
    # sns.distplot(df['Total_Trans_Ct']);

    # Get some info about correlation between our target and numerical values

    # df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1).astype('int64')

    # plt.figure(figsize=(10,10)) 
    # corrplot(df.corr())
    # plt.tight_layout()
    # plt.gcf().subplots_adjust(bottom=0.15, left=0.2)
    # plt.show()

    # plt.figure(figsize=(20,10)) 
    # df['Attrition_Flag'].hist()
    # plt.show()

    genders = (df
               [['Attrition_Flag', 'Gender', 'CLIENTNUM']]
               .groupby(['Gender', 'Attrition_Flag'])
               .count()
               .reset_index(level=1)
               .pivot_table('CLIENTNUM', 'Gender', 'Attrition_Flag')
               )

    print(genders)
    proportions = (genders
                   .apply(lambda row: {'Attrited Customer': row['Attrited Customer']/sum(row), 
                                       'Existing Customer':row['Existing Customer']/sum(row)}, 
                                       axis=1)
                   .apply(pd.Series))

    proportions.plot(kind='barh', 
                stacked=True, 
                colormap='tab10', 
                figsize=(10, 6))

    plt.legend(loc="lower left", ncol=2)
    plt.ylabel("Attrited Customer/Gender")
    plt.xlabel("Proportion")
    plt.show()


    # for n, x in enumerate([*cross_tab.index.values]):
    #     for (proportion, count, y_loc) in zip(cross_tab_prop.loc[x],
    #                                         cross_tab.loc[x],
    #                                         cross_tab_prop.loc[x].cumsum()):
                    
    #         plt.text(x=(y_loc - proportion) + (proportion / 2),
    #                 y=n - 0.11,
    #                 s=f'{count}\n({np.round(proportion * 100, 1)}%)', 
    #                 color="black",
    #                 fontsize=12,
    #                 fontweight="bold")
    
    

def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
        df: pandas dataframe
        category_lst: list of columns that contain categorical features
        response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
        df: pandas dataframe with new columns for
    '''
    pass


def perform_feature_engineering(df, response):
    '''
    input:
        df: pandas dataframe
        response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
        X_train: X training data
        X_test: X testing data
        y_train: y training data
        y_test: y testing data
    '''

def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
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
    '''
    pass


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
        model: model object containing feature_importances_
        X_data: pandas dataframe of X values
        output_pth: path to store the figure

    output:
        None
    '''
    pass

def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
        X_train: X training data
        X_test: X testing data
        y_train: y training data
        y_test: y testing data
    output:
        None
    '''
    pass

if __name__ == '__main__':
    df = import_data("./data/bank_data.csv", verbose=False)
    perform_eda(df)