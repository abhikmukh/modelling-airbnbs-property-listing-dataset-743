import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot


class DataTransform:
    """
    This class is used to transform the data
    """
    @staticmethod
    def change_date_format(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        """
        This function is used to change the date format of a column
        :param df:
        :param column_name:
        :param date_format:
        :return: dataframe
        """
        df[column_name] = df[column_name].astype(object).astype('datetime64[ns]')
        df[column_name] = pd.to_datetime(df[column_name])
        return df

    @staticmethod
    def change_column_type(df: pd.DataFrame, column_name: str, new_type: str) -> pd.DataFrame:
        """
        This function is used to change the type of a column
        :param df:
        :param column_name:
        :param new_type:
        :return: dataframe
        """
        df[column_name] = df[column_name].astype(new_type)
        return df


class DataFrameTransform:
    """
    This class is used to transform the dataframe
    """

    @staticmethod
    def fill_null_values_with_mean(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        """
        This function is used to fill the null values with mean
        :param df:
        :param column_name:
        :return: dataframe
        """
        df[column_name] = df[column_name].fillna(df[column_name].mean())
        return df

    @staticmethod
    def fill_null_values_with_median(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        """
        This function is used to fill the null values with median
        :param df:
        :param column_name:
        :return: dataframe
        """
        df[column_name] = df[column_name].fillna(df[column_name].median())
        return df

    @staticmethod
    def fill_null_values_with_mode(df:pd.DataFrame, column_name: str) -> pd.DataFrame:
        """
        This function is used to fill the null values with mode
        :param df:
        :param column_name:
        :return: dataframe
        """
        df[column_name] = df[column_name].fillna(df[column_name].mode()[0])
        return df

    @staticmethod
    def fill_null_values_with_custom_value(df, column_name: str, value: float) -> pd.DataFrame:
        """
        This function is used to fill the null values with custom value
        :param df:
        :param column_name:
        :param value:
        :return: dataframe
        """
        df[column_name] = df[column_name].fillna(value)
        return df

    @staticmethod
    def fill_null_values_with_most_frequent_value(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        """
        This function is used to fill the null values with most frequent value
        :param df:
        :param column_name:
        :return: dataframe
        """
        df[column_name] = df[column_name].fillna(df[column_name].value_counts().index[0])
        return df

    @staticmethod
    def get_log_transform(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        """
        This function is used to get the log transform of a column
        :param df:
        :param column_name:
        :return: DataFrame
        """
        df[column_name] = df[column_name].map(lambda i: np.log(i) if i > 0 else 0)
        return df


class DataFrameInfo:
    """
    This class is used to get information about the dataframe
    """
    @staticmethod
    def get_df_mean(df: pd.DataFrame, column_name: str) -> None:
        """
        This function is used to get the mean of a column
        :param df:
        :param column_name:
        :return: None
        """
        print(f"mean: {df[column_name].mean()}, median: {df[column_name].median()}, mode: {df[column_name].mode()[0]}")

    @staticmethod
    def count_null_values_percentage(df: pd.DataFrame,  list_of_columns: list) -> None:
        """
        This function is used to get the percentage of null values in a column
        :param df:
        :param list_of_columns:
        :return: None
        """
        for column in list_of_columns:
            print(f"Percentage of nulls in {column}: {df[column].isnull().sum()/len(df)}")

    @staticmethod
    def get_all_null_values(df) -> pd.DataFrame:
        """
        This function is used to get the total number of null values in a column
        :return: dataframe
        """
        return df.isnull().sum()

    @staticmethod
    def get_df_unique_values(df: pd.DataFrame) -> pd.Series:
        """
        This function is used to get the unique values in a column
        :param df:
        :return: series
        """
        return df.nunique()

    @staticmethod
    def list_all_numeric_columns(df: pd.DataFrame) -> list:
        """
        This function is used to get the list of all numeric columns
        :param df:
        :return: List
        """
        return df.select_dtypes(include='number').columns.tolist()

    @staticmethod
    def list_all_categorical_columns(df: pd.DataFrame) -> list:
        """
        This function is used to get the list of all categorical columns
        :param df:
        :return: List
        """
        return df.select_dtypes(include='object').columns.tolist()

    @staticmethod
    def list_all_datetime_columns(df: pd.DataFrame) -> list:
        """
        This function is used to get the list of all datetime columns
        :param df:
        :return: List
        """
        return df.select_dtypes(include='datetime').columns.tolist()

    @staticmethod
    def list_all_columns_with_missing_values(df: pd.DataFrame) -> list:
        """
        This function is used to get the list of all columns with missing values
        :param df:
        :return: List
        """
        return df.columns[df.isnull().any()].tolist()

    @staticmethod
    def analyse_categorical_data(df: pd.DataFrame, column_name: str) -> pd.Series:
        """
        This function is used to get the value counts of a column
        :param df:
        :param column_name:
        :return: series
        """
        return df[column_name].value_counts()

    @staticmethod
    def calculate_z_score(df: pd.DataFrame, column_name: str) -> pd.Series:
        """
        This function is used to calculate the z-score of a column
        :param df:
        :param column_name:
        :return: series
        """
        return (df[column_name] - df[column_name].mean()) / df[column_name].std()

    @staticmethod
    def calculate_iqr(df: pd.DataFrame, column_name: str) -> pd.Series:
        """
        This function is used to calculate the iqr of a column
        :param df:
        :param column_name:
        :return: series
        """
        return df[column_name].quantile(0.75) - df[column_name].quantile(0.25)

    @staticmethod
    def get_column_type_of_list_of_columns(df: pd.DataFrame, list_of_columns: list) -> None:
        """
        This function is used to get the type of a column
        :param df:
        :param list_of_columns:
        :return: None
        """
        for column in list_of_columns:
            print(f"Type of {column}: {df[column].dtypes}")

    @staticmethod
    def calculate_iqr_outliers(df: pd.DataFrame, column_name: str) -> pd.Series:
        """
        This function is used to calculate the IQR outliers of a column
        :param df:
        :param column_name:
        :return: series
        """
        q1 = df[column_name].quantile(0.25)
        q3 = df[column_name].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return df[(df[column_name] < lower_bound) | (df[column_name] > upper_bound)][column_name]


class DataFrameVisualize:
    """
    This class is used to visualize the dataframe
    """

    @staticmethod
    def plot_histogram(df: pd.DataFrame, column_name: str) -> None:
        """
        This function is used to plot a histogram
        :param df:
        :param column_name:
        :return:
        """
        return sns.histplot(data=df, x=column_name, kde=True)


    @staticmethod
    def plot_boxplot(df: pd.DataFrame, column_name: str) -> None:
        """
        This function is used to plot a boxplot
        :param df:
        :param column_name:
        :return:
        """
        return df.boxplot(column=column_name)

    @staticmethod
    def plot_scatter(df: pd.DataFrame, x, y):
        """
        This function is used to plot a scatter plot
        :param df:
        :param x:
        :param y:
        :return:
        """
        return df.plot.scatter(x=x, y=y)

    @staticmethod
    def plot_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
        """
        This function is used to plot a correlation matrix
        :param df:
        :return:
        """
        return df.corr()

    @staticmethod
    def plot_heatmap(df: pd.DataFrame):
        """
        This function is used to plot a heatmap
        :param df:
        :return:
        """
        plt.figure(figsize=(10, 8))
        return sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

    @staticmethod
    def plot_qqplot(df: pd.DataFrame, column_name: str):
        """
        This function is used to plot a qqplot
        :param df:
        :param column_name:
        :return:
        """
        return qqplot(df[column_name], scale=1, line='q', fit=True).show()

    @staticmethod
    def plot_prob_distribution(df: pd.DataFrame, column_name: str):
        """
        This function is used to plot a probability distribution
        :param df:
        :param column_name:
        :return:
        """
        probs = df[column_name].value_counts(normalize=True)

        # Create bar plot
        plt.xlabel('Values')
        plt.ylabel('Probability')
        plt.title(column_name)
        return sns.barplot(y=probs.values, x=probs.index)

    @staticmethod
    def box_plot_with_scatter_points(df: pd.DataFrame, column_name: str):
        """
        This function is used to plot a box plot with scatter points
        :param df:
        :param column_name:
        :return:
        """
        sns.boxplot(y=df[column_name], color='lightgreen', showfliers=True)
        sns.swarmplot(y=df[column_name], color='black', size=5)
        plt.title(f'Box plot with scatter points {column_name}')

    @staticmethod
    def plot_facet_grids(df: pd.DataFrame, column_name: str, target: str):
        """
        This function is used to plot facet grids
        :param df:
        :param column_name:
        :param target:
        :return:
        """
        plotting_df = df.loc[:, [column_name, target]]
        g = sns.FacetGrid(plotting_df, col=target)
        g.map(sns.histplot, column_name)
        g.set_xticklabels(rotation=90)
















