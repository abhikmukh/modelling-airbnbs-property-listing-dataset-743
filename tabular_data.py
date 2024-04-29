import pandas as pd
from ast import literal_eval

import modelling_utils
from eda_utils import DataFrameInfo


class CleanTabularData:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def remove_rows_with_missing_ratings(self) -> pd.DataFrame:
        for col in self.df.columns:
            if "rating" in col.lower():
                self.df.dropna(subset=col, inplace=True)
                return self.df

    @staticmethod
    def _parses_description(description: str) -> str:
        description_to_list = literal_eval(description)
        description_to_list.pop(0)
        while '' in description_to_list:
            description_to_list.remove('')
        clean_description = ' '.join(description_to_list)
        clean_description.replace('\n', ' ')
        return clean_description

    def combine_description_strings(self) -> pd.DataFrame:
        self.df.dropna(subset=['Description'], inplace=True)
        self.df['Description'] = self.df['Description'].apply(CleanTabularData._parses_description)
        return self.df

    def set_default_feature_values(self) -> pd.DataFrame:
        list_of_columns = ["bedrooms", "bathrooms", "beds", "guests"]
        for col in list_of_columns:
            self.df[col].fillna(1.0, inplace=True)
        return self.df

    def clean_category_column(self) -> pd.DataFrame:
        self.df['Category'] = \
            self.df['Category'].str.replace(r'Amazing pools,Stunning Cotswolds Water Park, sleeps 6 with pool',
                                            'Amazing pools', regex=True)
        return self.df

    def drop_outliers(self):
        dataframe_info = DataFrameInfo()
        numeric_df = self.df.select_dtypes(include=['float64', 'int64'])
        column_list = numeric_df.columns[numeric_df.skew() > 3].tolist()
        for column in column_list:
            outliers = dataframe_info.calculate_iqr_outliers(self.df, column)
            self.df = self.df.drop(outliers.index)
        return self.df


def load_airbnb_data(data_df, label_column):

    features = data_df.drop(columns=[label_column])  # Drop the label column to get features
    labels = data_df[label_column]  # Extract the label column

    return features, labels


if __name__ == "__main__":
    df = pd.read_csv("data/listing.csv")
    cleaner_df = CleanTabularData(df)

    cleaner_df.remove_rows_with_missing_ratings()
    cleaner_df.combine_description_strings()

    cleaner_df.set_default_feature_values()
    cleaner_df.drop_outliers()

    cleaner_df.df.to_csv("data/cleaned_data.csv", index=False)


