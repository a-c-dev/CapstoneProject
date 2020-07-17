import pandas as pd
import numpy as np
import warnings
import os
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


class DataManager:
    def __init__(self):
        self.X_train = pd.DataFrame()
        self.y_train = []
        self.X_test = pd.DataFrame()
        self.y_test = []
        self.label_column = 'user-definedlabeln'

    @staticmethod
    def impute_dataset(df):
        """
        This function imputes a dataset, replacing NaNs values with K-Nearest Neighbours median value
        :param df: pandas dataframe to be imputed
        :return: imputed pandas dataframe
        """
        imputer = KNNImputer(missing_values=np.nan)
        ds_idxs = df.index
        ds_cols = df.columns
        df = pd.DataFrame(imputer.fit_transform(df), index=ds_idxs, columns=ds_cols)
        return df

    @staticmethod
    def remove_outliers(EEG_data_df, outlier_subjects):
        """
        This function filters the dataframe removing samples with specific SubcjectIDs
        :param EEG_data_df: EEG_data brainwaves pandas dataframe
        :param outlier_subjects: list of SubjectIDs to filter
        :return: filtered pandas dataframe
        """
        # for each outlier subject: removing samples correlated to it
        for outlier in outlier_subjects:
            EEG_data_df = EEG_data_df[EEG_data_df['SubjectID'] != outlier]
        # returning filtered df
        return EEG_data_df

    @staticmethod
    def preprocess_data(EEG_data_df, outlier_subjects):
        """
        This function pre-processes dataset: impute it, removes outliers and drops useless columns
        :param outlier_subjects: list of SubjectIDs to filter
        :param EEG_data_df: EEG_data brainwaves pandas dataframe
        :return: processed pandas dataframe
        """
        EEG_data_df = DataManager.impute_dataset(EEG_data_df)
        EEG_data_df = DataManager.remove_outliers(EEG_data_df, outlier_subjects)
        # Take columns to drop, if they exists in the df: avoid dataframe column key error
        columns_to_drop = list(filter(lambda x: x in ['VideoID', 'predefinedlabel', 'SubjectID'],
                                      list(EEG_data_df.columns)
                                      )
                               )
        if len(columns_to_drop) > 0:
            EEG_data_df.drop(columns=columns_to_drop, inplace=True)

        return EEG_data_df

    @staticmethod
    def load_data(csv_file_path, preprocess: object = True, outlier_subjects=[]):
        """
        This function loads a csv_file into a pandas dataframe, with optional data preprocessing
        :param csv_file_path: path of the csv file
        :param preprocess: True or False, based on the preprocessing data needs
        :param outlier_subjects: in case of preprocessing, SubjectIDs outliers
        :return: loaded (and preprocessed) pandas dataframe
        """
        df = pd.read_csv(csv_file_path)
        df = DataManager.preprocess_data(df, outlier_subjects)
        return df

    @staticmethod
    def save_df(df, csv_file_path):
        """
        This function saves a df on a csv file
        :param csv_file_path: saving file path
        :param df: dataframe to save
        :return: none
        """
        assert isinstance(df, pd.DataFrame)
        df.to_csv(csv_file_path)

    def split_X_y(self, EEG_brainwave_df):
        """
        This method split the EEG brainwave dataset into features and labels
        :param EEG_brainwave_df: dataframe
        :return: X, y (features, labels)
        """
        X = EEG_brainwave_df.drop(columns=[self.label_column])
        y = EEG_brainwave_df[self.label_column]
        return X, y

    def format_topredict_df(self, topredict_brainwave_df):
        """
        This method assures that in the df there is no the label column
        :param topredict_brainwave_df: df with data on which predicting labels
        :return: df
        """
        if self.label_column in topredict_brainwave_df.columns:
            topredict_brainwave_df.drop(columns=[self.label_column], inplace=True)

        return topredict_brainwave_df

    def create_train_test_split(self, EEG_brainwave_df, train_file_path="data/train.csv",
                                test_file_path="data/test.csv"):
        """
        This method splits a dataframe into training and testing data.
        This split is saved in the class
        :param EEG_brainwave_df:  dataset to shuffle and split, preprocessed
        :param test_file_path: file path on which saving testing data
        :param train_file_path: file path on which saving training data
        :return: none
        """
        X, y = self.split_X_y(EEG_brainwave_df)

        # Split the 'features' and 'y' data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,
                                                                                y,
                                                                                test_size=0.2,
                                                                                random_state=0)
        # Save file train.py and test.py
        y_cols = [self.label_column]
        X_cols = list(filter(lambda x: not x == self.label_column, list(EEG_brainwave_df.columns)))

        pd.concat([pd.DataFrame(self.y_train, columns=y_cols),
                   pd.DataFrame(self.X_train, columns=X_cols)
                   ],
                  axis=1
                  ).to_csv(train_file_path,
                           header=True,
                           index=False)

        pd.concat([pd.DataFrame(self.y_test, columns=y_cols),
                   pd.DataFrame(self.X_test, columns=X_cols)
                   ],
                  axis=1
                  ).to_csv(test_file_path,
                           header=True,
                           index=False)

    @property
    def training_data(self, train_file_path="data/train.csv"):
        """
        This property returns training data. If the class has no training data loaded, tries to load it from file
        :return: X_train, y_train
        """
        if len(self.y_train) == 0:
            if os.path.exists(train_file_path):
                self.X_train, self.y_train = self.split_X_y(pd.read_csv(train_file_path))

        return self.X_train, self.y_train

    @property
    def testing_data(self, test_file_path="data/test.csv"):
        """
        This property returns testing data. If the class has no testing data loaded, tries to load it from file
        :return: X_test, y_test
        """
        if len(self.y_train) == 0:
            if os.path.exists(test_file_path):
                self.X_test, self.y_test = self.split_X_y(pd.read_csv(test_file_path))

        return self.X_test, self.y_test

    def save_predictions(self, topredict_brainwave_df, predictions, file):
        """
        This method saves on a file the result of the prediction: creates the original preprocessed df with addictional label column
        :param topredict_brainwave_df: df with data on which predicting labels
        :param predictions: predictions labels
        :param file: file where to save df
        :return: df
        """
        pd.concat([pd.DataFrame(predictions, columns=[self.label_column]), topredict_brainwave_df],
                  axis=1
                  ).to_csv(file,
                           header=True,
                           index=False)
