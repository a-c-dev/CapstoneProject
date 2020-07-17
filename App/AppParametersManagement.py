import argparse
from pathlib import Path


class AppParametersLoader:
    def __init__(self):
        # ArgumentParser init
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--models_dir',
                                 action='store',
                                 default='models',
                                 type=str,
                                 help='directory for the model checkpoint')

        self.parser.add_argument('--outliers_students',
                                 action='store',
                                 default='2,6',
                                 type=str,
                                 help='list of outliers students')

        self.parser.add_argument('--data_dir',
                                 action='store',
                                 default='..\data',
                                 type=str,
                                 help='directory for data')

        self.parser.add_argument('--beta',
                                 action='store',
                                 default=0.5,
                                 type=float,
                                 help='beta parameter for FScore')

        self.parser.add_argument('--predictor_file_name',
                                 action='store',
                                 default='model.joblib',
                                 type=str,
                                 help='the file dump of the trained model to use')

        self.parser.add_argument('--topredict_file_name',
                                 action='store',
                                 default='to_predict.csv',
                                 type=str,
                                 help='the file name with data to predict')

        # parsing arguments
        self.args = self.parser.parse_args()

    @staticmethod
    def get_by_line(param_name):
        """
        This function gets a parameter by prompt line
        :param param_name:
        :return: None
        """
        return input(
            f"Parameter {param_name} is undefined in the command line and its default value is incorrect in the current context. Please, insert its value manually: ")

    def models_dir(self):
        """
        This method returns models_dir parameter
        :return: models_dir
        """
        if not Path(self.args.models_dir).is_dir():
            self.args.models_dir = self.get_by_line('models_dir')
        return self.args.models_dir

    def data_dir(self):
        """
        This method returns data_dir parameter
        :return: models_dir
        """
        if not Path(self.args.data_dir).is_dir():
            self.args.data_dir = self.get_by_line('data_dir')
        return self.args.data_dir

    def outliers_students(self):
        """
        This method returns outliers_students parameter
        :return: outliers_students list
        """
        return str(self.args.outliers_students).split(',')

    def beta(self):
        """
        This method returns beta parameter
        :return: beta
        """
        return self.args.beta

    def predictor_file_name(self):
        """
        This method returns predictor_file_name parameter
        :return: beta
        """
        return self.args.predictor_file_name

    def topredict_file_name(self):
        """
        This method returns topredict_file_name parameter
        :return: beta
        """
        return self.args.topredict_file_name



    def print_all(self):
        """
        This method prints parameters
        :return:
        """
        print(f"Parameters: "
              f"models_dir: {self.models_dir()},",
              f"outliers_students: {self.outliers_students()},",
              f"data_dir: {self.data_dir()},",
              f"beta: {self.beta()}, ",
              f"beta: {self.predictor_file_name()}, "
              )
