from joblib import load
import os

from AppParametersManagement import AppParametersLoader
from DataManagement import DataManager

# ---------------------------------------------------------------------------------------------Load and instance Helpers
# DataManger instance
dm = DataManager()
# App parameters loading and parsing
params = AppParametersLoader()
params.print_all()

# -------------------------------------------------------------------------------------------------Load and prepare data
print("Load and prepare data")
# Dataset Loading
# Creating train_test_split subsets and saving them
if not os.path.exists(params.data_dir()):
    os.makedirs(params.data_dir())

topredict_brainwave_df = dm.load_data(os.path.join(params.data_dir(), params.topredict_file_name()),
                                      preprocess=True,  # preprocess data if it has not been done yet
                                      outlier_subjects=[])  # no outliers in data to predict

topredict_brainwave_df = dm.format_topredict_df(topredict_brainwave_df)

# --------------------------------------------------------------------------------------------------------Load predictor
predictor = load(os.path.join(params.models_dir(), params.predictor_file_name()))
# ------------------------------------------------------------------------------------------------------make predictions
predictions = [round(value) for value in predictor.predict(topredict_brainwave_df)]
# ------------------------------------------------------------------------------------------------------save predictions
pred_path = os.path.join(params.data_dir(),
                         "predicted_" + params.topredict_file_name())
dm.save_predictions(topredict_brainwave_df,
                    predictions,
                    pred_path
                    )

print(f"Predictions saved on file {pred_path}")
v = input("insert a value to continue...")
