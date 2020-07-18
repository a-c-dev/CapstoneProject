from sklearn.metrics import fbeta_score, accuracy_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import numpy as np
from joblib import dump
from time import gmtime, strftime
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

EGG_brainwave_df = dm.load_data(os.path.join(params.data_dir(), "EEG_data.csv"),
                                preprocess=True,
                                outlier_subjects=params.outliers_students())


dm.create_train_test_split(EGG_brainwave_df,
                           train_file_path=os.path.join(params.data_dir(), "train.csv"),
                           test_file_path=os.path.join(params.data_dir(), "test.csv"))
# --------------------------------------------------------------------------------------Define the supervised classifier
print("Define the supervised classifier")
# Supervised classifier instance
learner = xgb.XGBClassifier(seed=0)
learning_params = {'base_score': list(np.arange(0.2, 0.5, 0.1)),
                   'n_estimators': [10, 40, 60, 100, 120, 130, 140],
                   'objective': ['binary:logistic']}
# -----------------------------------------------------------------------------------------Train and Tune the classifier
print("Train and Tune the classifier")
grid_obj = GridSearchCV(learner,
                        learning_params,
                        scoring=make_scorer(fbeta_score, beta=params.beta()))
# Fit the grid search object to the training data and find the optimal parameters
grid_fit = grid_obj.fit(dm.training_data[0], dm.training_data[1])
# Get the nest estimator
learner = grid_fit.best_estimator_
# ---------------------------------------------------------------------------------------------Save classifier for later
print("Save classifier for later")
# Save trained model on a file for later usage
dump(learner, os.path.join(params.models_dir(),
                           learner.__class__.__name__ + strftime("%Y-%m-%d-%H-%M-%S", gmtime()) + ".joblib"))
# ---------------------------------------------------------------------------------------------------------Print metrics
print("Print metrics")
y_test_pred = [round(value) for value in learner.predict(dm.testing_data[0])]
print(f"Accuracy Score: {accuracy_score(dm.testing_data[1], y_test_pred)}")
print(f"FBeta Score: {fbeta_score(dm.testing_data[1], y_test_pred, beta=params.beta())}")
print(f"Precision Score: {precision_score(dm.testing_data[1], y_test_pred)}")
print(f"Recall Score: {recall_score(dm.testing_data[1], y_test_pred)}")
print("")
print("")
print("")
v = input("insert a value to continue...")