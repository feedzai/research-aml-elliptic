import os
import sys

ROOT_DIR = os.getcwd()
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

import numpy as np
from experiments.general_functions.elliptic_data_preprocessing import run_elliptic_preprocessing_pipeline
from reaml.models import batch_pyod_per_contamination_level
from reaml.model_performance import metric_per_contamination_level
from pyod.models.pca import PCA
from pyod.models.lof import LOF
from pyod.models.cblof import CBLOF
from pyod.models.knn import KNN
from pyod.models.abod import ABOD
from pyod.models.ocsvm import OCSVM
from pyod.models.iforest import IForest
from reaml.models import supervised_model_per_contamination_level
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Import Elliptic data set and set variables
last_time_step = 49
last_train_time_step = 34
only_labeled = True

X_train_df, X_test_df, y_train, y_test = run_elliptic_preprocessing_pipeline(last_train_time_step=last_train_time_step,
                                                                             last_time_step=last_time_step,
                                                                             only_labeled=only_labeled)

# Run anomaly detection methods and save the results. In this case return_model_predictions = True.

contamination_levels = np.arange(0, 1.01, 0.05)

model_dict = {'PCA': PCA(), 'LOF': LOF(), 'CBLOF': CBLOF(), 'IF': IForest(), 'KNN': KNN(), 'ABOD': ABOD(),
              'OCSVM': OCSVM()}

model_predictions, model_predicted_scores = batch_pyod_per_contamination_level(X_train_df=X_train_df, X_test_df=X_test_df, y_train=y_train,
                                                                                   contamination_levels=contamination_levels, model_dict=model_dict)

random_forest = RandomForestClassifier()

predictions_at_contamination_levels, _ = supervised_model_per_contamination_level(X_train_df, X_test_df, y_train,
                                                                                  contamination_levels, random_forest)

model_predictions['Random Forest'] = predictions_at_contamination_levels


model_stats_df = metric_per_contamination_level(y_true=y_test, metric='f1', model_predictions=model_predictions)

f1_per_contamination_level = model_stats_df.pivot(index='model', columns='contamination_level', values='f1')
f1_per_contamination_level = f1_per_contamination_level.round(2)
f1_per_contamination_level.columns = np.round(f1_per_contamination_level.columns, 2)

f1_per_contamination_level_reduced = f1_per_contamination_level.iloc[:, 1:5]
f1_per_contamination_level_reduced.sort_values(0.2, ascending=False)

f1_per_contamination_level_reduced.to_csv(os.path.join(ROOT_DIR, 'output/table_2_illicit_f1_per_contamination_level.csv'))
