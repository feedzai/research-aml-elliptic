import os
import sys

ROOT_DIR = os.getcwd()
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

from pyod.models.iforest import IForest
from experiments.general_functions.elliptic_data_preprocessing import load_elliptic_data, setup_train_test_idx, \
    train_test_split
from reaml.models import batch_pyod_per_contamination_level
import umap
from experiments.general_functions.plotting import plot_UMAP_projection
import pandas as pd

last_train_time_step = 34
last_time_step = 49
only_labeled = True
import warnings

warnings.filterwarnings('ignore')

X, y = load_elliptic_data(only_labeled=only_labeled)
train_test_idx = setup_train_test_idx(X, last_train_time_step, last_time_step)
X_train_df, X_test_df, y_train, y_test = train_test_split(X, y, train_test_idx)

# set contamination_levels to predict
contamination_levels = [0.1]

model_predictions, model_predicted_scores = batch_pyod_per_contamination_level(X_train_df, X_test_df, y_train,
                                                                               contamination_levels, predict_on='test',
                                                                               model_dict={'IF': IForest()})

# Define the data subsets to plot
X_subset = X_test_df
y_true_subset = y_test

reducer = umap.UMAP(n_components=2, min_dist=0.1, n_neighbors=70)
embedding = reducer.fit_transform(X_subset)

embedding_df = pd.DataFrame(embedding, columns=('dim_0', 'dim_1'))
embedding_df['class'] = y_true_subset.tolist()
embedding_df['class'] = embedding_df['class'].replace({1: 'Illicit', 0: 'Licit'})

model = 'IF'
contamination_level = 0.1
embedding_df['prediction'] = ['Illicit' if pred == 1 else 'Licit' for pred in
                              model_predictions[model][contamination_level]]

# Plot UMAP projections
plot_UMAP_projection(embedding_df=embedding_df, hue_on='prediction', fontsize=19, labelsize=22,
                     palette=['cadetblue', 'coral'],
                     savefig_path=os.path.join(ROOT_DIR, 'output/figure_3_umap_predicted_label.png'))

plot_UMAP_projection(embedding_df=embedding_df, hue_on='class', fontsize=19, labelsize=22,
                     palette=['cadetblue', 'coral'],
                     savefig_path=os.path.join(ROOT_DIR, 'output/figure_4_umap_true_label.png'))


