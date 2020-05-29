import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
import numpy as np


def calculate_model_score(y_true, y_pred, metric):
    metric_dict = {'accuracy': accuracy_score(y_true, y_pred), 'f1': f1_score(y_true, y_pred, pos_label=1),
                   'f1_micro': f1_score(y_true, y_pred, average='micro'),
                   'f1_macro': f1_score(y_true, y_pred, average='macro'),
                   'precision': precision_score(y_true, y_pred), 'recall': recall_score(y_true, y_pred),
                   'roc_auc': roc_auc_score(y_true, y_pred)}
    model_score = metric_dict[metric]
    return model_score


def calculate_multiple_model_scores(y_true, y_pred, metric_list):
    metric_dict = {'accuracy': accuracy_score(y_true, y_pred), 'f1': f1_score(y_true, y_pred, pos_label=1),
                   'f1_micro': f1_score(y_true, y_pred, average='micro'),
                   'f1_macro': f1_score(y_true, y_pred, average='macro'),
                   'precision': precision_score(y_true, y_pred), 'recall': recall_score(y_true, y_pred),
                   'roc_auc': roc_auc_score(y_true, y_pred)}
    model_scores = {}
    for metric in metric_list:
        model_scores[metric] = metric_dict[metric]
    return model_scores


def metric_per_contamination_level(y_true, model_predictions, metric='f1'):
    columns_ = ['model', 'contamination_level', metric]
    model_stats_df = pd.DataFrame(columns=columns_)

    i = 0
    for model_name, predictions in model_predictions.items():
        for contamination_level, cont_predictions in predictions.items():
            score = calculate_model_score(y_true, cont_predictions, metric=metric)
            model_stats_df.loc[i] = [model_name, contamination_level, score]

            i += 1

    return model_stats_df


def calc_model_performance_over_time(X_test_df, y_test,
                                     contamination_levels_subset, scoring='f1', aggregated_timestamp_column='time_step',
                                     **model_predictions):
    first_test_time_step = np.sort(X_test_df[aggregated_timestamp_column].unique())[0]
    last_time_step = np.sort(X_test_df[aggregated_timestamp_column].unique())[-1]

    model_scores_dict = {scoring: {key: {} for key in contamination_levels_subset}}
    for contamination_level in contamination_levels_subset:
        for model_name, predictions in model_predictions.items():
            model_scores = []
            y_pred_ = predictions[contamination_level]
            for time_step in range(first_test_time_step, last_time_step + 1):
                time_step_idx = np.flatnonzero(X_test_df[aggregated_timestamp_column] == time_step)
                y_true = y_test[X_test_df[aggregated_timestamp_column] == time_step]
                y_pred = [y_pred_[i] for i in time_step_idx]

                model_scores.append(calculate_model_score(y_true.astype('int'), y_pred, scoring))
            model_scores_dict[scoring][contamination_level][model_name] = model_scores
    return model_scores_dict

def calc_average_score(y_test, y_preds, scoring = 'f1'):
    all_model_scores = []
    for y_pred in y_preds:
        model_score = calculate_model_score(y_test.astype('int'), y_pred, scoring)
        all_model_scores.append(model_score)

    avg = np.mean(all_model_scores)

    return avg


def calc_average_score_and_std_per_timestep(X_test_df, y_test, y_preds, aggregated_timestamp_column='time_step', scoring= 'f1'):
    last_train_time_step = min(X_test_df['time_step']) - 1
    last_time_step = max(X_test_df['time_step'])
    all_model_scores = []
    for y_pred in y_preds:
        model_scores = []
        for time_step in range(last_train_time_step + 1, last_time_step + 1):
            time_step_idx = np.flatnonzero(X_test_df[aggregated_timestamp_column] == time_step)
            y_true_ts = y_test.iloc[time_step_idx]
            y_pred_ts = [y_pred[i] for i in time_step_idx]
            model_scores.append(calculate_model_score(y_true_ts.astype('int'), y_pred_ts, scoring))
        all_model_scores.append(model_scores)

    avg_f1 = np.array([np.mean([f1_scores[i] for f1_scores in all_model_scores]) for i in range(15)])
    std = np.array([np.std([f1_scores[i] for f1_scores in all_model_scores]) for i in range(15)])

    return avg_f1, std