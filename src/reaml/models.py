import numpy as np
import random

def supervised_model_cv_fit_predict(X_train_df, y_train, X_test_df, model, runs=5):
    y_preds = []

    for i in range(runs):
        random.seed(i)
        model.fit(X_train_df, y_train)
        y_pred = model.predict(X_test_df)
        y_preds.append(y_pred)

    return y_preds


def pyod_fit_predict(X_train_df, X_test_df, y_train, model, semisupervised=False):
    if semisupervised == True:
        X_train_df = X_train_df[y_train == 0]
    model.fit(X_train_df)
    y_pred = model.predict(X_test_df)

    return y_pred


def pyod_predict_scores(X_train_df, X_test_df, y_train, model, predict_on='test', semisupervised=False):
    if semisupervised == True:
        X_train_df = X_train_df[y_train == 0]

    model.fit(X_train_df)

    if predict_on == 'test':
        predicted_scores = model.decision_function(X_test_df)
    elif predict_on == 'train':
        predicted_scores = model.decision_scores_

    return predicted_scores


def contamination_to_threshold(contamination, prediction_scores):
    prediction_threshold = np.quantile(prediction_scores, 1 - contamination)
    return prediction_threshold


def predict_based_on_threshold(threshold, predicted_scores, formula='greater_or_equal'):
    if formula == 'greater_or_equal':
        y_pred = [1 if score >= threshold else 0 for score in predicted_scores]
    if formula == 'greater':
        y_pred = [1 if score > threshold else 0 for score in predicted_scores]
    return y_pred


def get_thresholds_for_all_contamination_levels(contamination_levels, predicted_scores):
    thresholds = {}
    for contamination in contamination_levels:
        thresholds[contamination] = contamination_to_threshold(contamination, predicted_scores)
    return thresholds


def pyod_per_contamination_level(X_train_df, X_test_df, y_train, contamination_levels, model, predict_on='test',
                                 semisupervised=False):
    """ Accepts a list of contamination levels and a single model"""
    predicted_scores = pyod_predict_scores(X_train_df, X_test_df, y_train, model, predict_on, semisupervised)
    thresholds = get_thresholds_for_all_contamination_levels(contamination_levels, predicted_scores)

    predictions_at_contamination_levels = {}
    for level, thresh in thresholds.items():
        predictions = predict_based_on_threshold(thresh, predicted_scores)
        predictions_at_contamination_levels[level] = predictions
    return predictions_at_contamination_levels, predicted_scores


def batch_pyod_per_contamination_level(X_train_df, X_test_df, y_train, contamination_levels, model_dict, predict_on='test',
                                       semisupervised=False):
    """ Accepts a dictionary of {'model_name': model} and a list of contamination levels"""

    predictions_dict = model_dict.copy()
    predicted_scores_dict = {key: None for key in model_dict}
    for model_name, model in model_dict.items():
        print('Starting model ', model_name)
        predictions, predicted_scores = pyod_per_contamination_level(X_train_df, X_test_df, y_train,
                                                                     contamination_levels, model, predict_on,
                                                                     semisupervised)
        predictions_dict[model_name] = predictions
        predicted_scores_dict[model_name] = predicted_scores
    return predictions_dict, predicted_scores_dict


def supervised_model_per_contamination_level(X_train_df, X_test_df, y_train, contamination_levels, model):
    """ Accepts a list of contamination levels and a single model"""
    model.fit(X_train_df, y_train)
    predicted_scores = model.predict_proba(X_test_df)[:, 1]
    thresholds = get_thresholds_for_all_contamination_levels(contamination_levels, predicted_scores)

    predictions_at_contamination_levels = {}
    for level, thresh in thresholds.items():
        predictions = predict_based_on_threshold(thresh, predicted_scores, formula='greater_or_equal')
        if level == 0:
            predictions = predict_based_on_threshold(thresh, predicted_scores, formula='greater')
        predictions_at_contamination_levels[level] = predictions
    return predictions_at_contamination_levels, predicted_scores
