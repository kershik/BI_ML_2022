import numpy as np


def binary_classification_metrics(y_pred, y_true):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score

    tp = (y_pred[y_pred=='1'] == y_true[y_pred=='1']).sum()
    tn = (y_pred[y_pred=='0'] == y_true[y_pred=='0']).sum()
    fp = (y_pred[y_pred=='1'] != y_true[y_pred=='1']).sum()
    fn = (y_pred[y_pred=='0'] != y_true[y_pred=='0']).sum()
    
    precision = tp/(tp + fp) if tp != 0 else 0
    recall = tp/(tp + fn) if tp != 0 else 0
    f1 = 2 * (precision * recall)/(precision + recall) if (precision + recall) != 0 else 0
    accuracy = (tp + tn)/(tp + tn + fp + fn) if tp+tn != 0 else 0
    return precision, recall, f1, accuracy

def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    conf_matrix = np.array([[(y_true[y_pred==i]==j).sum() for j in np.unique(y_true)] for i in np.unique(y_pred)]) # сломала голову пока придумывала это шикарное выражение, по вретикали y_pred по горизонтали y_true
    sum_accuracy = 0
    for i in range(len(conf_matrix)): 
        tp_i = conf_matrix[i,i]
        tn_i = np.sum(conf_matrix[:i,:i]) + np.sum(conf_matrix[(i+1):,(i+1):]) + np.sum(conf_matrix[(i+1):,:i]) + np.sum(conf_matrix[:i,(i+1):])
        fp_i = np.sum(conf_matrix[i]) - conf_matrix[i,i]
        fn_i = np.sum(conf_matrix[:,i]) - conf_matrix[i,i]
        accuracy_i = (tp_i + tn_i)/(tp_i + tn_i + fp_i + fn_i)
        sum_accuracy += accuracy_i
    return (sum_accuracy/len(conf_matrix))

def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """

    SSres = np.sum(np.power(y_true - y_pred, 2))
    SStot = np.sum(np.power(y_true - np.mean(y_true), 2))
    r2 = 1 - SSres/SStot
    return r2

def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """

    mse = 1/len(y_pred) * np.sum(np.power((y_true - y_pred), 2))
    return mse


def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """

    mae = 1/len(y_pred) * np.sum(abs(y_true - y_pred))
    return mae
    