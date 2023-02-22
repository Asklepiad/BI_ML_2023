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

    y_true = y_true.astype("int64").astype("bool")
    confusion_voc = {"tp":0, "fp":0, "fn":0, "tn":0}
    for y in range(len(y_pred)):
        if y_pred[y] == False and y_true[y] == False:
            confusion_voc["tn"] += 1
        elif y_pred[y] == True and y_true[y] == True:
            confusion_voc["tp"] += 1
        elif y_pred[y] == False and y_true[y] == True:
            confusion_voc["fn"] += 1
        elif y_pred[y] == True and y_true[y] == False:
            confusion_voc["fp"] += 1

        tp = confusion_voc["tp"]
        fp = confusion_voc["fp"]
        fn = confusion_voc["fn"]
        tn = confusion_voc["tn"]
        
        try:
            precision = tp / (tp + fp)
        except ZeroDivisionError:
            precision = 0
        
        try:
            recall = tp / (tp + fn)
        except ZeroDivisionError:
            recall = 0

        try:
            f1 = 2 * ((precision * recall) / (precision + recall))
        except:
            f1 = 0

        accuracy = (tp + tn) / (tp + tn + fp + fn)

        return (precision, recall, f1, accuracy)


def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """
    accuracy = (y_pred == y_true).astype("int8").mean()
    return accuracy


def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """
    r2 = 1 - (((y_true - y_pred)**2).sum() / ((y_true - (y_true.mean()))**2).sum())
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
    mse = ((y_true - y_pred)**2).sum() / len(y_pred)
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
    mae = (np.abs(y_true - y_pred)).sum() / len(y_pred)
    return mae
