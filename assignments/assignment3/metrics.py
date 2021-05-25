import numpy as np


def binary_classification_metrics(prediction, ground_truth):
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    # TODO: implement metrics!
    
    return accuracy, precision, recall, f1


def multiclass_accuracy(prediction, ground_truth):
    tp_tn = np.sum(prediction == ground_truth)
    return tp_tn / len(ground_truth) 
