def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    tp = ((ground_truth == True) & (prediction == True)).sum()
    fp = ((ground_truth == False) & (prediction == True)).sum()
    tn = ((ground_truth == False) & (prediction == False)).sum()
    fn = ((ground_truth == True) & (prediction == False)).sum()
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    f1 = 2 * precision * recall / (precision + recall)

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    tp_tn = 0
    for pred, target in zip(prediction, ground_truth):
        if target == pred:
            tp_tn += 1
    
    return tp_tn / len(ground_truth) 
