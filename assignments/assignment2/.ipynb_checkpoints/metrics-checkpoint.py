def multiclass_accuracy(prediction, ground_truth):
    """
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    # TODO: Implement computing accuracy
    tp_tn = 0
    for pred, target in zip(prediction, ground_truth):
        if target == pred:
            tp_tn += 1
    
    return tp_tn / len(ground_truth) 
