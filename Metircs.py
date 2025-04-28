import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score

def compute_sensitivity(y_true, y_pred, num_classes):
    """
    Compute sensitivity (recall) for each class.
    """
    sensitivity_list = []
    cm = confusion_matrix(y_true, y_pred)
    for i in range(num_classes):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        if tp + fn == 0:
            sensitivity_list.append(0.0)
        else:
            sensitivity_list.append(tp / (tp + fn))
    return sensitivity_list

def compute_precision(y_true, y_pred, num_classes):
    """
    Compute precision for each class.
    """
    precision_list = []
    cm = confusion_matrix(y_true, y_pred)
    for i in range(num_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        if tp + fp == 0:
            precision_list.append(0.0)
        else:
            precision_list.append(tp / (tp + fp))
    return precision_list

def compute_specificity(y_true, y_pred, num_classes):
    """
    Compute specificity for each class.
    """
    specificity_list = []
    cm = confusion_matrix(y_true, y_pred)
    total = np.sum(cm)
    for i in range(num_classes):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = total - tp - fn - fp
        if tn + fp == 0:
            specificity_list.append(0.0)
        else:
            specificity_list.append(tn / (tn + fp))
    return specificity_list

def compute_accuracy(y_true, y_pred):
    """
    Compute overall accuracy.
    """
    cm = confusion_matrix(y_true, y_pred)
    correct = np.trace(cm)
    total = np.sum(cm)
    return correct / total

def format_float_list(values):
    """
    Format list of floats to 4 decimal places.
    """
    return [float(f"{v:.4f}") for v in values]

def plot_confusion_matrix(cm, class_names, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    Plot confusion matrix.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           ylabel='True label',
           xlabel='Predicted label',
           title=title or ("Normalized Confusion Matrix" if normalize else "Confusion Matrix"))

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    plt.show()
