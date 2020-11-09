### ------ --- Assignment 1 - Question 5 PART A -------- ###
### ----------- 470386390, 470203101, 470205127 -------- ###

# Plot confusion matrix

# Import required packages
import matplotlib.pyplot as plt
import itertools
import numpy as np
from sklearn.metrics import confusion_matrix

# Plot the confusion matrix in a plot with colorbar and colour coded   
def plotConfusionMatrix(ground_truth, predictions, classes):
    cm = confusion_matrix(ground_truth, predictions)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt =  'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()