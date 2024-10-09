import numpy
import os
import sklearn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle


class Evaluation:
  def __init__(self, y_test, y_pred):
    self.y_test = y_test
    self.y_pred = y_pred
    self.confusion_matrix = sklearn.metrics.confusion_matrix(self.y_test, self.y_pred)
    self.n_classes = len(numpy.unique(self.y_test))

  def get_TFPN(self, y_test, y_pred):
    for i in range(self.n_classes):
      TP = self.confusion_matrix[i, i]
      FP = self.confusion_matrix[:, i].sum() - TP
      FN = self.confusion_matrix[i, :].sum() - TP
      TN = self.confusion_matrix.sum() - (TP + FP + FN)

      return TP, FP, FN, TN
  def get_Precision_Recall(self):
    TP, FP, FN, TN = self.get_TFPN(self.y_test, self.y_pred)
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    return Precision, Recall
  def get_F1_Score(self):
    Precision, Recall = self.get_Precision_Recall()
    F1_Score = 2 * (Precision * Recall) / (Precision + Recall)
    return F1_Score
  def get_Specificity(self):
    TP, FP, FN, TN = self.get_TFPN(self.y_test, self.y_pred)
    Specificity = TN / (TN + FP)
    return Specificity
  def get_Sensitivity(self):
    TP, FP, FN, TN = self.get_TFPN(self.y_test, self.y_pred)
    Sensitivity = TP / (TP + FN)
    return Sensitivity
  def draw_ROC_Curve(self):
    # three classes in label
    y_test_bin = label_binarize(self.y_test, classes=[0, 1, 2])
    n_classes = y_test_bin.shape[1]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], self.y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                  label='ROC curve of class {0} (area = {1:0.2f})'
                  ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic for Multi-class')
    plt.legend(loc="lower right")
    plt.show()

    return roc_auc