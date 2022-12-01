from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import mean_squared_error,make_scorer
from sklearn.metrics import average_precision_score,auc,PrecisionRecallDisplay
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import roc_curve,roc_auc_score,precision_recall_curve,precision_recall_curve
from matplotlib.colors import ListedColormap

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import torch

def get_classification_report(y, y_pred):
       # y_prime = np.stack([np.stack([d for d in d_]) for d_ in y]).flatten()
       # y_pred_prime=np.stack([np.stack([d for d in d_]) for d_ in y_pred]).flatten()   
        print(classification_report(y, y_pred))



def get_confusion_matrix(y, y_pred, plot=True):    
       
    cm = confusion_matrix(y, y_pred)
    # Normalise
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    if plot:
        fig, axs = plt.subplots(1, 2, figsize=(20,5))
        sns.heatmap(cm, ax=axs[0],annot=True, fmt='.2f', xticklabels=np.arange(len(torch.unique(torch.tensor(y_pred)))),
                    yticklabels=np.arange(len(torch.unique(torch.tensor(y_pred)))))
        axs[0].set_ylabel('Actual')
        axs[0].set_xlabel('Predicted')
        axs[0].set_title('CM')
        sns.heatmap(cmn, ax=axs[1] ,annot=True, fmt='.2f', xticklabels=np.arange(len(torch.unique(torch.tensor(y_pred)))),
                    yticklabels=np.arange(len(torch.unique(torch.tensor(y_pred)))))
        axs[1].set_ylabel('Actual')
        axs[1].set_xlabel('Predicted')
        axs[1].set_title('NCM')
        plt.show(block=False)
    return cm

def get_curves(y, y_pred,y_pred_score):
    n_classes=n_classes=len(np.unique(y))

    lw=2
    if n_classes==2:
      roc_auc = roc_auc_score(y, y_pred_score)
      fpr, tpr, thresh=roc_curve(y, y_pred_score)
      auc=metrics.auc(fpr, tpr)
      
      # Data to plot precision - recall curve
      precision, recall, thresholds = precision_recall_curve(y, y_pred_score)
      # Use AUC function to calculate the area under the curve of precision recall curve
      auc_precision_recall = metrics.auc(recall, precision)
      
      
      fig, axs = plt.subplots(1, 2, figsize=(20,5))
      axs[0].plot(fpr, tpr, label='ROC curve (area = %.2f)' %auc)
      axs[0].plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random guess')
      axs[0].set_ylabel('False Positive Rate')
      axs[0].set_xlabel('Predicted')
      axs[0].set_title('ROC curve')
      axs[0].grid()
      axs[0].legend()
      axs[1].plot(precision, recall, label='PR curve (area = %.2f)' %auc_precision_recall)
      #axs[1].plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random guess')
      axs[1].set_ylabel('Precision')
      axs[1].set_xlabel('Recall')
      axs[1].set_title('Binary class Precision-Recall curve')
      axs[1].grid()
      axs[1].legend()
      plt.show(block=False)

    if n_classes>2 :
      classes=np.arange(n_classes)
      y=label_binarize(y, classes=classes)
      fpr=dict()
      tpr = dict()
      roc_auc = dict()
      precision=dict()
      recall=dict()
      thresholds=dict()
      avg_precision=dict()
      for i in range(n_classes):
          fpr[i], tpr[i], _ = roc_curve(y[:, i], y_pred_score[:, i])
          roc_auc[i] =  metrics.auc(fpr[i], tpr[i])
          # Data to plot precision - recall curve
          precision[i], recall[i], thresholds[i] = precision_recall_curve(y[:, i], y_pred_score[:, i])
          # Use AUC function to calculate the area under the curve of precision recall curve
          avg_precision[i] = average_precision_score(y[:, i], y_pred_score[:, i])

      # Compute micro-average ROC curve and ROC area and PR
      fpr["micro"], tpr["micro"], _ = roc_curve(y.ravel(), y_pred_score.ravel())
      roc_auc["micro"] =  metrics.auc(fpr["micro"], tpr["micro"])
      precision["micro"], recall["micro"], thresholds["micro"] = precision_recall_curve(y.ravel(), y_pred_score.ravel())
      avg_precision["micro"] = average_precision_score(y,y_pred_score, average="micro")
          # First aggregate all false positive rates
      all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

      # Then interpolate all ROC curves at this points
      mean_tpr = np.zeros_like(all_fpr)
      for i in range(n_classes):
          mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

      # Finally average it and compute AUC
      mean_tpr /= n_classes

      fpr["macro"] = all_fpr
      tpr["macro"] = mean_tpr
      roc_auc["macro"] =  metrics.auc(fpr["macro"], tpr["macro"])

      fig, axs = plt.subplots(1,2 ,figsize=(20, 5))
      axs[0].plot(
          fpr["micro"],
          tpr["micro"],
          label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
          color="deeppink",
          linestyle=":",
          linewidth=4,
      )

      axs[0].plot(
          fpr["macro"],
          tpr["macro"],
          label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
          color="navy",
          linestyle=":",
          linewidth=4,
      )

      colors = cycle(["aqua", "darkorange", "cornflowerblue","teal","navy","crimson", "darkgreen","lavenderblush","mistyrose"])
      for i, color in zip(range(n_classes), colors):
          axs[0].plot(
              fpr[i],
              tpr[i],
              color=color,
              lw=2,
              label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
          )

      axs[0].plot([0, 1], [0, 1], "k--", lw=lw)
      axs[0].set_xlim([0.0, 1.0])
      axs[0].set_ylim([0.0, 1.05])
      axs[0].set_xlabel("False Positive Rate")
      axs[0].set_ylabel("True Positive Rate")
      axs[0].set_title("Some extension of Receiver operating characteristic to multiclass")
      axs[0].grid()
      axs[0].legend(loc="lower right")
     
      f_scores = np.linspace(0.2, 0.8, num=4)
      lines, labels = [], []
      for f_score in f_scores:
          x = np.linspace(0.01, 1)
          y = f_score * x / (2 * x - f_score)
          (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
          plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))
      display = PrecisionRecallDisplay(
      recall=recall["micro"],
      precision=precision["micro"],
      average_precision=avg_precision["micro"],
  )
      display.plot(ax=axs[1], name="Micro-average precision-recall", color="gold")

      for i, color in zip(range(n_classes), colors):
          display = PrecisionRecallDisplay(
              recall=recall[i],
              precision=precision[i],
              average_precision=avg_precision[i],
          )
          display.plot(ax=axs[1], name=f"Precision-recall for class {i}", color=color)

      # add the legend for the iso-f1 curves
      handles, labels = display.ax_.get_legend_handles_labels()
      handles.extend([l])
      labels.extend(["iso-f1 curves"])
      # set the legend and the axes
      axs[1].set_xlim([0.0, 1.0])
      axs[1].set_ylim([0.0, 1.05])
      axs[1].legend(handles=handles, labels=labels, loc="best")
      axs[1].set_title("Extension of Precision-Recall curve to multi-class")

      plt.show(block=False)
            
    return 0
              
def get_report(y, y_pred,y_pred_score):
    get_classification_report(y, y_pred)
    get_confusion_matrix(y, y_pred, plot=True)
    get_curves(y, y_pred,y_pred_score)
    pass


def decompose_trial_into_relevant_columns_pandas(trials):
    df = pd.DataFrame()
    df['loss']=trials.losses()
    if 'learning_rate' in trials.vals.keys():
      df['learning_rate']=trials.vals['learning_rate']
    if 'num_h_layers' in trials.vals.keys():
      df['num_h_layers']=[int(a) for a in trials.vals['num_h_layers']]
    if 'batch_size'  in trials.vals.keys():
      df['batch_size']=[int(a) for a in trials.vals['batch_size']]
    return df


def resettable(f):
    import copy

    def __init_and_copy__(self, *args, **kwargs):
        f(self, *args)
        self.__original_dict__ = copy.deepcopy(self.__dict__)

        def reset(o = self):
            o.__dict__ = o.__original_dict__

        self.reset = reset

    return __init_and_copy__