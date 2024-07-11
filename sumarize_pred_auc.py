import os, sys
import pandas as pd
from sklearn.metrics import roc_curve, auc, f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from eval import calc_auc

f_name='/scratch/work/huangg5/muse/MuSe-2024/results/prediction_muse/humor/lf/predictions_devel.csv'
df=pd.read_csv(f_name, index_col=0)
#print(df.shape, df.head(3))
preds=df['prediction'].to_numpy()
labels=df['label'].to_numpy()
print(preds.shape, labels.shape)
auc = calc_auc(preds, labels)

testy=labels
yhat_probs=preds
yhat_classes=preds>0.5

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(testy, yhat_classes)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(testy, yhat_classes)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(testy, yhat_classes)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(testy, yhat_classes)
print('F1 score: %f' % f1)
 
# kappa
kappa = cohen_kappa_score(testy, yhat_classes)
print('Cohens kappa: %f' % kappa)
# ROC AUC
auc = roc_auc_score(testy, yhat_probs)
print('ROC AUC: %f' % auc)
# confusion matrix
matrix = confusion_matrix(testy, yhat_classes)
print('Confusion Matrix: \n', matrix)