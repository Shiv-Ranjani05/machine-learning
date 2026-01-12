from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,roc_curve,auc

y_true = [0,1,1,0,1,0,0,1,1,0]
y_pred=[0,1,0,0,1,1,0,1,1,1]

cm=confusion_matrix(y_true,y_pred)
accuracy=accuracy_score(y_true,y_pred)
precision=precision_score(y_true,y_pred)
recall=recall_score(y_true,y_pred)
f1=f1_score(y_true,y_pred)
fpr,tpr,thresholds=roc_curve(y_true,y_pred)
roc_auc=auc(fpr,tpr)
print("confusion matrix:",cm)
print("accuracy score:",accuracy)
print("precision score:",precision)
print("recall score:",recall)
print("f1 score:",f1)
print("roc auc:",roc_auc)