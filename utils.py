from sklearn.metrics import accuracy_score,precision_score,recall_score


def get_metrics(y_train,y_pred,y_test,y_test_pred):
        metrics={}
        metrics['training_accuracy']=accuracy_score(y_train,y_pred)
        metrics['training_precision']= precision_score(y_train,y_pred)
        metrics['training_recall']= recall_score(y_train,y_pred)
        metrics['test_accuracy']=accuracy_score(y_test,y_test_pred)
        metrics['test_precision']= precision_score(y_test,y_test_pred)
        metrics['test_recall']=recall_score(y_test,y_test_pred)
        return get_metrics