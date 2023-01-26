from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
import logging
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
import time
import xgboost as xgb 
from utils import get_metrics



"""
This module consists of naivebayes,logistic regression and xgboost algorithms with and without hyperparameter tuning
"""


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])


class NaiveBayes:
    """
    Class for naive bayes implementation
    """

    def train_model_without_hp(self,X_train,y_train,X_test,y_test):
       metrics ={}
       logging.info("Training the model without hyperparameter tuning")
       nb = BernoulliNB()
       nb.fit(X_train,y_train)
       y_pred = nb.predict(X_train)
       y_test_pred = nb.predict(X_test)
       metrics = get_metrics(y_train,y_pred,y_test,y_test_pred)
       logging.info("Finished training at {}".format(time.gmtime()))
       return nb,metrics
    
    def train_model_with_hp(self,X_train,y_train,X_test,y_test):
        param_grid={'alpha':[1e-7,1e-5,1e-3,1e-1]}
        nb = BernoulliNB()
        logging.info("Started training naive bayes with hyperparameter tuning")
        random_cv = RandomizedSearchCV(nb,cv=5,random_state=42,param_distributions=param_grid)
        random_cv.fit(X_train,y_train)
        logging.info("Best params {} ".format(random_cv.best_params_))
        best_model = random_cv.best_estimator_
        best_model.fit(X_train,y_train)
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)
        metrics=get_metrics(y_train,y_train_pred,y_test,y_test_pred)
        logging.info("Finished training at {}".format(time.gmtime()))
        return best_model,metrics

class LRClassification:
    """
    Class for logistic regression implementation
    """

    def train_model_without_hp(self,X_train,y_train,X_test,y_test):
       metrics ={}
       logging.info("Training the model without hyperparameter tuning")
       lr = LogisticRegression(random_state=42,class_weight='balanced')
       lr.fit(X_train,y_train)
       y_pred = lr.predict(X_train)
       y_test_pred = lr.predict(X_test)
       metrics = get_metrics(y_train,y_pred,y_test,y_test_pred)
       logging.info("Finished training at {}".format(time.gmtime()))
       return lr,metrics
    
    def train_model_with_hp(self,X_train,y_train,X_test,y_test):
        param_grid={'C':[0.1,0.5,1],'tol':[1e-2,1e-4,1e-1]}
        lr = LogisticRegression(random_state=42,class_weight='balanced')
        logging.info("Started training logistic regression with hyperparameter tuning")
        random_cv = RandomizedSearchCV(lr,random_state=42,param_distributions=param_grid,cv=5)
        random_cv.fit(X_train,y_train)
        logging.info("Best params {} ".format(random_cv.best_params_))
        best_model = random_cv.best_estimator_
        best_model.fit(X_train,y_train)
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)
        metrics=get_metrics(y_train,y_train_pred,y_test,y_test_pred)
        logging.info("Finished training at {}".format(time.gmtime()))
        return best_model,metrics

class XGBoost:

    """
    Xgboost implementation with and without hyperparameter tuning
    """
    def train_model_without_hp(self,X_train,y_train,X_test,y_test):
       metrics ={}
       logging.info("Training the model without hyperparameter tuning")
       lr = xgb.XGBClassifier(random_state=42,n_jobs=-1,objective='binary:logistic')
       lr.fit(X_train,y_train)
       y_pred = lr.predict(X_train)
       y_test_pred = lr.predict(X_test)
       metrics = get_metrics(y_train,y_pred,y_test,y_test_pred)
       logging.info("Finished training at {}".format(time.gmtime()))
       return lr,metrics
    
    def train_model_with_hp(self,X_train,y_train,X_test,y_test):
        param_grid={
            'n_estimators':['100','200','300'],
            'learning_rate':[1e-2,0.05,0.25],'min_child_weight':[2,5,7,9],'max_depth':[3,4,5,7], 'gamma': [0.1, 0.5, 1, 5],
        'subsample': [0.6, 0.8, 1.0]}
        xgbc = xgb.XGBClassifier(random_state=42,n_jobs=-1)
        logging.info("Started training xgboost  with hyperparameter tuning")
        random_cv = RandomizedSearchCV(xgbc,cv=5,random_state=42,param_distributions=param_grid,n_iter=100,n_jobs=-1)
        random_cv.fit(X_train,y_train)
        logging.info("Best params {} ".format(random_cv.best_params_))
        best_model = random_cv.best_estimator_
        best_model.fit(X_train,y_train)
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)
        metrics=get_metrics(y_train,y_test_pred,y_test,y_test_pred)
        logging.info("Finished training at {}".format(time.gmtime()))
        return best_model,metrics