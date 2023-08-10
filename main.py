from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.base import BaseEstimator, ClassifierMixin


class CustomClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, model):
        self.model = model
    
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def train(self, X_train, y_train, X_test, y_test):
        self.fit(X_train, y_train)
        y_pred = self.predict(X_test)
        return y_pred
    
    def f1_score(self, y_true, y_pred):
        return metrics.f1_score(y_true, y_pred, average='macro')
    
    def accuracy_score(self, y_true, y_pred):
        return metrics.accuracy_score(y_true, y_pred)
    
    def confusion_matrix(self, y_true, y_pred):
        return metrics.confusion_matrix(y_true, y_pred)
