import numpy as np
from collections import defaultdict

class MyBernoulliNBClassifier():
    def __init__(self, priors=None):
        self.class_proba_dct = defaultdict(lambda: 0)
        self.omp = defaultdict(lambda: 0)
        
    def fit(self, X, y):
        L = len(y)
        
        # вычисляем выборочное распределение классов y
        for i in y:
            self.class_proba_dct[i] += (1. / L)
            
        # вычисляем оценки максимального правдободобия для каждого признака в X и класса в y
        
        for row, k in zip(X, y):
            for j in range(len(row)):
                if row[j] == 1:
                    self.omp[(k, row[j])] += L * (1. / self.class_proba_dct[k]) 
    
    def predict(self, X):
        
        K = len(self.class_proba_dct)
        self.pred = []
        
        for row in X:
            dct = dict(zip(self.class_proba_dct.keys(), np.zeros(K)))
            for k in dct:
                p = self.class_proba_dct[k] 
                for i in range(len(row)):
                    p *= (self.omp[(k, row[i])]*row[i] + (1 - self.omp[(k, row[i])])*(1 - row[i]))
                dct[k] = p
            self.pred.append(max(dct, key = dct.get))
                       
        return self.pred
    
    def predict_proba(self, X):
        
        K = len(self.class_proba_dct)
        n_samples = np.shape(X)[0]
        
        self.pred_proba = []
        
        for row in X:
            dct = dict(zip(self.class_proba_dct.keys(), np.zeros(K)))
            for k in dct.keys():
                p = self.class_proba_dct[k] 
                for i in range(len(row)):
                    p *= (self.omp[(k, row[i])]*row[i] + (1 - self.omp[(k, row[i])])*(1 - row[i]))
                dct[k] = p
            self.pred_proba.append(max(dct))
                       
        return self.pred_proba
    
    def score(self, X, y):
        acc = 0
        L = len(y)
        pred = self.predict(X)
        
        for i in range(len(y)):
            if (pred[i] == y[i]):
                acc += 1
        acc = acc / L     
        
        return acc