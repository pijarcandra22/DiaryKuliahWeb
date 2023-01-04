import numpy as np

class MultiNB:
    def __init__(self,alpha=1):
        self.alpha = alpha
    
    def _prior(self): # CHECKED
        """
        Calculates prior for each unique class in y. P(y)
        """
        P = np.zeros((self.n_classes_))
        _, self.dist = np.unique(self.y,return_counts=True)
        for i in range(self.classes_.shape[0]):
            P[i] = self.dist[i] / self.n_samples
        return P
            
    def fit(self, X, y): # CHECKED, matches with sklearn
        """
        Calculates the following things- 
            class_priors_ is list of priors for each y.
            N_yi: 2D array. Contains for each class in y, the number of time each feature i appears under y.
            N_y: 1D array. Contains for each class in y, the number of all features appear under y.
            
        params
        ------
        X: 2D array. shape(n_samples, n_features)
            Multinomial data
        y: 1D array. shape(n_samples,). Labels must be encoded to integers.
        """
        self.y = y
        self.n_samples, self.n_features = X.shape
        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.shape[0]
        self.class_priors_ = self._prior()
        
        # distinct values in each features
        self.uniques = []
        for i in range(self.n_features):
            tmp = np.unique(X[:,i])
            self.uniques.append( tmp )
            
        self.N_yi = np.zeros((self.n_classes_, self.n_features)) # feature count
        self.N_y = np.zeros((self.n_classes_)) # total count 
        for i in self.classes_: # x axis
            indices = np.argwhere(self.y==i).flatten()
            columnwise_sum = []
            for j in range(self.n_features): # y axis
                columnwise_sum.append(np.sum(X[indices,j]))
                
            self.N_yi[i] = columnwise_sum # 2d
            self.N_y[i] = np.sum(columnwise_sum) # 1d
            
    def _theta(self, x_i, i, h):
        """
        Calculates theta_yi. aka P(xi | y) using eqn(1) in the notebook.
        
        params
        ------
        x_i: int. 
            feature x_i
            
        i: int.
            feature index. 
            
        h: int or string.
            a class in y
        
        returns
        -------
        theta_yi: P(xi | y)
        """
        
        Nyi = self.N_yi[h,i]
        Ny  = self.N_y[h]
        
        numerator = Nyi + self.alpha
        denominator = Ny + (self.alpha * self.n_features)
        
        return  (numerator / denominator)**x_i
    
    def _likelyhood(self, x, h):
        """
        Calculates P(E|H) = P(E1|H) * P(E2|H) .. * P(En|H).
        
        params
        ------
        x: array. shape(n_features,)
            a row of data.
        h: int. 
            a class in y
        """
        tmp = []
        for i in range(x.shape[0]):
            tmp.append(self._theta(x[i], i,h))
        
        return np.prod(tmp)
    
    def predict(self, X):
        samples, features = X.shape
        self.predict_proba = np.zeros((samples,self.n_classes_))
        
        for i in range(X.shape[0]):
            joint_likelyhood = np.zeros((self.n_classes_))
            
            for h in range(self.n_classes_):
                joint_likelyhood[h]  = self.class_priors_[h] * self._likelyhood(X[i],h) # P(y) P(X|y) 
                
            denominator = np.sum(joint_likelyhood)
            
            for h in range(self.n_classes_):
                numerator = joint_likelyhood[h]
                self.predict_proba[i,h] = (numerator / denominator)
            
        indices = np.argmax(self.predict_proba,axis=1)
        return self.classes_[indices]