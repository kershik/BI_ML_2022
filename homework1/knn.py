import numpy as np


class KNNClassifier:
    """
    K-neariest-neighbor classifier using L1 loss
    """
    
    def __init__(self, k=1):
        self.k = k
    

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y


    def predict(self, X, n_loops=0):
        """
        Uses the KNN model to predict clases for the data samples provided
        
        Arguments:
        X, np array (num_samples, num_features) - samples to run
           through the model
        num_loops, int - which implementation to use

        Returns:
        predictions, np array of ints (num_samples) - predicted class
           for each sample
        """
        
        if n_loops == 0:
            distances = self.compute_distances_no_loops(X)
        elif n_loops == 1:
            distances = self.compute_distances_one_loops(X)
        else:
            distances = self.compute_distances_two_loops(X)
        
        if len(np.unique(self.train_y)) == 2:
            return self.predict_labels_binary(distances)
        else:
            return self.predict_labels_multiclass(distances)


    def compute_distances_two_loops(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Uses simplest implementation with 2 Python loops

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """
        distances = np.zeros((len(X), len(self.train_X)))
            
        for i in range(len(X)):
            for j in range(len(self.train_X)):
                distances[i][j] = np.sum(abs(X[i] - self.train_X[j]))
        return distances   
       
    def compute_distances_one_loop(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Vectorizes some of the calculations, so only 1 loop is used

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """
        distances = np.zeros((len(X), len(self.train_X)))
        for i in range(len(X)):
                distances[i] = np.sum(abs(X[i] - self.train_X), 1)
        return distances 

    def compute_distances_no_loops(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Fully vectorizes the calculations using numpy

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """
        X = np.expand_dims(X, 1) # драсьте скинула всем статью, а сама делаю бродкастинг руками logic
        distances = np.sum(abs(X - self.train_X), 2)
        return distances       
       
    
    def predict_labels_binary(self, distances):
        """
        Returns model predictions for binary classification case
        
        Arguments:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        Returns:
        pred, np array of bool (num_test_samples) - binary predictions 
           for every test sample
        """

#         n_train = distances.shape[1] я не пони как это использовать
#         n_test = distances.shape[0]
#         prediction = np.zeros(n_test)
        
        min_dist_index = [el[:self.k] for el in np.argsort(distances, 1)] # ищем индексы минимальных расстояний (ака ближайшего трейна) в строке теста
        prediction = np.array(['0' if (self.train_y[i]=='0').sum()>(self.train_y[i]=='1').sum() else '1' for i in min_dist_index]) 
        # смотрим на самый часто встречающийся класс ближайшего трейна, можно наверное было бы написать проще используя collections или scipy,но раз уж мы по харду только с numpy
        
        return prediction

    def predict_labels_multiclass(self, distances):
        """
        Returns model predictions for multi-class classification case
        
        Arguments:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        Returns:
        pred, np array of int (num_test_samples) - predicted class index 
           for every test sample
        """

#         n_train = distances.shape[0] 
#         n_test = distances.shape[0]
#         prediction = np.zeros(n_test, np.int)
        
        min_dist_index = np.array([el[:self.k] for el in np.argsort(distances, 1)]) # ищем индексы минимальных расстояний (ака ближайшего трейна) в строке теста
        prediction = np.array([self.train_y[np.argmax(np.bincount(i))] for i in min_dist_index]) # нечто в квадратных скобках это я ищу индекс самого популярного элемета в массиве индексов самых маленьких дистанций
        return prediction 
             
             
             
        

       
