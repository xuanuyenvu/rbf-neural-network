import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder

class RBFNeuralNetwork:
    def __init__(self, num_centers, learning_rate=0.01, epochs=100, ):
        self.num_centers = num_centers
        self.learning_rate = learning_rate
        self.epochs = epochs
        
        self.centers = None
        self.weights = None
        self.beta = None
        
        self.encoder = None
        self.y_onehot = None


    def rbf(self, x, center):
        return np.exp(-self.beta * np.linalg.norm(x - center) ** 2)

    def compute_activations(self, X):
        G = np.zeros((X.shape[0], self.num_centers))
        for i, center in enumerate(self.centers):
            for j, x in enumerate(X):
                G[j, i] = self.rbf(x, center)
        return G
    
    def select_centers_kmeans(self, X):
        kmeans = KMeans(n_clusters=self.num_centers, random_state=42)
        kmeans.fit(X)
        self.centers = kmeans.cluster_centers_

    def train_output_weights_gd(self, y, Phi):    
        if (self.y_onehot is None) or (self.y_onehot.shape[0] != y.shape[0]):
            self.encoder = OneHotEncoder(sparse_output=False)
            self.y_onehot = self.encoder.fit_transform(y.reshape(-1, 1))  

        N, M = Phi.shape
        C = self.y_onehot.shape[1]
        W = np.random.randn(M, C) * 0.01
        
        # Gradient Descent
        for epoch in range(self.epochs):
            y_pred = Phi @ W
            errors = y_pred - self.y_onehot  
            gradient = (2 / N) * (Phi.T @ errors) 
            W -= self.learning_rate * gradient
        
        return W

    def fit(self, X, y):
        # Step 1: Determine centers using KMeans
        self.select_centers_kmeans(X)

        # Step 2: Calculate beta (spread parameter)
        d_max = np.max([np.linalg.norm(c1 - c2) for c1 in self.centers for c2 in self.centers])
        self.beta = 1 / (2 * (d_max / np.sqrt(2 * self.num_centers)) ** 2)

        # Step 3: Compute activations
        Phi = self.compute_activations(X)

        # Step 4: Initialize weights
        self.weights = self.train_output_weights_gd(y, Phi) 
        print("weights:", self.weights)  

    def predict(self, X):
        Phi = self.compute_activations(X)
        y_pred = Phi @ self.weights

        return np.argmax(y_pred, axis=1)