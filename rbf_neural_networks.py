import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, RocCurveDisplay
import matplotlib.pyplot as plt

class RBFNeuralNetwork:
    def __init__(self, num_centers, learning_rate=0.01, epochs=100):
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
        if (self.encoder is None) or (self.y_onehot is None) or (self.y_onehot.shape[0] != y.shape[0]):
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

    def predict(self, X):
        Phi = self.compute_activations(X)
        y_pred = Phi @ self.weights

        return np.argmax(y_pred, axis=1)
    

class RBFExperiment:
    def __init__(self, data_name, num_centers, X_train, y_train,X_test, y_test, learning_rate=0.01, epochs=100):
        self.data_name = data_name
        self.model = RBFNeuralNetwork(num_centers=num_centers, 
                                      learning_rate=learning_rate, 
                                      epochs=epochs)
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        self.y_pred = None
        self.y_scores = None
        self.y_test_onehot = None

    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        self.y_pred = self.model.predict(self.X_test)
    
    def calculate_scores(self):
        Phi_test = self.model.compute_activations(self.X_test)
        self.y_scores = Phi_test @ self.model.weights
    
    def calculate_y_test_onehot(self):
        self.y_test_onehot = self.model.encoder.transform(self.y_test.reshape(-1, 1))
        
    def test_accuracy(self):
        overall_acc = np.mean(self.y_pred == self.y_test)
        
        class_acc = {}
        classes = np.unique(self.y_test)
        for c in classes:
            cls_acc = np.mean(self.y_pred[self.y_test == c] == self.y_test[self.y_test == c])
            class_acc[c] = cls_acc
        return overall_acc, class_acc
    
    def train_accuracy(self):
        y_pred_train = self.model.predict(self.X_train)
        
        overall_acc = np.mean(y_pred_train == self.y_train)
        
        class_acc = {}
        classes = np.unique(self.y_train)
        for c in classes:
            cls_acc = np.mean(y_pred_train[self.y_train == c] == self.y_train[self.y_train == c])
            class_acc[c] = cls_acc
        return overall_acc, class_acc

    
    def plot_precision_recall_multiclass(self):
        if self.y_scores is None:
            self.calculate_scores()
        if self.y_test_onehot is None:
            self.calculate_y_test_onehot()
        
        plt.figure(figsize=(8, 6))

        n_classes = self.y_scores.shape[1]
        for i in range(n_classes):
            precision, recall, _ = precision_recall_curve(
                self.y_test_onehot[:, i],
                self.y_scores[:, i]
            )

            disp = PrecisionRecallDisplay(
                precision=precision,
                recall=recall
            )
            disp.plot(ax=plt.gca(), label=f'Classe {i}')

        plt.title("Courbe Précision-Rappel pour " + self.data_name)
        plt.xlabel("Rappel")
        plt.ylabel("Précision")
        plt.legend()
        plt.grid(True)
        plt.show()

    
    def plot_roc_multiclass(self):
        if self.y_scores is None:
            self.calculate_scores()
        if self.y_test_onehot is None:
            self.calculate_y_test_onehot()
        
        plt.figure(figsize=(8, 6))

        n_classes = self.y_scores.shape[1]
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(
                self.y_test_onehot[:, i],
                self.y_scores[:, i]
            )

            disp = RocCurveDisplay(fpr=fpr, tpr=tpr)
            disp.plot(
                ax=plt.gca(),
                curve_kwargs={"label": f"Classe {i}"}
            )

        plt.title("Courbe ROC pour " + self.data_name)
        plt.xlabel("Taux de Faux Positifs")
        plt.ylabel("Taux de Vrais Positifs")
        plt.legend()
        plt.grid(True)
        plt.show()

    
    def plot_confusion_matrix(self):
        labels = np.unique(self.y_test)
        cm = confusion_matrix(self.y_test, self.y_pred, labels=labels)

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=[f"Classe {l}" for l in labels] 
        )

        disp.plot(cmap=plt.cm.Blues)
        plt.title("Matrice de Confusion pour " + self.data_name)
        plt.xlabel("Étiquette prédite")
        plt.ylabel("Étiquette réelle")
        plt.grid(False)
        plt.show()