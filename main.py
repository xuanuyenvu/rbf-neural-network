import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from rbf_neural_networks import RBFNeuralNetwork

if __name__ == "__main__":
    # Load Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the RBF Neural Network
    print("Num training samples:", X_train.shape[0])
    rbf_nn = RBFNeuralNetwork(num_centers=10)
    rbf_nn.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = rbf_nn.predict(X_test)

    # Calculate accuracy
    accuracy = np.mean(predictions == y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")