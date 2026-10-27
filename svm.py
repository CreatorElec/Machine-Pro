import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_decision_boundary(model, X, y):
    # Create a mesh grid for plotting decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    # Predict the class for each point in the mesh grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary and the training points
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.title('Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

def main():
    # Load dataset
    data = pd.read_csv('data.csv')  # Replace 'data.csv' with your dataset file
    X = data.iloc[:, :-1].values  # Features
    y = data.iloc[:, -1].values   # Target variable

    # Create and train the SVM model
    model = sklearn.svm.SVC(kernel='linear')  # You can change the kernel as needed
    model.fit(X, y)
    model.score(X, y)  # Evaluate the model (optional)
    model.predict(X)  # Make predictions (optional)


    # Plot the decision boundary
    plot_decision_boundary(model, X, y)

if __name__ == "__main__":
    main()