import numpy as np
import pandas as pd

grid_type="diagonal_left","diagonal_right","diagonal","horizontal","vertical","square","circle","triangle","random"
def generate_diagonal_left_data(num_samples=100, num_features=2, num_classes=2,noise=0.1):
    anchor_points = np.array([[i/(num_samples-1), -i/(num_samples-1)] for i in range(num_samples)])
    X = np.random.rand(num_samples, num_features)+anchor_points+np.random.normal(0, noise, (num_samples, num_features))  # Random features
    y=X[:, 1]>0
    return X, y

def generate_diagonal_right_data(num_samples=100, num_features=2, num_classes=2,noise=0.1):
    anchor_points = np.array([[i/(num_samples-1), i/(num_samples-1)] for i in range(num_samples)])
    X = np.random.rand(num_samples, num_features)+anchor_points+np.random.normal(0, noise, (num_samples, num_features))  # Random features
    y = X[:, 1]>0
    return X, y

def generate_diagonal_data(num_samples=100, num_features=2, num_classes=2,noise=0.1):
    anchor_points = np.array([[i/(num_samples-1), i/(num_samples-1)] for i in range(num_samples)])
    X = np.random.rand(num_samples, num_features)+anchor_points+np.random.normal(0, noise, (num_samples, num_features))  # Random features
    y = X[:, 1]>0
    return X, y

def generate_horizontal_data(num_samples=100, num_features=2, num_classes=2,noise=0.1):
    anchor_points = np.array([[0, i/(num_samples-1)] for i in range(num_samples)])
    X = np.random.rand(num_samples, num_features)+anchor_points+np.random.normal(0, noise, (num_samples, num_features))  # Random features
    y = X[:, 1]>0
    return X, y

def generate_vertical_data(num_samples=100, num_features=2, num_classes=2,noise=0.1):
    anchor_points = np.array([[i/(num_samples-1), 0] for i in range(num_samples)])
    X = np.random.rand(num_samples, num_features)+anchor_points  # Random features
    y = np.random.randint(0, num_classes, size=num_samples)  # Random target
    return X, y

def generate_square_data(num_samples=100, num_features=2, num_classes=2,noise=0.1):
    pass

def generate_circle_data(num_samples=100, num_features=2, num_classes=2,noise=0.1):
    pass

def generate_triangle_data(num_samples=100, num_features=2, num_classes=2,noise=0.1):
    pass

def generate_random_data(num_samples=100, num_features=2, num_classes=2,noise=0.1):
    """
    Generate a random dataset for classification.

    Parameters:
    - num_samples: Number of samples to generate.
    - num_features: Number of features for each sample.
    - num_classes: Number of classes for the target variable.
    - noise: Amount of noise to add to the data.

    Returns:
    - X: Feature matrix of shape (num_samples, num_features).
    - y: Target vector of shape (num_samples,).
    """
    # Generate random features
    X = np.random.rand(num_samples, num_features)

    # Generate random target variable
    y = np.random.randint(0, num_classes, size=num_samples)

    return X, y

def save_data_to_csv(X, y, filename='data.csv'):
    """
    Save the generated dataset to a CSV file.

    Parameters:
    - X: Feature matrix.
    - y: Target vector.
    - filename: Name of the CSV file to save the data.
    """
    # Create a DataFrame from the features and target
    df = pd.DataFrame(X, columns=[f'feature_{i+1}' for i in range(X.shape[1])])
    df['target'] = y

    # Save to CSV
    df.to_csv(filename, index=False)

def main():
    # Generate random data
    X, y = generate_diagonal_left_data(num_samples=100, num_features=2, num_classes=2, noise=0.1)

    # Save the data to a CSV file
    save_data_to_csv(X, y, filename='data.csv')

if __name__ == "__main__":
    main()