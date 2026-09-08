import numpy as np
import pandas as pd

def generate_random_data(num_samples=100, num_features=2, num_classes=2):
    """
    Generate a random dataset for classification.

    Parameters:
    - num_samples: Number of samples to generate.
    - num_features: Number of features for each sample.
    - num_classes: Number of classes for the target variable.

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
    X, y = generate_random_data(num_samples=100, num_features=2, num_classes=2)

    # Save the data to a CSV file
    save_data_to_csv(X, y, filename='data.csv')

if __name__ == "__main__":
    main()