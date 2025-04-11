import numpy as np
import matplotlib.pyplot as plt



def generate_n_dimensions(size,num_features):
    np.random.seed(707)
    X = np.random.uniform(-10,10,size = (size,num_features)) # Matrix with random elemenets between -10 and 10.
    return X


def calculate_pca(X, n_components):
    #First phase the asumption is that the X values are centered but is better working with standarized values
    X_cent = X - np.mean(X, axis=0)
    X_std = np.std(X,axis=0)
    x_stda = X_cent/X_std
    #Covariance matrix
    dot_prod = x_stda.T @ x_stda
    #eigen values and eigene vectors
    eigenvalues, eigenvectors = np.linalg.eigh(dot_prod)
    #Sort the values from hig to low
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    #select the principal compnents
    V_d = eigenvectors[:, :n_components]
    #project the data using the componenets
    Y = x_stda @ V_d  # shape: (n_samples, n_components)
    #total variance explained in proportion
    total_variance = np.sum(eigenvalues)
    explained_variance = eigenvalues[:n_components]
    pve = explained_variance / total_variance

    return Y, V_d, pve


def expected_pve(X: np.ndarray, pve: float):
    #In here we only work with the assumption of center values but we can make the change to standarize for avoiding the pears v/s apples
    X_cent = X - np.mean(X, axis=0)
    #Covariance matrix
    dot_prod = X_cent.T @ X_cent
    eigenvalues, eigenvectors = np.linalg.eigh(dot_prod)

    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    total_variance = np.sum(eigenvalues)
    explained_variance_ratio = eigenvalues / total_variance
    cumulative_pve = np.cumsum(explained_variance_ratio)
    #in here we add the finder of the index that fulffil the pve explained
    k = np.argmax(cumulative_pve >= pve) + 1
    #If k is upper or equal to the dimensions it's a mistake so we apply a correction by force
    if k >= X.shape[1]:
        k = X.shape[1] - 1

    V_d = eigenvectors[:, :k]

    Y = X_cent @ V_d

    return Y, V_d,k