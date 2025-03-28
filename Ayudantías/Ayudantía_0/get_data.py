import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 

#Space to generete the functions for the main code

def generate_data_1_d(num_gen,w,b,random_scale = 10):
    """
    The objective of the function: Is to generate a vector X with the input numbers and generate a Y vector with the values to be predicted

    Args:
    
    num_gen:Numbers to generate in the vecto
    w: desired slope
    b: desired intercept
    random_scale = uniformm noise between the range |random_scale|

    """
    X = np.arange(num_gen) # Would create a vector with numbers from 1 to 200
    np.random.seed(707)
    deltas = np.random.uniform(low=-random_scale,high=random_scale,size=X.shape) #Noise
    # Imagine the next function M(x) = 2 + 2x, in here M(X) would takke the value of the vector Y for the values in X would generate Y, 
    Y = b + deltas + w * X
    return X, Y, w


def generate_data_1_d_uni(num_gen,w,b,random_scale = 10):
    """
    The objective of the function: Is to generate a vector X with the input numbers and generate a Y vector with the values to be predicted

    Args:
    
    num_gen:Numbers to generate in the vecto
    w: desired slope
    b: desired intercept
    random_scale = uniformm noise between the range |random_scale|

    """
    np.random.seed(707)
    X = np.random.uniform(0, 20, size=(200,)) # Would create a vector with numbers from 1 to 200
    np.random.seed(707)
    deltas = np.random.uniform(low=-random_scale,high=random_scale,size=X.shape) #Noise
    # Imagine the next function M(x) = 2 + 2x, in here M(X) would takke the value of the vector Y for the values in X would generate Y, 
    Y = deltas + w*X
    return X, Y, w


def generate_data_n_dim(size, num_features, noise):

    np.random.seed(707)
    X = np.random.uniform(-10,10,size = (size,num_features)) # Matrix with random elemenets between -10 and 10.
    slopes = np.random.uniform(-1,1,num_features) # for each feature will generate a random number for the 
    noise_term = np.random.normal(0,noise,size=size)
    Y = np.dot(X,slopes) + noise_term #functon to generate the vector 
    return X, Y, slopes



def plot_corr_matrix(X,Y):

    data = np.stack((X,Y))
    corr_matrix = np.corrcoef(data, rowvar=False)
    plt.figure(figsize= (8,4.5))
    sns.heatmap(corr_matrix,annot=True,cmap='coolwarm',fmt='.2f')
    plt.title("Feature and target Variable \n Correlation Matrix")
    plt.ylabel("Features and target")
    plt.xlabel("Features and target")
    plt.show                



def compute_predictons(X,slopes):

    #Que ocurre si son distintos
    random_scale = 2
    noise_term = np.random.uniform(low=-random_scale, high=random_scale, size=X.shape)#It woul genrate values by the quantity of the rows
    if X.ndim == 1:
        Y_pred = X *slopes + noise_term + 10
        return Y_pred
    elif X.ndim > 1:
        if X.shape[1] != len(slopes):
            return print(f"the features of the matrix {X.shape[1]}are different from vector with the slopes {len(slopes)}")
        else:
             Y_pred = X @ slopes + noise_term + 10
             return Y_pred
    else:
        print("Error no tiene dimensiones")
        return print(f"{X.ndim}")    

def compute_mse(Y,Y_pred):

    Y = np.array(Y)
    Y_pred = np.array(Y_pred)

    if len(Y) != len(Y_pred):
        return print(f"The observed values {len(Y)} are different form the predicted values {len(Y_pred)}")
    else: 
        mse = np.mean((Y - Y_pred)**2)
    return mse












