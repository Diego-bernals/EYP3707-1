import numpy as np
import matplotlib.pyplot as plt
from get_data import compute_predictons,compute_mse


def gd_method(X,Y,num_epochs,learning_rate,epsilon):

    m,n = X.shape
    W = np.ones(n)

    weights = []
    losses = []

    for epoch in range(num_epochs):
        weights.append(W.copy())

        Y_pred = X @ W if X.ndim > 1 else X * W
        error = Y_pred - Y

        gradient = (2/m) * X.T @ error

        W = W - learning_rate * gradient
        
        Y_pred = X @ W if X.ndim > 1 else X * W
        loss = compute_mse(Y,Y_pred)
        losses.append(loss)

        if np.linalg.norm(gradient) < epsilon:
            print(F'Convergioen la iteracion {epoch}')
            break

    return np.array(weights),np.array(losses)



#Funcion exacta con el learning rate para funcion cuadratica entregada en 
def gd_method_exact_learning(X,Y,num_epochs,epsilon):

    m,n = X.shape
    W = np.ones(n)

    weights = []
    losses = []

    for epoch in range(num_epochs):
        weights.append(W.copy())
        
        Y_pred = X @ W
        error = Y_pred - Y
        gradient = (2/m) * X.T @ error

        direction = -gradient

        #Busqueda exacta del alpha

        numerator = error @ (X @ direction)
        denominator = (X @ direction) @ (X @ direction)
        alpha = numerator / denominator

        W_new = W + alpha *direction

        Y_pred_new = X @ W_new
        loss = compute_mse(Y, Y_pred_new)
        losses.append(loss)

        if np.linalg.norm(W_new-W) < epsilon:
            print(F'Convergioen la iteracion {epoch}')
            W = W_new
            break

        W = W_new

    return np.array(weights), np.array(losses)



def plot_gd(X,Y,weights, losses):
    
    plt.figure(1,figsize=[12,5])
    plt.subplot(121)
    plt.xlabel('x')
    plt.ylabel('y')

    if X.shape[1] == 2:
        x_plot = X[:, 1]
    else:
        x_plot = X[:, 0]

    plt.scatter(x_plot,Y,color= 'black',label = 'Y observations')

    num_epochs = len(weights)
    for i in range(num_epochs):
       W = weights[i]
       predictions = X @ W
       sorted_indices = np.argsort(x_plot)
       plt.plot(x_plot[sorted_indices],predictions[sorted_indices],label = f'Epoch {i}')
    
    plt.title('Predicciones por Iteración')
    plt.legend()

    plt.subplot(122)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.xticks(range(len(losses)))
    plt.plot(range(len(losses)), losses, marker='o', linestyle='--', color='black')
    plt.title('Evolución del error')
    plt.tight_layout()

