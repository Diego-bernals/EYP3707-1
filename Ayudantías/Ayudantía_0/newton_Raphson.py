import numpy as np
import matplotlib.pyplot as plt
from get_data import compute_predictons,compute_mse



def newton_method(X,Y, num_epochs,epsilon):
  
 m,n = X.shape
 W = np.ones(n)

 weights = []
 losses = []

 for epoch in range(num_epochs):
    weights.append(W.copy())

    if X.ndim == 1:
        Y_pred =  X * W
    else:
        Y_pred = X @ W


    error = Y_pred - Y

    gradient = (2/m) * X.T @ error
    hessian = (2/m) * X.T @ X

    direction = np.linalg.solve(hessian, gradient)

    W = W - direction
    
    if X.ndim == 1:
        Y_pred =  X * W
    else:
        Y_pred = X @ W

    loss = compute_mse(Y, Y_pred)
    losses.append(loss)

    if np.linalg.norm(direction) < epsilon:
      print(F'Convergio en la iteración {epoch}')
      break

 return np.array(weights), np.array(losses)



def plot_newton_method(X,Y,weights, losses):
   
    plt.figure(1,figsize=[12,5])
    plt.subplot(121)
    plt.xlabel('x')
    plt.ylabel('y')

    if X.shape[1] == 2:
       plt.xticks(X[:,1])
       x_plot = X[:,1]
    else:
       plt.cticks(X[:,0])
       x_plot = X[:,0]
    plt.scatter(x_plot,Y,color= 'black',label = 'Y obervations')

    num_epochs = len(weights)
    for i in range(num_epochs):
       W = weights[i]
       predictions = X @ W
       plt.plot(x_plot,predictions,label = f'Epoch {i}')
    plt.title('Predicciones por Iteración')
    plt.legend()

    plt.subplot(122)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.xticks(range(len(losses)))
    plt.plot(range(len(losses)), losses, marker='o', linestyle='--', color='black')
    plt.title('Evolución del error')
    plt.tight_layout()



