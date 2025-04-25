import numpy as np
import itertools #combinaciones
from tqdm import tnrange


#Funciones utilies para el calculo de los criterios
def rss(y_true,y_pred):
    return np.sum((y_true - y_pred)**2)

def r_squared(y_true,y_pred):
    ss_total = np.sum((y_true - np.mean(y_true))**2)
    ss_red = rss(y_true, y_pred)
    return 1 - (ss_red / ss_total)


#Mestreo estratificado porporcional
def stratified_sampling(X,Y,clase, proporcion):

    unique_classes, n = np.unique(clase,return_counts=True)
    total = len(clase)
    proporciones = n/total


    n_train = int(total * proporcion)

    X_train_list, Y_train_list = [], []
    X_test_list, Y_test_list = [], []

    for i, clase_a in enumerate(unique_classes):

        idx = np.where(clase == clase_a)[0]

        n_train_clase = int(round(proporciones[i]*n_train))

        idx_sample = np.random.choice(idx,size=len(idx), replace=False)
        idx_train = idx_sample[:n_train_clase]
        idx_test = np.setdiff1d(idx, idx_train)

        X_train_list.append(X[idx_train])
        Y_train_list.append(Y[idx_train])
        X_test_list.append(X[idx_test])
        Y_test_list.append(Y[idx_test])

    X_train = np.vstack(X_train_list)
    Y_train = np.concatenate(Y_train_list)
    X_test = np.vstack(X_test_list)
    Y_test = np.concatenate(Y_test_list)    

    return X_train,Y_train, X_test, Y_test



#Modelo de OLS
def ols(X,Y):
    #añadimos la matriz de 1's
    if not np.all(X[:,0] == 1):
        X = np.column_stack((np.ones(X.shape[0]),X))
    #\beta = (X^{T} \cdot X)^{-1} * X^{T} *Y 
    betas = np.linalg.pinv(X.T @ X) @ X.T @ Y
    #\hat{y} = X * \beta  
    predicted = X @ betas

    return betas, predicted






def sub_set_selection(X,Y):
    variables = X.shape[1]
    rss_list = []
    r_squared_list = []
    variables_list = []
    num_var = []

    for k in tnrange(1, variables + 1, desc = "Looping over subsets"):
        for combo in itertools.combinations(range(variables),k):
            X_subset = X[:,combo]
            betas, y_pred = ols(X_subset,Y)
            rss_list.append(rss(Y,y_pred))
            r_squared_list.append(r_squared(Y,y_pred))
            variables_list.append(combo)
            num_var.append(k)

    results = {
        'num_variables': np.array(num_var),
        'RSS': np.array(rss_list),
        'R_squared': np.array(r_squared_list),
        'variables': variables_list
    }

    return results




def forward_selection(X,Y):
    variables = X.shape[1]
    all_variables = list(range(variables))
    selected_var = []
    reman_var = all_variables.copy()

    rss_list = []
    r_squared_list = []
    variables_list = []

    for i in range(1,variables + 1):
        best_rss = np.inf
        best_feature = None
        best_r2 = None
        best_model = None

        for combo in reman_var:
            combo_var = selected_var + [combo]
            X_subset = X[:,combo_var]
            _, y_pred = ols(X_subset,Y)
            current_rss = rss(Y,y_pred)
            current_r2 = r_squared(Y,y_pred)

            if current_rss < best_rss:
                best_rss = current_rss
                best_r2 = current_r2
                best_feature = combo
                best_model = combo_var
        
        selected_var.append(best_feature)
        reman_var.remove(best_feature)

        rss_list.append(best_rss)
        r_squared_list.append(best_r2)
        variables_list.append(best_model.copy())

    results = {
        'num_variables': np.arange(1,len(rss_list)+1),
        'RSS': np.array(rss_list),
        'R_squared': np.array(r_squared_list),
        'variables': variables_list
    }

    return results




def backward_selection(X,Y):
    variables = X.shape[1]
    selected_var = list(range(variables))

    rss_list = []
    r_squared_list = []
    variables_list = []


    for i in range(variables,1,-1):
        best_rss = np.inf
        best_var_rem = None
        best_r2 = None
        best_model = None

        for var in selected_var:
            combo_var = selected_var.copy()
            combo_var.remove(var)
            X_subset = X[:,combo_var]
            _,y_pred = ols(X_subset,Y)
            current_rss = rss(Y,y_pred)
            current_r2 = r_squared(Y,y_pred)

            if current_rss < best_rss:
                best_rss = current_rss
                best_r2 = current_r2
                best_var_rem = var
                best_model = combo_var


        selected_var.remove(best_var_rem)

        rss_list.append(best_rss)
        r_squared_list.append(best_r2)
        variables_list.append(best_model.copy())

    results = {
        'num_variables': np.arange(variables - 1, -1, -1), 
        'RSS': np.array(rss_list),
        'R_squared': np.array(r_squared_list),
        'variables': variables_list
    }


    return results















