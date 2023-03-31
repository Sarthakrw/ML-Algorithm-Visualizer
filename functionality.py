import numpy as np
import matplotlib.pyplot as plt


def model_output(x_train, w, b, m):
    
    #initalising output array for model
    f_wb_array = np.zeros(m)
    
    # Convert w and b to NumPy arrays
    w_array = np.array(w)
    b_array = np.array(b)
    
    f_wb_array = w_array * x_train + b_array
        
    return f_wb_array



def cost_output(x_train, y_train, w, b, m):
    
    f_wb = model_output(x_train, w, b, m)
    
    # Compute the sum of the squared differences over all training examples
    cost_tmp = np.sum((f_wb - y_train) ** 2)
    
    final_cost = cost_tmp / (2 * m)
    
    return final_cost




def gradient_output(x_train, y_train, w, b, m):
    
    dj_dw = 0
    dj_db = 0
    
    f_wb = model_output(x_train, w, b, m)
    
    for i in range(m):
        dj_dw += (f_wb[i] - y_train[i]) * x_train[i]
        dj_db += (f_wb[i] - y_train[i])
        
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    
    return dj_dw, dj_db


def gradient_descent_output(x_train, y_train, w_init, b_init, alpha, iterations, m):
    
    w = w_init
    b = b_init
    
    for i in range(iterations):
        
        dj_dw, dj_db = gradient_output(x_train, y_train, w, b, m)
        
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        
    return w,b