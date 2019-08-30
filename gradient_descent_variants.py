#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler


# In[ ]:


def do_adam(x, y, theta_init, step=0.001, maxsteps=20, precision=0.001, ):
    m = y.size # number of data points
    prev_v_w, gamma, eta = 0, 0.9, 1
    theta = theta_init
    history, costs, preds = [], [], [] # to store all weights
    m_w, m_b, v_w, v_b, eps, beta1, beta2, counter, oldcost = 0, 0, 0, 0, 1e-8, 0.9, 0.999, 0, 0
    
    pred = np.dot(x, theta)
    error = pred - y 
    currentcost = np.sum(error ** 2) / (2 * m)
    
    preds.append(pred)
    costs.append(currentcost)
    history.append(theta)
    counter+=1
    while abs(currentcost - oldcost) > precision:
        oldcost=currentcost
    
        m_w = beta1 * m_w + (1-beta1)*theta
        v_w = beta2 * v_w + (1-beta2) * theta**2
        
        m_w = m_w/(1-math.pow(beta1, counter+1))
        v_w = v_w/(1-math.pow(beta2, counter+1))
        
        theta = theta - (eta / np.sqrt(v_w + eps)) * m_w
        
        history.append(theta)
        pred = np.dot(x, theta)
        error = pred - y 
        currentcost = np.sum(error ** 2) / (2 * m)
        costs.append(currentcost)
        
        if counter % 5 == 0: preds.append(pred)
        counter+=1
        if maxsteps:
            if counter == maxsteps:
                break
        
    return history, costs, preds, counter


# In[ ]:


def do_rmsprop(x, y, theta_init, step=0.001, maxsteps=50, precision=0.001):
    m = y.size # number of data points
    theta = theta_init
    v_w, eps, beta1, eta = theta, 1e-8, 0.9, 1.0
    history, costs, preds, counter, oldcost = [], [], [], 0, 0 # to store all weights
    
    pred = np.dot(x, theta)
    error = pred - y 
    currentcost = np.sum(error ** 2) / (2 * m)
    
    preds.append(pred)
    costs.append(currentcost)
    history.append(theta)
    counter+=1
    while abs(currentcost - oldcost) > precision:
        oldcost=currentcost
        dw = x.T.dot(error)/m     
        
        v_w = beta1 * v_w + (1-beta1) * dw**2
        theta = theta - (eta / np.sqrt(v_w + eps)) * dw
             
        history.append(theta)
        currentcost = np.sum(error ** 2) / (2 * m)
        costs.append(currentcost)
        
        if counter % 5 == 0: preds.append(pred)
        counter+=1
        if maxsteps:
            if counter == maxsteps:
                break
        
        pred = np.dot(x, theta)
        error = pred - y
        
    return history, costs, preds, counter


# In[ ]:


def do_adagrad(x, y, theta_init, step=0.001, maxsteps=100, precision=0.001, ):
    m = y.size # number of data points
    theta = theta_init
    v_w, gamma, eta, eps, counter, oldcost = theta, 0.9, 1, 1e-8, 0, 0
    history, costs, preds = [], [], [] # to store all weights
    
    pred = np.dot(x, theta)
    error = pred - y 
    currentcost = np.sum(error ** 2) / (2 * m)
    
    preds.append(pred)
    costs.append(currentcost)
    history.append(theta)
    counter+=1
    while abs(currentcost - oldcost) > precision:
        oldcost=currentcost
        dw = x.T.dot(error)/m
        
        v_w = v_w + dw**2
        theta = theta - (eta / np.sqrt(v_w + eps)) * dw
        
        history.append(theta)
        currentcost = np.sum(error ** 2) / (2 * m)
        costs.append(currentcost)
        
        if counter % 5 == 0: preds.append(pred)
        counter+=1
        if maxsteps:
            if counter == maxsteps:
                break
        
        pred = np.dot(x, v_w)
        error = pred - y
        
    return history, costs, preds, counter


# In[ ]:


def do_nestorov_accelerated_gradient_descent(x, y, theta_init, step=0.001, maxsteps=10, precision=0.001, ):
    m = y.size # number of data points
    theta = theta_init
    prev_v_w, gamma, eta, counter, oldcost  = theta, 0.9, 1, 0, 0
    history, costs, preds = [], [], [] # to store all weights
    
    pred = np.dot(x, theta)
    error = pred - y 
    currentcost = np.sum(error ** 2) / (2 * m)
    
    preds.append(pred)
    costs.append(currentcost)
    history.append(theta)
    counter+=1
    while abs(currentcost - oldcost) > precision:
        oldcost=currentcost
        dw = x.T.dot(error)/m 
        
        v_w = gamma*prev_v_w + eta * dw # update
        theta = theta - v_w
        prev_v_w = v_w
        
        history.append(theta)
        currentcost = np.sum(error ** 2) / (2 * m)
        costs.append(currentcost)
        
        if counter % 5 == 0: preds.append(pred)
        counter+=1
        if maxsteps:
            if counter == maxsteps:
                break
        
        #do partial updates
        v_w = gamma*prev_v_w
        pred = np.dot(x, v_w)
        error = pred - y
        
    return history, costs, preds, counter


# In[ ]:


def do_momentum_gradient_descent(x, y, theta_init, step=0.001, maxsteps=10, precision=0.001, ):
    prev_v_w, gamma, eta, theta, counter, oldcost, m = 0, 0.9, 1, theta_init, 0, 0, y.size
    costs, history, preds = [], [], [] # to store all thetas 
    
    pred = np.dot(x, theta)
    error = pred - y 
    currentcost = np.sum(error ** 2) / (2 * m)
    
    preds.append(pred)
    costs.append(currentcost)
    history.append(theta)
    counter+=1
    while abs(currentcost - oldcost) > precision:
        oldcost=currentcost
        dw = x.T.dot(error)/m 
        
        v_w = gamma*prev_v_w + eta * dw # update
        theta = theta - v_w
        prev_v_w = v_w
        
        history.append(theta)
        
        pred = np.dot(x, theta)
        error = pred - y 
        currentcost = np.sum(error ** 2) / (2 * m)
        costs.append(currentcost)
        
        if counter % 5 == 0: preds.append(pred)
        counter+=1
        if maxsteps:
            if counter == maxsteps:
                break
        
    return history, costs, preds, counter


# In[ ]:


def do_gradient_descent(x, y, theta_init, step=0.001, maxsteps=10, precision=0.001, ):
    costs = []
    m = y.size # number of data points
    theta = theta_init
    history = [] # to store all thetas
    preds = []
    counter = 0
    oldcost = 0
    pred = np.dot(x, theta)
    error = pred - y 
    currentcost = np.sum(error ** 2) / (2 * m)
    preds.append(pred)
    costs.append(currentcost)
    history.append(theta)
    counter+=1
    while abs(currentcost - oldcost) > precision:
        oldcost=currentcost
        gradient = x.T.dot(error)/m 

        
        theta = theta - step * gradient  # update
        history.append(theta)
        
        pred = np.dot(x, theta)
        error = pred - y 
        currentcost = np.sum(error ** 2) / (2 * m)
        costs.append(currentcost)
        
        if counter % 5 == 0: preds.append(pred)
        counter+=1
        if maxsteps:
            if counter == maxsteps:
                break
        
    return history, costs, preds, counter


# In[ ]:


def do_stochastic_gradient_descent(x, y, theta_init, step=0.001, maxsteps=10, precision=0.001, ):
    costs = []
    m = y.size # number of data points
    theta = theta_init
    history = [] # to store all thetas
    preds = []
    counter = 0
    oldcost = 0
    pred = np.dot(x, theta)
    error = pred - y 
    currentcost = np.sum(error ** 2) / (2 * m)
    preds.append(pred)
    costs.append(currentcost)
    history.append(theta)
    counter+=1
    while counter < maxsteps :      
        for i in range(m):
            oldcost=currentcost
            pred = x[i] * theta
            error = pred - y[i]
            gradient = x[i]*(error)/m 
            
            #print("gradient: ", gradient)
            theta = theta - step * gradient  # update
            history.append(theta)
         
            currentcost = np.sum(error ** 2) / (2 * m)
            costs.append(currentcost)
        
            if counter % 5 == 0: preds.append(pred)
            counter+=1
            if maxsteps:
                if counter == maxsteps:
                    break
        
    return history, costs, preds, counter


# In[ ]:


def do_mini_batch_gradient_descent(x, y, theta_init, step=0.001, maxsteps=10, precision=0.001, ):
    costs = []
    m = y.size # number of data points
    theta = theta_init
    history = [] # to store all thetas
    preds = []
    counter, batch_size = 0, 3
    oldcost, gradient = 0, 0
    pred = np.dot(x, theta)
    error = pred - y 
    currentcost = np.sum(error ** 2) / (2 * m)
    preds.append(pred)
    costs.append(currentcost)
    history.append(theta)
    counter+=1
    while counter < maxsteps :      
        for i in range(m):
            oldcost=currentcost
            pred = x[i] * theta
            error = pred - y[i]
            gradient += x[i]*(error)/m  
            
            if counter%batch_size == 0:
                theta = theta - step * gradient  # update
                history.append(theta)
                currentcost = np.sum(error ** 2) / (2 * m)
                costs.append(currentcost)
                preds.append(pred)
                gradient = 0
        
            counter+=1
            if maxsteps:
                if counter == maxsteps:
                    break
        
    return history, costs, preds, counter


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import stats 
import math
from sklearn.datasets.samples_generator import make_regression 
from mpl_toolkits.mplot3d import Axes3D


x, y = make_regression(n_samples = 5000, 
                       n_features=1, 
                       n_informative=1, 
                       noise=20,
                       random_state=2017)


x = x.flatten()
slope, intercept, _,_,_ = stats.linregress(x,y)
best_fit = np.vectorize(lambda x: x * slope + intercept)
plt.plot(x,y, 'o', alpha=0.5)
grid = np.arange(-3,3,0.1)
plt.plot(grid,best_fit(grid), '.')


xaug = np.c_[np.ones(x.shape[0]), x]
theta_i = [-15, 40] + np.random.rand(2)
#history, cost, preds, iters = do_gradient_descent(xaug, y, theta_i, step=0.8)
#history, cost, preds, iters = do_stochastic_gradient_descent(xaug, y, theta_i, step=0.8)
#history, cost, preds, iters = do_mini_batch_gradient_descent(xaug, y, theta_i, step=0.8)
#history, cost, preds, iters = do_momentum_gradient_descent(xaug, y, theta_i, step=0.9)
history, cost, preds, iters = do_nestorov_accelerated_gradient_descent(xaug, y, theta_i, step=0.9)
#history, cost, preds, iters = do_adagrad(xaug, y, theta_i, step=9999999)
#history, cost, preds, iters = do_rmsprop(xaug, y, theta_i, step=0.001)
#history, cost, preds, iters = do_adam(xaug, y, theta_i, step=1000000)

theta = history[-1]
print("Gradient Descent: {:.2f}, {:.2f}, {:d}".format(theta[0], theta[1], iters))
print("Least Squares: {:.2f}, {:.2f}".format(intercept, slope))


def error(X, Y, THETA):
    return np.sum((X.dot(THETA) - Y)**2)/(2*Y.size)

ms = np.linspace(theta[0] - 20 , theta[0] + 20, 20)
bs = np.linspace(theta[1] - 40 , theta[1] + 40, 40)

M, B = np.meshgrid(ms, bs)

zs = np.array([error(xaug, y, theta) 
               for theta in zip(np.ravel(M), np.ravel(B))])
Z = zs.reshape(M.shape)

fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(M, B, Z, rstride=1, cstride=1, color='b', alpha=0.2)
#ax.contour(M, B, Z, 20, color='b', alpha=0.5, offset=0, stride=30)


ax.set_xlabel('x1', labelpad=30, fontsize=24, fontweight='bold')
ax.set_ylabel('x2', labelpad=30, fontsize=24, fontweight='bold')
ax.set_zlabel('f(x1,x2)', labelpad=30, fontsize=24, fontweight='bold')
ax.view_init(elev=20., azim=30)
ax.plot([theta[0]], [theta[1]], [cost[-1]] , markerfacecolor='r', markeredgecolor='r', marker='o', markersize=7)
ax.plot([history[0][0]], [history[0][1]], [cost[0]] , markerfacecolor='r', markeredgecolor='r', marker='o', markersize=7)


ax.plot([t[0] for t in history], [t[1] for t in history], cost , markerfacecolor='r', markeredgecolor='r', marker='.', markersize=2)
ax.plot([t[0] for t in history], [t[1] for t in history], 0 , markerfacecolor='r', markeredgecolor='r', marker='.', markersize=2)

fig.suptitle("Adam_Gradient_Descent", fontsize=24, fontweight='bold')
plt.savefig("Minimization_image.png")


# In[ ]:





# In[ ]:




