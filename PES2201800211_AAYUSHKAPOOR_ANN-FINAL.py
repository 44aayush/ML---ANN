#!/usr/bin/env python
# coding: utf-8

# In[138]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[139]:


#To import the dataset
df = pd.read_csv('LBW_Dataset.csv')
df.head()


# In[140]:


df.info()


# In[141]:


df.shape


# In[142]:


#To check missing values
df.isna().sum()


# In[143]:


#To drop data with missing values
df.dropna(inplace=True)


# In[144]:


df.isna().sum()


# In[145]:


df.describe()


# In[146]:


#To check correlation between data along with a heatmap
corr=df.corr()
plt.figure(figsize=(14,6))
sns.heatmap(corr,annot=True)


# In[147]:


X = df[['Community','Age','Weight','Delivery phase','HB','IFA','BP','Residence']]
y = df['Result']
y.unique()


# In[148]:


#Split data into train and test sets

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)


# In[149]:


#To normalize data

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test  = sc.transform(X_test)


# In[150]:


#X_train = X_train.to_numpy()
y_train = y_train.to_numpy().reshape(y_train.shape[0],1)
#X_test = X_test.to_numpy()
y_test = y_test.to_numpy().reshape(y_test.shape[0],1)


# In[151]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[152]:


X_train = X_train.T
y_train = y_train.T
X_test = X_test.T
y_test = y_test.T
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[153]:


#Define sigmoid function and its gradient

def sigmoid(s):
    return 1/(1 + np.exp(-s))

def sigmoid_derv(s):
    return s * (1 - s)


# In[154]:


#To obtain layer sizes

def layer_sizes(X, Y, n_h):
    
    n_x = X.shape[0]
    n_y = Y.shape[0]
    
    return (n_x, n_h, n_y)


# In[155]:


#Initialize parameters. Weights are randomly initialized whereas biases are initialized to zero

def initialize_parameters(n_x, n_h, n_y):
    
    np.random.seed(2)
    
    W1 = np.random.randn(n_h, n_x) *0.01
    b1 = np.zeros((n_h , 1))
    W2 = np.random.randn(n_y, n_h) *0.01
    b2 = np.zeros((n_y,1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters


# In[156]:


#Forward propagation of the data

def forward_propagation(X, parameters):

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    Z1 = np.dot(W1, X) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    
    A_and_Z = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, A_and_Z


# In[157]:


#To compute the binary cross entropy from predicted and real values of y

def compute_cost(A2, Y):

    m = Y.shape[1] # number of example

    
    logprobs = np.multiply(np.log(A2), Y) + np.multiply((1-Y), np.log(1-A2))
    cost = -1/m * np.sum(logprobs) 
    
    
    cost = float(np.squeeze(cost))  
    assert(isinstance(cost, float))
    
    return cost


# In[158]:


#Backward propagation

def backward_propagation(parameters, cache, X, Y):
    
    m = X.shape[1]
    
    W1 = parameters["W1"]
    W2 = parameters["W2"]
        
    A1 = cache["A1"]
    A2 = cache["A2"]
    
    dZ2= A2 - Y
    dW2 = (1/m) * np.dot(dZ2, A1.T) 
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.multiply(np.dot(W2.T, dZ2), sigmoid_derv(A1) )
    dW1 = (1/m) * np.dot(dZ1, X.T)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
    
    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads


# In[159]:


#To update parameters after 1 pass of forward and backward propagation

def update_parameters(parameters, grads, learning_rate = 1):
    
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    
    W1 = W1 - learning_rate*dW1
    b1 = b1 - learning_rate*db1
    W2 = W2 - learning_rate*dW2
    b2 = b2 - learning_rate*db2
    
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters


# In[174]:


def ANN(X, Y, n_h, num_iterations = 100):
    
    np.random.seed(3)
    n_x = layer_sizes(X, Y, n_h)[0]
    n_h = layer_sizes(X, Y, n_h)[1]
    n_y = layer_sizes(X, Y, n_h)[2]
    
    
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    for i in range(0, num_iterations):
         
        A2, cache = forward_propagation(X, parameters)
 
        cost = compute_cost(A2, Y)
        cost_list.append(cost*1000)
 
        grads = backward_propagation(parameters, cache, X, Y)

        parameters = update_parameters(parameters, grads, 0.1)
        
       # print ("Cost after iteration %i: %f" %(i, cost))

    return parameters


# In[185]:


#Plot graph of cost vs epoch

cost_list=[]
iters= 1400
it = range(iters)
parameters = ANN(X_train, y_train, 15, iters)
plt.plot(it,cost_list)
plt.xlabel('epoch')
plt.ylabel('cost')


# In[186]:


#To check training accuracy

A2, cache = forward_propagation(X_train,parameters)
np.set_printoptions(threshold=np.inf)
train_predictions = np.zeros((A2.shape[0], A2.shape[1]))
train_predictions[A2>0.5]=1

print ("training accuracy = ", (np.sum(train_predictions == y_train))/ y_train.shape[1] * 100)


# In[187]:


#To check test accuracy

A2, cache = forward_propagation(X_test,parameters)
np.set_printoptions(threshold=np.inf)
test_predictions = np.zeros((A2.shape[0], A2.shape[1]))
test_predictions[A2>0.5]=1

print ("testing accuracy = ", (np.sum(test_predictions == y_test))/ y_test.shape[1] * 100)


# In[ ]:




