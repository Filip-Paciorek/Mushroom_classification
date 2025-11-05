import pandas as pd
import numpy as np
import sys
#import feature models
base_data = pd.read_csv('../data/processed/base_model_data')
max_ord_data = pd.read_csv('../data/processed/max_ordinal_data')
def train_test_split(X,y,size,state):
    N = len(X)
    n_test = int(N*size)
    indx = np.arange(N)
    np.random.seed(state)
    np.random.shuffle(indx)
    test_indx = indx[:n_test]
    train_indx = indx[n_test:]
    X_train = X.iloc[train_indx].copy()
    X_test = X.iloc[test_indx].copy()
    #X_train = (X_train - X_train.mean()) / X_train.std()
    #X_test = (X_test - X_train.mean()) / X_train.std()
    y_train = y.iloc[train_indx].copy()
    y_test = y.iloc[test_indx].copy()
    return X_train,X_test,y_train,y_test
def sigmoid(a):
    return 1/(1+np.exp(-a))
def predict(w,b,y):
    y_pred = pd.Series(y_pred)
    y_pred = y_pred.mask(y_pred >= 0.5,1)
    y_pred = y_pred.mask(y_pred < 0.5,0)
    return np.array(y_pred)
def cross_enthropy_loss(y,y_pred):
    n = len(y)
    loss = -((np.dot(y,np.log(y_pred))) + (np.dot((1-y),np.log(1-y_pred))))/n
    return loss
def gradient_descent(X,y,batch,iter):
    #set learning rate
    lr = 0.1
    #set w and b
    N,D = X.shape
    w = np.zeros(D)
    b = 0
    #prep for cutting into batches
    dataset_len = N//batch
    data_left = N % batch
    for _ in range(iter):
        y_pred = []
        for i in range(dataset_len):
            #take a batch
            xbatch = X[(batch*i):(batch*(i+1))]
            ybatch = y[(batch*i):(batch*(i+1))]
            #predict the values of y
            z = np.dot(xbatch,w)+b
            z = z.astype(float)
            a = sigmoid(z)
            #print(f"--- Batch {i} Debug ---")
            #print(f"Type of 'a': {type(a)}, Dtype of 'a': {a.dtype}")
            #print(f"Type of 'ybatch': {type(ybatch)}, Dtype of 'ybatch': {ybatch.dtype}")
            #print(f"Mean of 'a': {a.mean():.4f}")
            #print(f"Mean of 'ybatch': {ybatch.mean():.4f}")
            #break
            #calculate the error
            error = a - ybatch
            #optimize w and b
            d_w = (np.transpose(xbatch) @ error) / batch
            d_b = error.mean()
            w = w - (lr*d_w)
            b = b - (lr*d_b)
            y_pred = np.append(y_pred,a)
            #print(w,b)
            #print(f"Batch {i}: y_batch_mean={ybatch.mean():.2f}, d_b={d_b:.4f}, new_b={b:.4f}")
        if data_left:
            xbatch = X[len(y)-data_left:]
            ybatch = y[len(y)-data_left:]
            z = np.dot(xbatch,w)+b
            z = z.astype(float)
            a = sigmoid(z)
            error = a - ybatch
            d_w = (np.transpose(xbatch) @ error) / data_left
            d_b = error.mean()
            w = w - (lr*d_w)
            b = b - (lr*d_b)
            y_pred = np.append(y_pred,a)
        #y_pred = predict(y_pred)
        #print(y_pred)
        print(cross_enthropy_loss(y,y_pred))
    return w,b
X = base_data.drop(columns=['class'])
y = base_data['class'] 
X_train,X_test,y_train,y_test = train_test_split(X,y,0.2,100)
w,b = gradient_descent(X_train,y_train,256,10)
#predicted = predict(y_pred)
print(w,b)
print('---')
X = max_ord_data.drop(columns=['class'])
y = max_ord_data['class']
X_train,X_test,y_train,y_test = train_test_split(X,y,0.2,100)
w,b = gradient_descent(X_train,y_train,256,10)
#predict(y_pred)
print(w,b)
print('Feature distribution')
print(y_train.unique())
