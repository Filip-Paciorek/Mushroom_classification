import numpy as np

def sigmoid(a):
    return 1/(1+np.exp(-a))
def cross_enthropy_loss(y,y_pred):
    n = len(y)
    loss = -(np.dot(y,np.log(y_pred)) + np.dot((1-y),np.log(1-y_pred)))/n
    return loss
def gradient_descent(X,y,batch,iter):
    lr = 0.001
    N,D = X.shape
    w = np.zeros(D)
    b = 0
    dataset_len = N//batch
    for _ in range(iter):
        for i in range(dataset_len):
            xbatch = X[(batch*i):(batch*(i+1))]
            ybatch = y[(batch*i):(batch*(i+1))]
            z = np.dot(xbatch,w)+b
            a = sigmoid(z)
            error = a - ybatch
            d_w = (xbatch @ error) / batch
            d_b = error.mean()
            w = w - (lr*d_w)
            b = b - (lr*d_b)

    

