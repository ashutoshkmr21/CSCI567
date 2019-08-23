import numpy as np


def binary_train(X, y, loss="perceptron", w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: binary training labels, a N dimensional numpy array where 
    N is the number of training points, indicating the labels of 
    training data
    - loss: loss type, either perceptron or logistic
    - step_size: step size (learning rate)
	- max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of logistic or perceptron regression
    - b: scalar, which is the bias of logistic or perceptron regression
    """
    N, D = X.shape
    assert len(np.unique(y)) == 2


    w = np.zeros(D)
    if w0 is not None:
        w = w0
    
    b = 0
    if b0 is not None:
        b = b0
    np.place(y, y==0, -1)
    if loss == "perceptron":
        ############################################
        # TODO 1 : Edit this if part               #
        #          Compute w and b here            #
        w = np.zeros(D+1)
        b = 0
        a_ = np.array([1]*N)
        X = np.column_stack((a_,X))
        ############################################
        for i in range(max_iterations):
            new_mat = y*np.dot(X,w)
            k_new = np.array(new_mat)
            new_mat = np.where(new_mat <= 0, 1.0, 0.0)
            res = new_mat*y
            to_ = step_size*np.dot(X.T, res)
            w += to_

            # if k_new <= 0:
                # grad = (step_size*np.subtract(y, new_mat))/N
                # grad = np.dot(new_mat, X)
                # w += (step_size/N)*grad
                # w += np.dot(grad, X)
                # b += grad


            # if (y*(np.dot(w.T,X)+b)) <= 0:

            # if y*(np.sum(w*X, axis=-1)+self.b) <= 0:
            #     self.W += self.eta*y[i]*x[i, :]
            #     self.b += self.eta*y[i]
            # new_mat = np.dot(X,w) + b
            # new_mat = [1 if x > 0 else 0 for x in new_mat]
            # new_mat = np.array(new_mat)
            # step_change  = step_size * np.subtract(y, new_mat)
            # w += (np.dot(X.T,step_change)).T
            # b += step_change
            # for x, y_ in zip(X, y):
            #     val = check_prediction(x, w)
            #     step_change = step_size * (y_ - val)
            #     w += step_change * x
            #     b += step_change

    elif loss == "logistic":
        ############################################
        # TODO 2 : Edit this if part               #
        #          Compute w and b here            #
        w = np.zeros(D+1)
        b = 0
        a_ = np.array([1]*N)
        X = np.column_stack((a_,X))
        ############################################
        for i in range(max_iterations):
            # new_mat = np.dot(X,w)
            # predict = sigmoid(new_mat)
            # err = y - predict
            # grad = np.dot(X.T, err)
            # w += step_size*grad
            # step_change = np.dot(X.T, (predict-y))/y.size
            # w -= step_size*step_change
            new_mat = np.dot(X,w)
            predict = sigmoid(new_mat)
            res = -predict*y
            to_ = step_size*np.dot(X.T, res)
            w += to_
            # loss = -np.mean(y*np.log(predict)+(1-y))
        

    else:
        raise "Loss Function is undefined."

    b = w[0]
    w = w[1:]
    assert w.shape == (D,)
    return w, b

def check_prediction(x, w):
    w_ = np.dot(x,w)
    return np.where(w_ > 0.0, 1, 0)

def sigmoid(z):
    
    """
    Inputs:
    - z: a numpy array or a float number
    
    Returns:
    - value: a numpy array or a float number after computing sigmoid function value = 1/(1+exp(-z)).
    """

    ############################################
    # TODO 3 : Edit this part to               #
    #          Compute value                   #
    value = z
    ############################################
    value =  1/(1+np.exp(-value))
    return value

def binary_predict(X, w, b, loss="perceptron"):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of your learned model
    - b: scalar, which is the bias of your model
    - loss: loss type, either perceptron or logistic
    
    Returns:
    - preds: N dimensional vector of binary predictions: {0, 1}
    """
    N, D = X.shape
    
    if loss == "perceptron":
        ############################################
        # TODO 4 : Edit this if part               #
        #          Compute preds                   #
        preds = np.zeros(N)
        ############################################
        # np.sign((np.sum(X*w, axis=-1)+b))
        new_mat = np.dot(X,w) + b
        # new_mat = [1.0 if x > 0 else 0.0 for x in new_mat]
        # preds = np.array(new_mat)
        preds = np.array(np.where(new_mat >= 0.0, 1.0, 0.0))
        

    elif loss == "logistic":
        ############################################
        # TODO 5 : Edit this if part               #
        #          Compute preds                   #
        preds = np.zeros(N)
        ############################################
        new_mat = np.dot(X,w) + b
        sigm = sigmoid(new_mat)
        preds =sigm.round()
        

    else:
        raise "Loss Function is undefined."
    

    assert preds.shape == (N,) 
    return preds



def multiclass_train(X, y, C,
                     w0=None, 
                     b0=None,
                     gd_type="sgd",
                     step_size=0.5, 
                     max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of 
    training data
    - C: number of classes in the data
    - gd_type: gradient descent type, either GD or SGD
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: C-by-D weight matrix of multinomial logistic regression, where 
    C is the number of classes and D is the dimensionality of features.
    - b: bias vector of length C, where C is the number of classes
    """

    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0
    
    b = np.zeros(C)
    if b0 is not None:
        b = b0

    np.random.seed(42)
    if gd_type == "sgd":
        ############################################
        # TODO 6 : Edit this if part               #
        #          Compute w and b                 #
        w = np.zeros((C, D+1))
        b = np.zeros(C)
        a_ = np.array([1]*N)
        X = np.column_stack((X,a_))
        ############################################

        for i in range(max_iterations):
            # new_mat = np.dot(X,w.T)
            index = np.random.choice(N,1)
            X_n = X[index[0]]
            y_n = y[index[0]]
            new_mat = np.dot(w,X_n.T)
            p = get_softmax(new_mat)
            p[y_n] -= 1
            p = np.reshape(p,(C,1))
            X_n = np.reshape(X_n,(1,D+1))
            w -= step_size*(np.dot(p,X_n))
        b = w[:,-1]
        w = w[:,:D]
        

    elif gd_type == "gd":
        ############################################
        # TODO 7 : Edit this if part               #
        #          Compute w and b                 #
        w = np.zeros((C, D+1))
        b = np.zeros(C)
        a_ = np.array([1]*N)
        X = np.column_stack((a_,X))
        ############################################
        for i in range(max_iterations):
            vec=np.dot(w,X.T)
            p = get_softmax(vec)
            # smax = (np.exp(vec.T) / np.sum(np.exp(vec), axis=1)).T
            for index, val in enumerate(y):
                p[val][index] = p[val][index] - 1
            w -= (step_size/N)*np.dot(p,X)
        b = w[:,0]
        w = w[:,1:]
        

    else:
        raise "Type of Gradient Descent is undefined."
    

    assert w.shape == (C, D)
    assert b.shape == (C,)

    return w, b

def get_softmax(new_mat):
    # new_mat = np.dot(w,x_n.T)
    new_mat -= np.max(new_mat, axis=0)
    ex_m = np.exp(new_mat)
    smax = (ex_m / np.sum(ex_m, axis=0))
    return smax


def softmax(x):
    e = np.exp(x - np.max(x))
    if e.ndim == 1:
        return e / np.sum(e, axis=0)
    else:
        return e / np.array([np.sum(e, axis=1)]).T


def multiclass_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: weights of the trained multinomial classifier, C-by-D 
    - b: bias terms of the trained multinomial classifier, length of C
    
    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Outputted predictions should be from {0, C - 1}, where
    C is the number of classes
    """
    N, D = X.shape
    ############################################
    # TODO 8 : Edit this part to               #
    #          Compute preds                   #
    preds = np.zeros(N)
    ############################################
    vec=np.dot(X,w.T)
    vec=np.add(vec,b)
    smax = (np.exp(vec.T) / np.sum(np.exp(vec), axis=1)).T
    # vec1=np.exp(vec)
    # res=vec1.T/np.sum(vec1,axis=1)
    # preds =  res.T
    preds = smax.argmax(axis=1)
    assert preds.shape == (N,)

    return preds




