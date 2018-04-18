import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_train = len(X)
    dim, classes = W.shape
    scores = X.dot(W)
    dscores = np.zeros_like(scores)
    
    for i in range(num_train):        
        for j in range(classes):
            denr = np.sum(np.exp(scores[i]- scores[i][j]))
            dscores[i][j] = 1 / denr
            if(j == y[i]):
                dscores[i][j] -= 1
        loss_i = -scores[i][y[i]] + np.log(np.sum(np.exp(scores[i])))
        loss += loss_i            
    
    loss /= num_train
    loss += reg * np.sum(W*W)
    
    dW = np.dot(X.T, dscores)
    dW /= num_train
    dW += 2 * reg * W    
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_train = len(X)
    dim, classes = W.shape
    scores = X.dot(W)
    dscores = np.zeros_like(scores)
    
    loss = (-1) * np.choose(y, scores.T) + np.log(np.sum(np.exp(scores), axis = 1))    
    loss = np.sum(loss)    
    loss /= num_train
    loss += reg * np.sum(W*W)    
    
    dscores = np.exp(scores)
    scores_sum = dscores.sum(axis=1)
    dscores = dscores / scores_sum[:, np.newaxis]
    dscores[ np.arange(num_train), y] -= 1    
    dW = np.dot(X.T, dscores)
    dW /= num_train
    dW += 2 * reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW

