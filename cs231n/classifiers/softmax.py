import numpy as np
from random import shuffle

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
  score = X.dot(W)
  for i in xrange(X.shape[0]):
      score[i,:] -= np.max(score[i,:]) #normalization to avoid numeric instability
      exp_all = np.exp(score[i,:])
      exp = exp_all[y[i]]/np.sum(exp_all)
      loss -= np.log(exp)
      for j in xrange(W.shape[1]):
          dW[:,j] += (exp_all[j]/np.sum(exp_all)-(j==y[i]))*X[i,:]
   
  loss = loss/X.shape[0] + 0.5*np.sum(W*W)*reg
  dW = dW/X.shape[0]+reg*W
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
  train_num = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  tmp = np.arange(train_num)
  score = X.dot(W)
  c = np.zeros_like(score)
  c[np.arange(train_num),y] = -1
  score_exp = np.exp(score) 
  softmax = score_exp[tmp,y]/np.sum(score_exp, axis=1)
  loss = -np.log(softmax)
  loss = np.sum(loss)/train_num + 0.5*reg*np.sum(W*W)

  dW = np.transpose(X).dot(score_exp/np.sum(score_exp,axis=1)[:,np.newaxis]+c)
  dW = dW/train_num + reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

