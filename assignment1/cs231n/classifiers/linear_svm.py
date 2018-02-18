import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero
  dWn = np.zeros(W.shape) # initialize the gradient as zero
  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,y[i]] -= X[i]
        dWn[:,y[i]] += X[i]
        dW[:,j] += X[i]
  print('naive big dWn' + str(dWn[0]) + '\n')
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  #print('naive_loss: ' + str(loss) + '\n' )
  print('naive big dW' + str(dW[0]) + '\n')
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  dW /= num_train
  #print(dW[:3,:5])
    
  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = X.dot(W)
  #print(np.shape(X), np.shape(W))
  #print('scores: ' + str(scores) + '\n' )
  scores_correct = scores[np.arange(len(scores)),y]
  score_diffs_p1 = (scores - scores_correct.reshape(len(scores_correct),1) ) + 1
  score_diffs_p1[np.arange(len(score_diffs_p1)),y] = -10
  losses_all = np.maximum(score_diffs_p1, 0)
  losses_per_image = np.sum(losses_all, axis = 1)
  loss_mean = np.mean(losses_per_image)
  loss = loss_mean + reg * np.sum(W * W)   
  #print('vec_loss: ' + str(loss) + '\n' )
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  
  print('X[:,0]: ' + str(X[:,0]) + '\n' )
  print('y: ' + str(y) + '\n' )
  #print('scores: ' + str(scores) + '\n')
  #print('score_diffs_p1: ' + str(score_diffs_p1) + '\n')
  #print('losses_all: ' + str(losses_all) + '\n')
  
  has_loss = (losses_all > 0) 
  print('has_loss: ' + str(has_loss) + '\n' )
  #mult_loss = np.sum(has_loss, axis = 1)
  pos_grad = (X.T).dot(has_loss)
  
  #print('mult_loss: ' + str(mult_loss) + '\n' )
  print('pos_grad: ' + str(pos_grad[0]) + '\n' )
  # print('X: ' + str(X) + '\n' )
  
  corrects = np.array([]).reshape(0,np.shape(y)[0])
  for a in range(np.shape(W)[1]):
    corrects = np.vstack([corrects, y == a])
  corrects = corrects.T
  print('corrects: ' + str(corrects) + '\n' )
  neg_grad = (X.T).dot(corrects)
  #neg_grad *= has_loss
  neg_grad *= np.shape(W)[1]-1
  print('neg_grad: ' + str(neg_grad[0]) + '\n' )
  
  #print(np.shape(corrects))
  #print(np.shape(X))
  #print(corrects[:1])
  #print(X[:5,0])
  #print(neg_grad[:2,0])
  
  dW = pos_grad - neg_grad
  print('big dW: ' + str(dW[0]) + '\n' )
  dW /= X.shape[0]
  print(dW[:3,:5])
  
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
