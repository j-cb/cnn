from builtins import range
import numpy as np
import itertools


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    #print('the input x: ' + str(x))
    #print('Shape of the input x: ' + str(x.shape))
    xf = x.reshape(x.shape[0],-1)
    wf = w.reshape(w.shape[0],-1)
    #print('Shape of the flattened images: ' + str(xf.shape))
    #print('Shape of the weight matrix w: ' + str(w.shape))
    #print('Shape of the flattened weight matrix wf: ' + str(wf.shape))
    out = np.dot(xf,w) + b
    #print(out.shape)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    #print('Shape of the weight matrix w: ' + str(w.shape))
    xf = x.reshape(x.shape[0],-1)
    #print('Shape of the flattened images: ' + str(xf.shape))
    dwN = np.einsum('...i,...j->...ij',xf,dout)
    dw = np.sum(dwN, axis = 0)
    #print('Shape of dw: ' + str(dw.shape))
    dxD = np.einsum('ij,...j',w,dout)
    #print('Shape of dxD: ' + str(dxD.shape))
    dx = dxD.reshape(x.shape)
    #print('Shape of dx: ' + str(dx.shape))
    db = np.sum(dout, axis = 0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    out = np.where(x>0, x, 0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    drelu = np.where(x>0, 1, 0)
    #print('Shape of drelu: ' + str(drelu.shape))
    #print('Shape of dout: ' + str(dout.shape))
    dx = drelu*dout
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #######################################################################
        #print('x.shape = ' + str(x.shape)) 
        xmean = x.sum(axis = 0)/N
        xvar = ((x - xmean)**2).sum(axis = 0)/N
        #print('bn_param before: ' + str(bn_param))
        running_mean = momentum * running_mean + (1 - momentum) * xmean
        running_var  = momentum * running_var + (1 - momentum) * xvar
        #print('bn_param updated: ' + str(bn_param))
        #print('xmean = ' + str(xmean) + ' and xvar = ' + str(xvar))
        xnorm = (x - xmean)/np.sqrt(xvar+eps)
        xBN = gamma * xnorm + beta
        #print('xBN = ' + str(xBN) + ' and xnorm = ' + str(xnorm))
        out = xBN
        cache = (x, beta, gamma, xnorm, xmean, xvar, eps, N, D)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        #print('x.shape = ' + str(x.shape)) 
        #print('xmean = ' + str(xmean) + ' and xvar = ' + str(xvar))
        #print('bn_param: ' + str(bn_param))
        xnorm = (x - running_mean)/np.sqrt(running_var+eps)
        xBN = gamma * xnorm + beta
        #print('xBN = ' + str(xBN) + ' and xnorm = ' + str(xnorm))
        out = xBN
        cache = (x, beta, gamma)
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    ###########################################################################
    x, beta, gamma, xnorm, xmean, xvar, eps, N, D = cache
    dxnorm = dout*gamma
    #print('dout: ' + str(dout))
    #print('xnorm: ' + str(xnorm))
    #print('N: ' + str(N))
    dgamma = (dout*xnorm).sum(axis = 0)
    dbeta = dout.sum(axis = 0)
    r = 1/np.sqrt(xvar + eps)
    a = np.einsum('ij,k', np.identity(N), np.ones(D))*r
    #print(a)
    b = -1/N*np.einsum('ij,k', np.ones((N,N)),np.ones(D))*r
    #print('b: ' + str(b))
    c = np.einsum('jk,ik->jik',(x-xmean), (xmean - x))*r**3/N
    #print('c: ' + str(c))
    dxnormjdxi = a + b + c
    outjdxi = gamma * dxnormjdxi
    dx = np.einsum('jk, ijk -> ik', dout, outjdxi)
    #print('dx: ' + str(dx))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        mask = np.random.binomial(1, 1-p, x.shape)
        out = mask*x/(1-p)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        out = x
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        p = dropout_param['p']
        dx = dout*mask/(1-p)
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    #print('N, C, H, W = ', N, C, H, W)
    #print('F, C, HH, WW = ', F, C, HH, WW)
    stride, pad = conv_param['stride'], conv_param['pad']
    #print('stride, pad = ', stride, pad)
    A = 1 + (H + 2 * pad - HH) // stride
    B = 1 + (W + 2 * pad - WW) // stride
    xpadded = np.zeros((N,C,H+2*pad,W+2*pad))
    xpadded[:,:,pad:H+pad, pad:W+pad] = x
    #print(xpadded)
    xst = np.zeros((N,C,A,B,HH,WW))
    for i in range(A):
        for j in range(B):
            xst[:,:,i,j] = xpadded[:,:,i*stride:i*stride+HH, j*stride:j*stride+WW]
    #print(xst)
    out = np.einsum('ncabhw,fchw->nfab', xst,w) + b[None,:,None,None]
    #print(out)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    x, w, b, conv_param = cache
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    print('x.shape: ' + str(x.shape))    
    print('w.shape: ' + str(w.shape))
    print('dout.shape: ' + str(dout.shape))
    stride, pad = conv_param['stride'], conv_param['pad']
    A = 1 + (H + 2 * pad - HH) //stride
    B = 1 + (W + 2 * pad - WW) // stride
    Hp = H + 2*pad
    Wp = W + 2*pad
    dxpdx = np.zeros((Hp,Wp,H,W))
    for (hp,wp,h0,w0) in itertools.product(range(Hp),range(Wp),range(H),range(W)):
        dxpdx[hp,wp,h0,w0] = (hp - pad == h0) and (wp - pad == w0)
    dxstdxp = np.zeros((A,B,HH,WW, Hp,Wp))
    for (a,b,hh,ww,h0,w0) in itertools.product(range(A),range(B),range(HH),range(WW),range(Hp),range(Wp)):
        if h0 == a*stride + hh and w0 == b*stride + ww:
            dxstdxp[a,b,hh,ww,h0,w0] = 1
    doutdxst = np.zeros((A,B,A,B, F, C, HH, WW))
    for (a,b,z,y) in itertools.product(range(A),range(B),range(A),range(B)):
        doutdxst[a,b,z,y] = w * (a == z) * (b == y)
    #print(w)
    #print('shape of doutdxst: ', doutdxst.shape)
    #print('shape of dout: ', dout.shape)
    dxst = np.einsum('nfab,abzyfcHW->nzycHW',dout,doutdxst)
    #print(A,C)
    #print('shape of dxst: ','nzycHW', dxst.shape)
    #print('shape of dxstdxp: ', 'zyHWhw', dxstdxp.shape)
    dxp = np.einsum('nzycHW, zyHWhw->nchw',dxst, dxstdxp)
    #print('shape of dxp: ', 'nchw', dxp.shape)
    #print('shape of dxpdx: ', ' ', dxpdx.shape)
    dx = np.einsum('nchw,hwij->ncij',dxp,dxpdx)

    xpadded = np.zeros((N,C,H+2*pad,W+2*pad))
    xpadded[:,:,pad:H+pad, pad:W+pad] = x
    #print(xpadded)
    xst = np.zeros((N,C,A,B,HH,WW))
    for i in range(A):
        for j in range(B):
            xst[:,:,i,j] = xpadded[:,:,i*stride:i*stride+HH, j*stride:j*stride+WW]
    doutdw = xst[:,:,:,:]
    #print('doutdw: ' + str(doutdw))
    dw = np.einsum('nfhw,nchwzy->fczy',dout,xst[:,:,:,:])
    
    
    db = dout.sum(axis=(0,2,3))
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max pooling forward pass                            #
    ###########################################################################
    N, C, H1, W1 = x.shape
    FH = pool_param['pool_height']
    FW = pool_param['pool_width']
    S = pool_param['stride']
    H2 = (H1-FH)//S + 1
    W2 = (W1-FW)//S + 1
    out = np.zeros((N,C,H2,W2))
    for (i,j) in itertools.product(range(H2),range(W2)):
        out[:,:,i,j] = x[:,:,i*S:i*S+FH,j*S:j*S+FW].max(axis=(2,3))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max pooling backward pass                           #
    ###########################################################################
    x, pool_param = cache
    N, C, H1, W1 = x.shape
    FH = pool_param['pool_height']
    FW = pool_param['pool_width']
    S = pool_param['stride']
    H2 = (H1-FH)//S + 1
    W2 = (W1-FW)//S + 1
    dx = np.zeros(x.shape)
    outmask = {}
    for (n,c,i,j) in itertools.product(range(N),range(C),range(H2),range(W2)):
        outmask[(n,c,i,j)] = np.unravel_index(np.argmax(x[n,c,i*S:i*S+FH,j*S:j*S+FW]), x[n,c,i*S:i*S+FH,j*S:j*S+FW].shape)
    print('x: ', x[0,0,:4,:4])
    print('outmask: ')
    print(outmask[0,0,0,0], outmask[0,0,0,1])
    print(outmask[0,0,1,0], outmask[0,0,1,1])
    for (n,c,i,j) in itertools.product(range(N),range(C),range(H2),range(W2)):
        dx[n,c,i*S+outmask[(n,c,i,j)][0], j*S+outmask[(n,c,i,j)][1]] += dout[n,c,i,j]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    xnhwc = np.transpose(x, (0,2,3,1))
    xc = np.reshape(xnhwc, (-1, xnhwc.shape[-1]))
    xcbn, cache = batchnorm_forward(xc, gamma, beta, bn_param)
    out = np.transpose(xcbn.reshape(xnhwc.shape), (0,3,1,2))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    doutnhwc = np.transpose(dout, (0,2,3,1))
    doutc =  np.reshape(doutnhwc, (-1, doutnhwc.shape[-1]))
    dxb, dgamma, dbeta = batchnorm_backward(doutc, cache)
    dx = np.transpose(dxb.reshape(doutnhwc.shape), (0,3,1,2))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
