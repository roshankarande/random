from calendar import c
import struct
import numpy as np
import gzip
import numdifftools as nd

try:
    from simple_ml_ext import *
except:
    pass


def add(x, y):
    """ A trivial 'add' function you should implement to get used to the
    autograder and submission system.  The solution to this problem is in the
    the homework notebook.

    Args:
        x (Python number or numpy array)
        y (Python number or numpy array)

    Returns:
        Sum of x + y
    """
    return x + y

def relu(x):
    """ returns relu of X    
    Args:
        X (numpy Array - Vector)
    
    Returns:
        numpy array - max(0,x) elementwise
    """

    return np.maximum(x,0)

def normexp(X):
    """normalizes the exponential of the matrix row wise.

    Args:
        X (numpy Array - Matrix)
    
    Returns:
        row wise normalized exponential of the matrix
    
    """
    rows, cols = X.shape
    exp = np.exp(X)

    Z = exp / ( exp @ np.ones((cols,1)) )

    return Z

def one_hot_encode(v,classes=10):
    """
    creates a one hot encoded matrix from a vector 

    Args:
    v  (numpy Array - vector)
    classes (total number of classes)  # need because a batch might not have all the classes and then the encoding will be wrong if we try to use max(v) + 1

    Returns:
    One-hot row-wise encoded matrix  (numpy Array - Matrix )
    """
    n = len(v)
    Z = np.zeros((n, classes))
    Z[np.arange(n),v] = 1
    return Z



def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0. The normalization should be applied uniformly
                across the whole dataset, _not_ individual images.

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    with gzip.open(image_filename) as f:
        magic, size, nrows, ncols = struct.unpack('>IIII', f.read(16))
        buffer = f.read()
        data = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32) / 255.0
        X = data.reshape((size, nrows * ncols))

    with gzip.open(label_filename, 'rb') as f:
        magic, size = struct.unpack('>II', f.read(8))
        buffer = f.read()
        y = np.frombuffer(buffer, dtype=np.dtype(np.uint8))

    return X, y


def softmax_loss(Z, y):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (np.ndarray[np.float32]): 2D numpy array of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (np.ndarray[np.int8]): 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """

    n = len(y)
    logsumexp = np.log(np.sum(np.exp(Z), axis=1)) - Z[np.arange(n), y]
    return np.sum(logsumexp)/n


def softmax_regression_epoch(X, y, theta, lr=0.1, batch=100):
    """ Run a single epoch of SGD for softmax regression on the data, using
    the step size lr and specified batch size.  This function should modify the
    theta matrix in place, and you should iterate through batches in X _without_
    randomizing the order.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        theta (np.ndarrray[np.float32]): 2D array of softmax regression
            parameters, of shape (input_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """

    assert X.shape[1] == theta.shape[0]

    total_batches = len(X) // batch
    for i in range(total_batches):
        X_batch = X[i*batch:(i+1)*batch]
        y_batch = y[i*batch:(i+1)*batch]

        m,_ = X_batch.shape
        _, k = theta.shape

        Z = normexp(X_batch @ theta)        
        Iy = one_hot_encode(y_batch,classes = k)

        grad = 1/m * (X_batch.T @ ( Z - Iy ))
        theta -= lr * grad



def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).  It should modify the
    W1 and W2 matrices in place.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (np.ndarray[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (np.ndarray[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """

    assert X.shape[1] == W1.shape[0]
    assert W1.shape[1] == W2.shape[0]

    ### BEGIN YOUR CODE
    total_batches = len(X) // batch
    for i in range(total_batches):
        X_batch = X[i*batch:(i+1)*batch]
        y_batch = y[i*batch:(i+1)*batch]

        m,n = X_batch.shape
        n,d = W1.shape
        d,k = W2.shape

        Z2 = relu(X_batch @ W1)

        Iy = one_hot_encode(y_batch,classes=k)
        G2 = normexp(Z2 @ W2) - Iy

        _I = Z2.copy()   # indicator
        _I[_I > 0] = 1
        G1 = _I * (G2 @ W2.T)

        grad_W1 = 1/m * X_batch.T @ G1
        grad_W2 = 1/m * Z2.T @ G2

        W1 -= lr * grad_W1
        W2 -= lr * grad_W2
    ### END YOUR CODE


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h, y):
    """ Helper funciton to compute both loss and error"""
    return softmax_loss(h, y), np.mean(h.argmax(axis=1) != y)


def train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.5, batch=100,
                  cpp=False):
    """ Example function to fully train a softmax regression classifier """
    theta = np.zeros((X_tr.shape[1], y_tr.max() + 1), dtype=np.float32)
    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        if not cpp:
            softmax_regression_epoch(X_tr, y_tr, theta, lr=lr, batch=batch)
        else:
            softmax_regression_epoch_cpp(X_tr, y_tr, theta, lr=lr, batch=batch)
        train_loss, train_err = loss_err(X_tr @ theta, y_tr)
        test_loss, test_err = loss_err(X_te @ theta, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |" \
              .format(epoch, train_loss, train_err, test_loss, test_err))


def train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=500,
             epochs=10, lr=0.5, batch=100):
    """ Example function to train two layer neural network """
    n, k = X_tr.shape[1], y_tr.max() + 1
    np.random.seed(0)
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)

    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        nn_epoch(X_tr, y_tr, W1, W2, lr=lr, batch=batch)
        train_loss, train_err = loss_err(np.maximum(X_tr @ W1, 0) @ W2, y_tr)
        test_loss, test_err = loss_err(np.maximum(X_te @ W1, 0) @ W2, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |" \
              .format(epoch, train_loss, train_err, test_loss, test_err))

if __name__ == "__main__":
    
    X_tr, y_tr = parse_mnist("data/train-images-idx3-ubyte.gz",
                             "data/train-labels-idx1-ubyte.gz")
    X_te, y_te = parse_mnist("data/t10k-images-idx3-ubyte.gz",
                             "data/t10k-labels-idx1-ubyte.gz")

    print("Training softmax regression")
    train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.1)

    print("Training softmax regression - cpp")
    train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.1,cpp=True)

    print("\nTraining two layer neural network w/ 100 hidden units")
    train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=100, epochs=20, lr=0.2)




    #---------------------

   


# mugrade submit $YOUR_GRADER_KEY_HERE -k "nn_epoch"