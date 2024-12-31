import struct
import numpy as np
import gzip
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

    Return:
        Sum of x + y
    """
    ### BEGIN YOUR CODE
    return x + y
    pass
    ### END YOUR CODE


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
                maximum value of 1.0 (i.e., scale original values of 0 to 0.0 
                and 255 to 1.0).

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    with gzip.open(image_filename, 'rb') as img_f:
        img_f.read(4) #skip magic number
        num_images = int.from_bytes(img_f.read(4), 'big') # stored by high(big) endian
        rows = int.from_bytes(img_f.read(4), 'big')
        cols = int.from_bytes(img_f.read(4), 'big')
        
        image_data = img_f.read(num_images * rows * cols)
        X = np.frombuffer(image_data, dtype=np.uint8).astype(np.float32)
        X = X.reshape(num_images, rows * cols)
        X /= 255.0 # normalize to [0,1]
        
    with gzip.open(label_filename, 'rb') as lb_f:
        lb_f.read(4)
        num_labels = int.from_bytes(lb_f.read(4), 'big')
        
        lable_data = lb_f.read(num_labels)
        y = np.frombuffer(lable_data, dtype=np.uint8)
        
    return X, y
    pass
    ### END YOUR CODE


def softmax_loss(Z, y):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (np.ndarray[np.float32]): 2D numpy array of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (np.ndarray[np.uint8]): 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """
    ### BEGIN YOUR CODE
    y = Z[np.arange(len(y)), y]   # 我们需要的是正确分类位置的预测概率值而不是标签本身，所以需要Z把对应位置的值取出来，此行代码是按照x，y轴两个列表来取的
    y = np.expand_dims(y, axis= 1) # 第一步出来的结果是(batch, ), 需要在第1维度上加一个
    exp_Z = np.exp(Z)  
    sum_exp_Z = np.sum(exp_Z, axis=1, keepdims=True)  # 这里没有将reduce的维度去掉   (batch, 1)
    log_sum_exp_Z = np.log(sum_exp_Z)
    
    # print("log_sum_exp_Z shape:"+str(log_sum_exp_Z.shape))
    # print("y shape:"+str(y.shape))
    out = log_sum_exp_Z - y
    return np.mean(out)  # 最后出来一个值，而不是向量，所以取平均

    np.mean(-np.log(softmax(Z)[np.indices(y.shape)[0], y]))
    pass
    ### END YOUR CODE
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=1,keepdims=True)

def softmax_regression_epoch(X, y, theta, lr = 0.1, batch=100):
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
    ### BEGIN YOUR CODE
    for i in range(X.shape[0]//batch):
        x = X[0 + i * batch: (i + 1) * batch]
        y_b = y[0 + i * batch: (i + 1) * batch]
        z = softmax(np.matmul(x, theta))
        z[np.arange(batch),y_b] -= 1
        grad = np.matmul(x.transpose(),z)/batch
        theta -= lr * grad

    pass
    ### END YOUR CODE


def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
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
    ### BEGIN YOUR CODE
    # 梯度 W_i = Z_i(G_i+1 esm delta_i+1) 
    # G_i = G_i+1 delta_i+1 Wi
    for i in range(X.shape[0] // batch):
        x = X[0 + i * batch : (i+1) * batch]
        y_hat = y[0 + i * batch : (i+1) * batch]
        # forward
        Z2 = np.matmul(x,W1)
        np.maximum(0, Z2, Z2)
        delta_3 = softmax(np.matmul(Z2,W2))
        # backward
        delta_3[np.arange(batch), y_hat] -=1
        G2 = np.matmul(delta_3, W2.transpose())  # G2要带上W2，Gx都要带Wx
        G1 = np.multiply(np.where(Z2 >0, 1, Z2), G2)  #理论上来说G1应该带上W1，这里为了与讲义一致就不带
        
        W1 -= lr/batch * np.matmul(x.transpose(), np.multiply(np.where(Z2 >0, 1, Z2), G2))
        W2 -= lr/batch * np.matmul(Z2.transpose(), delta_3)
    pass
    ### END YOUR CODE



### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper funciton to compute both loss and error"""
    return softmax_loss(h,y), np.mean(h.argmax(axis=1) != y)


def train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.5, batch=100,
                  cpp=False):
    """ Example function to fully train a softmax regression classifier """
    theta = np.zeros((X_tr.shape[1], y_tr.max()+1), dtype=np.float32)
    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        if not cpp:
            softmax_regression_epoch(X_tr, y_tr, theta, lr=lr, batch=batch)
        else:
            softmax_regression_epoch_cpp(X_tr, y_tr, theta, lr=lr, batch=batch)
        train_loss, train_err = loss_err(X_tr @ theta, y_tr)
        test_loss, test_err = loss_err(X_te @ theta, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))


def train_nn(X_tr, y_tr, X_te, y_te, hidden_dim = 500,
             epochs=10, lr=0.5, batch=100):
    """ Example function to train two layer neural network """
    n, k = X_tr.shape[1], y_tr.max() + 1
    np.random.seed(0)
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)

    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        nn_epoch(X_tr, y_tr, W1, W2, lr=lr, batch=batch)
        train_loss, train_err = loss_err(np.maximum(X_tr@W1,0)@W2, y_tr)
        test_loss, test_err = loss_err(np.maximum(X_te@W1,0)@W2, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))



if __name__ == "__main__":
    X_tr, y_tr = parse_mnist("data/train-images-idx3-ubyte.gz",
                             "data/train-labels-idx1-ubyte.gz")
    X_te, y_te = parse_mnist("data/t10k-images-idx3-ubyte.gz",
                             "data/t10k-labels-idx1-ubyte.gz")

    print("Training softmax regression")
    # train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr = 0.1)
    train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr = 0.2, batch=100, cpp=True)

    # print("\nTraining two layer neural network w/ 100 hidden units")
    # train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=100, epochs=20, lr = 0.2)
