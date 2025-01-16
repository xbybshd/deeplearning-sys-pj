from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import struct
import gzip
import sys


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        self.transforms = transforms
        self.X, self.y = self.parse_mnist(image_filename, label_filename)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        x = self.apply_transforms(self.X[index].reshape(28, 28, -1))
        return x.reshape(-1, 28*28), self.y[index]
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.X.shape[0]
        ### END YOUR SOLUTION

    def parse_mnist(self, image_filesname, label_filename):
        """Read an images and labels file in MNIST format.  See this page:
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
                    maximum value of 1.0.

                y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                    labels of the examples.  Values should be of type np.int8 and
                    for MNIST will contain the values 0-9.
        """
        ### BEGIN YOUR SOLUTION
        with gzip.open(image_filesname, 'rb') as img_f:
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
        ### END YOUR SOLUTION`