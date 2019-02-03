import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data", one_hot = True)

no_classes = 10
batch_size = 128

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def convolutional_neural_network(x);
    weights = {
        'W_conv1': tf.Variable(tf.random_normal([5,5,1,32])),
        'W_conv2': tf.Variable(tf.random_normal([5,5,32,64])),
        'W_fc': tf.Variable(tf.random_normal([7*7*64, 1024])),
        'out': tf.Variable(tf.random_normal([1024, no_classes))
    }
    
    biases = {
        'b_conv1': tf.Variable(tf.random_normal([32])),
        'b_conv2': tf.Variable(tf.random_normal([64])),
        'b_fc': tf.Variable(tf.random_normal([1064]),
        'out': tf.Variable(tf.random_normal([1024, no_classes]),
    }

    x = tf.reshape(x, shape = [-1, 28, 28, 1])

    return output

def train_neural_network(x):
    prediction = 