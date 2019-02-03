import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data", one_hot = True)

hl1nodes = 500
hl2nodes = 500
hl3nodes = 500

classes = 10
batch_size = 100

x = tf.placeholder('float',[None, 28*28])
y = tf.placeholder('float')


def neural_netork_model(data):
    
    hl1_layer = {
        'weights': tf.Variable(tf.random_normal([784, hl1nodes])),
        'biases': tf.Variable(tf.random_normal([hl1nodes]))
    }

    hl2_layer = {
        'weights': tf.Variable(tf.random_normal([hl1nodes, hl2nodes])),
        'biases':tf.Variable(tf.random_normal([hl2nodes]))
    }

    hl3_layer = {
        'weights': tf.Variable(tf.random_normal([hl2nodes, hl3nodes])),
        'biases':tf.Variable(tf.random_normal([hl3nodes]))
    }

    output_layer = {
        'weights': tf.Variable(tf.random_normal([hl3nodes, classes])),
        'biases':tf.Variable(tf.random_normal([classes]))
    }

    l1 = tf.add(tf.matmul(data, hl1_layer['weights']), hl1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hl2_layer['weights']), hl2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hl3_layer['weights']), hl3_layer['biases'])
    l3 = tf.nn.relu(l3)

    return tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])


def train_nn(x):
    
    prediction = neural_netork_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels= y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    total_epoch = 10
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(total_epoch):
            loss = 0
            for _ in range(int(mnist.train._num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c =  sess.run([optimizer, cost], feed_dict = { x: epoch_x, y: epoch_y })
                loss+=c
            print('Epoch: ', epoch, '/', total_epoch, '  loss: ', loss)
        
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy: ',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))
    
train_nn(x)