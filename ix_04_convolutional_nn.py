import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from datetime import datetime


def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    # Strides: no of pixels by which filter kernel moves (here: 1 in every direction)
    # Padding: how to fill missing values when kernel moves beyond picture boundaries
    # Input matrix x is multiplied with filter kernel W.
    # Optimiser will determine best values for W.
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    # Max pooling determines the maximum value within a certain area, defined by
    # a small matrix which moves within a larger one: here 2 * 2 moving by 2.
    # E.g. a 24 * 24 matrix would be reduced to a 12 * 12 matrix.
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


now = datetime.now()
GRAPH_LOG_FILE = 'tensorboard/ix_04_convolutional_nn/'+now.strftime("%Y%m%d-%H%M%S")+"/"

mnist = input_data.read_data_sets('MNIST/', one_hot=True)

with tf.name_scope('modelc'):
    # x is again a two-dimensional array: unknown number of pictures * picture size
    x = tf.placeholder(tf.float32, [None, 784])

    # Layer Convolution 1
    # First step: convert pictures in x into four-dimensional array: number of pictures,
    # height and width of pictures, number of colour channels (here 1 for monochrome)
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    # 5 * 5 pixel calculation window, 1 colour channel, 32 kernels
    # ML experience has shown that a single kernel would "learn" only one important feature.
    W_conv1 = weight_variable([5, 5, 1, 32])
    # Bias must have 32 elements, too = number of kernels
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # The whole convolutional layer could be created by a tf.layers method:
    # conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[5, 5], padding='same',
    #                          activation=tf.nn.relu)

    # Layer Pool 1
    # Used to calculate a "condensed" version of the layer
    h_pool1 = max_pool_2x2(h_conv1)

    # Layer Convolution 2
    # Same steps again like in first convolution layer
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # Layer Pool 2
    h_pool2 = max_pool_2x2(h_conv2)

    # Dense Layer
    # This layer corresponds to the regular NN layer in ix_03_two_layer_nn.py.
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1)+ b_fc1)

    # Dropout
    # Used to reduce overfitting: keep only a percentage of results
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    # Softmax
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

with tf.name_scope('train'):
    y_ = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y_conv))
    # Adam Optimizer: see Kingma and Ba, https://arxiv.org/abs/1412.6980
    # Adam: algorithm for first-order gradient-based optimization of stochastic objective functions
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.name_scope('eval'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as session:
    writer = tf.summary.FileWriter(GRAPH_LOG_FILE, session.graph)
    init = tf.global_variables_initializer()
    session.run(init)
    for i in range(2000):
        batch = mnist.train.next_batch(100)
        if i % 100 == 0:
            train_accuracy = session.run(accuracy, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print(i, train_accuracy)
        session.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.4})

    # Evaluation
    print(session.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
