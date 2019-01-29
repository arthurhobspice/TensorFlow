import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from datetime import datetime


def weight_variable(shape):
    # Initialise weights with normal distribution
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    # Initialise biases with constant 0.1
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


now = datetime.now()
GRAPH_LOG_FILE = 'tensorboard/ix_03_two_layer_nn/'+now.strftime("%Y%m%d-%H%M%S")+"/"

mnist = input_data.read_data_sets('MNIST/', one_hot=True)

with tf.name_scope('model'):
    # Two-dimensional array: unknown number of pictures * picture size
    x = tf.placeholder(tf.float32, [None, 784])

    # Layer 1 - 784 = 28*28 (picture size)
    W1 = weight_variable([784, 100])
    b1 = bias_variable([100])
    # Rectified linear unit: 0 if input < limit (0), linearly increasing if input > limit
    y1 = tf.nn.relu(tf.matmul(x, W1) + b1)

    # Layer 2
    W2 = weight_variable([100, 10])
    b2 = bias_variable([10])
    y2 = tf.nn.softmax(tf.matmul(y1, W2) + b2)

    # Output
    y = y2

with tf.name_scope('train'):
    y_ = tf.placeholder(tf.float32, [None, 10])
    # Cross entropy: S_p(q) = sum (over x) of q(x)*ld(1/p(x))
    # (Compare with entropy: S(p) = sum (over x) p(x)*ld(1/p(x)) )
    # CE becomes smaller as numbers are more similar. In other words: the more different p is
    # from q, the more the cross-entropy of q with respect to p will be bigger than the entropy of q.
    # Here CE compares the current calculated result y for a given x with the actual result y_.
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

with tf.Session() as session:
    writer = tf.summary.FileWriter(GRAPH_LOG_FILE, session.graph)
    init = tf.global_variables_initializer()
    session.run(init)
    for _ in range(1000):
        # In every training loop, pick any 100 pictures randomly
        batch_xs, batch_ys = mnist.train.next_batch(100)
        session.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # Evaluation
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(session.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
