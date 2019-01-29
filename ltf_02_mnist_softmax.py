import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

DATA_DIR = 'MNIST'
NUM_STEPS = 1000
MINIBATCH_SIZE = 100

# Load data set and store locally
# First argument: local directory, second: method for labelling the data
data = input_data.read_data_sets(DATA_DIR, one_hot=True)

# A Variable will be modified by the calculation, a placeholder must be filled before the calculation.
# Picture (x) is a placeholder: vector of dimension 784 (28x28), all pixels in a single row.
# W is a 784x10 matrix of weights, pre-filled with zeroes.
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))

# Real and predicted labels
# None as first argument because we do not specify a priori how many pictures we will use.
y_true = tf.placeholder(tf.float32, [None, 10])
y_pred = tf.matmul(x, W)

# Cross entropy as measure for similarity (loss function)
# softmax operations map lists/vectors of values to range [0..1].
# Also in tf.losses.softmax_cross_entropy
# softmax_cross_entropy_with_logits is deprecated, use v2 instead.
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=y_pred, labels=y_true))

# Use Gradient Descent in order to train the model
# 0.5 is the rate for adapting the weights.
gd_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Check model accuracy
correct_mask = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
# accuracy = fraction of correctly classified test examples
accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))

# Apply the graph in a session
# Session will automatically be closed at the end.
with tf.Session() as sess:

    # Train (training data: data.train)
    sess.run(tf.global_variables_initializer())

    # Run Gradient Descent with NUM_STEPS (1000) steps
    # MINIBATCH_SIZE = number of examples per step
    # _ is used to ignore a returned value - we merely run the loop,
    # we do not need the loop index.
    for _ in range(NUM_STEPS):
        batch_xs, batch_ys = data.train.next_batch(MINIBATCH_SIZE)
        # Placeholders are filled via feed_dict argument of sess.run():
        sess.run(gd_step, feed_dict={x: batch_xs,
                                     y_true: batch_ys})

    # Test (test data: data.test, not used for training)
    ans = sess.run(accuracy, feed_dict={x: data.test.images,
                                       y_true: data.test.labels})

#print(ans)
print("Accuracy: {:.5}%".format(ans*100))
