import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xlrd
from datetime import datetime

DATA_FILE = 'input/prices.xls'
# Create a separate log directory for every run (with timestamp)
# Tensorboard will automatically load all runs, user can filter them.
now = datetime.now()
GRAPH_LOG_FILE = 'tensorboard/ix_02_linear_reg/'+now.strftime("%Y%m%d-%H%M%S")+"/"


# Read data (Excel reader)
def read_data(filename):
    book = xlrd.open_workbook(filename, encoding_override='utf-8')
    sheet = book.sheet_by_index(0)
    x_data = np.asarray([sheet.cell(i, 1).value for i in range(1, sheet.nrows)])
    y_data = np.asarray([sheet.cell(i, 2).value for i in range(1, sheet.nrows)])
    return x_data, y_data


x_data, y_data = read_data(DATA_FILE)

# Plot the input data
plt.plot(x_data, y_data, 'ro', label='Original data')
plt.legend()
plt.show()

# Define model
# name_scope('model') summarises all aspects of model under this name.
with tf.name_scope('model'):
    W = tf.Variable([2.5], name='Weights')
    b = tf.Variable([1.0], name='biases')
    y = W * x_data + b

# Training graph
with tf.name_scope('train'):
    # reduce_mean returns a scalar: the average of all vector elements.
    loss = tf.reduce_mean(tf.square(y - y_data), name='loss')
    # Contents of variable loss shall be written to the graph log, name "loss":
    tf.summary.scalar('loss',loss)
    # Argument of GradientDescentOptimizer: step width = learning_rate
    optimizer = tf.train.GradientDescentOptimizer(0.0001)
    train = optimizer.minimize(loss)

# Learn
# Query current values of all observed variables in one operation:
summary_op = tf.summary.merge_all()
with tf.Session() as session:
    writer = tf.summary.FileWriter(GRAPH_LOG_FILE, session.graph)
    init = tf.global_variables_initializer()
    session.run(init)

    # Training
    for i in range(10000):
        # The _ is used to ignore a returned value.
        # summary_op is executed in every loop run (together with a training step "train"),
        # add_summary writes the result to the graph log.
        summary, _ = session.run([summary_op, train])
        writer.add_summary(summary, i)

    # Assessment
    curr_W, curr_b, curr_loss = session.run([W, b, loss])
    print("W = %s, b = %s, loss = %s"%(curr_W, curr_b, curr_loss))

writer.close()
