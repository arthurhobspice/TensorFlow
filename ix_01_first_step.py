import tensorflow as tf

# Create graph
node_const_1 = tf.constant(3.0, tf.float32, name = 'const_1')
node_const_2 = tf.constant(4.0, tf.float32, name = 'const_2')

node_add = tf.add(node_const_1, node_const_2)

a = tf.placeholder(tf.float32, name = 'a')
node_mul = node_add * a

b = tf.placeholder(tf.float32, name = 'b')
node_sub = node_mul - b

# Execute session
with tf.Session() as session:
    # tensorboard will only show what the Python code has written into the logs.
    # FileWriter does this job.
    # Start tensorboard like this: tensorboard --logdir=./first_step (Windows Firewall must be opened)
    # Seems like tensorboard has problems starting when python executable is installed under
    # C:\Program Files (or any other directory whose name contains spaces).
    writer = tf.summary.FileWriter('tensorboard/first_step', session.graph)
    init = tf.global_variables_initializer()
    session.run(init)
    print(session.run(node_sub, {a:2.0, b:1.0}))

writer.close()
