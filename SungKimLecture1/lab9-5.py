import tensorflow as tf
import numpy as np


x_data = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

with tf.name_scope("layer1") as scope:
    W1 = tf.Variable(tf.random.normal([2,10]), name='weight1')
    b1 = tf.Variable(tf.random.normal([10]), name='bias1')
    w1_hist = tf.summary.histogram("weight1", W1)
    b1_hist = tf.summary.histogram("bias1", b1)
    layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

with tf.name_scope("layer2") as scope:
    W2 = tf.Variable(tf.random.normal([10,10]), name='weight2')
    b2 = tf.Variable(tf.random.normal([10]), name='bias2')
    w2_hist = tf.summary.histogram("weight2", W2)
    b2_hist = tf.summary.histogram("bias2", b2)
    layer2 = tf.sigmoid(tf.matmul(layer1, W2) + b2)

with tf.name_scope("layer3") as scope:
    W3 = tf.Variable(tf.random.normal([10,10]), name='weight3')
    b3 = tf.Variable(tf.random.normal([10]), name='bias3')
    w3_hist = tf.summary.histogram("weight3", W3)
    b3_hist = tf.summary.histogram("bias3", b3)
    layer3 = tf.sigmoid(tf.matmul(layer2, W3) + b3)

with tf.name_scope("layer4") as scope:
    W4 = tf.Variable(tf.random.normal([10,1]), name='weight4')
    b4 = tf.Variable(tf.random.normal([1]), name='bias4')
    w4_hist = tf.summary.histogram("weight4", W4)
    b4_hist = tf.summary.histogram("bias4", b4)
    hypothesis = tf.sigmoid(tf.matmul(layer3, W4) + b4)

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
cost_summ = tf.summary.scalar("cost", cost)
train = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
summary = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./logs/xor_logs')
    writer.add_graph(sess.graph)

    for step in range(10001):
        s, _ = sess.run([summary, train], feed_dict={X:x_data, Y:y_data})
        writer.add_summary(s, global_step=step)
        if step % 100 == 0:
            print(step, sess.run(cost, feed_dict={X:x_data, Y: y_data}), sess.run([W1,W2,W3,W4]))
        
    h,c,a = sess.run([hypothesis, predicted, accuracy], feed_dict={X:x_data, Y:y_data})
    print("\n", h, "\n", c, "\n", a)
