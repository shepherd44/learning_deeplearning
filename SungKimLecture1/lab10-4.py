import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
nb_classes = 10
# 28 * 28 = 784
X = tf.placeholder(tf.float32, [None, 784])
# 0-9 digits : 10 classes
Y = tf.placeholder(tf.float32, [None, nb_classes])
keep_prob = tf.placeholder(tf.float32)

W1 = tf.get_variable(
    "W1", shape=[784, 512], initializer=tf.contrib.layers.xavier_initializer())
W2 = tf.get_variable(
    "W2", shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())
W3 = tf.get_variable(
    "W3", shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())
W4 = tf.get_variable(
    "W4", shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())
W5 = tf.get_variable(
    "W5", shape=[512, nb_classes], initializer=tf.contrib.layers.xavier_initializer())

b1 = tf.Variable(tf.random.normal([512]), name="bias1")
b2 = tf.Variable(tf.random.normal([512]), name="bias2")
b3 = tf.Variable(tf.random.normal([512]), name="bias3")
b4 = tf.Variable(tf.random.normal([512]), name="bias4")
b5 = tf.Variable(tf.random.normal([nb_classes]), name="bias5")


with tf.name_scope("layer1") as scope:
    layer1 = tf.nn.relu(tf.matmul(X,W1) + b1)
    layer1 = tf.nn.dropout(layer1, keep_prob=keep_prob)
with tf.name_scope("layer2") as scope:
    layer2 = tf.nn.relu(tf.matmul(layer1,W2) + b2)
    layer2 = tf.nn.dropout(layer2, keep_prob=keep_prob)
with tf.name_scope("layer3") as scope:
    layer3 = tf.nn.relu(tf.matmul(layer2,W3) + b3)
    layer3 = tf.nn.dropout(layer3, keep_prob=keep_prob)
with tf.name_scope("layer4") as scope:
    layer4 = tf.nn.relu(tf.matmul(layer3,W4) + b4)
    layer4 = tf.nn.dropout(layer4, keep_prob=keep_prob)
with tf.name_scope("hypothesis") as scope:
    hypothesis = tf.matmul(layer4, W5) + b5

learning_rate = 0.001
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

cost_summ = tf.summary.scalar("cost", cost)

is_correct = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(Y,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

training_epochs = 15
batch_size = 100
summary = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("./logs/mnist_relu_test1")
    writer.add_graph(sess.graph)

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            s, c, _ = sess.run([summary, cost, optimizer], feed_dict = {X:batch_xs, Y:batch_ys, keep_prob: 0.7})
            writer.add_summary(s, epoch)
            avg_cost += c / total_batch

        print("Epoch:", '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

    print("Accuracy:", accuracy.eval(session=sess, feed_dict={X:mnist.test.images, Y:mnist.test.labels, keep_prob: 1}))

    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label:", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
    print("Prediction:", sess.run(tf.argmax(hypothesis, 1), feed_dict={X:mnist.test.images[r:r+1], keep_prob:1}))

    plt.imshow(mnist.test.images[r:r+1].reshape(28,28), cmap='Greys', interpolation='nearest')
    plt.show()


