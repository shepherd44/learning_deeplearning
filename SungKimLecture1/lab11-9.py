from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np


class MnistCnnModel:

    """
    """

    def __init__(self, sess, name):
       self.sess = sess
       self.name = name
       self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            self.X = tf.compat.v1.placeholder(tf.float32, [None, 784])
            X_img = tf.reshape(self.X, [-1, 28, 28, 1])
            self.Y = tf.compat.v1.placeholder(tf.float32, [None, 10])
            self.keep_prob = tf.compat.v1.placeholder(tf.float32)
            self.learning_rate = 0.001

            W1 = tf.Variable(tf.random.normal([3, 3, 1, 32], stddev=0.01))
            L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
            L1 = tf.nn.relu(L1)
            L1 = tf.nn.max_pool2d(L1, ksize=[1, 2, 2, 1], strides=[
                1, 2, 2, 1], padding='SAME')
            L1 = tf.nn.dropout(L1, rate=1 - self.keep_prob)

            W2 = tf.Variable(tf.random.normal([3, 3, 32, 64], stddev=0.01))
            L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
            L2 = tf.nn.relu(L2)
            L2 = tf.nn.max_pool2d(L2, ksize=[1, 2, 2, 1], strides=[
                1, 2, 2, 1], padding='SAME')
            L2 = tf.nn.dropout(L2, rate=1 - self.keep_prob)

            W3 = tf.Variable(tf.random.normal([3, 3, 64, 128], stddev=0.01))
            L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
            L3 = tf.nn.relu(L3)
            L3 = tf.nn.max_pool2d(L3, ksize=[1, 2, 2, 1], strides=[
                1, 2, 2, 1], padding='SAME')
            L3 = tf.nn.dropout(L3, rate=1 - self.keep_prob)
            L3 = tf.reshape(L3, [-1, 4*4*128])

            W4 = tf.get_variable(
                "W4", shape=[4*4*128, 625], initializer=tf.contrib.layers.xavier_initializer())
            b4 = tf.Variable(tf.random.normal([625]))
            L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
            L4 = tf.nn.dropout(L4, rate=1 - self.keep_prob)

            W5 = tf.get_variable(
                "W5", shape=[625, 10], initializer=tf.contrib.layers.xavier_initializer())
            b5 = tf.Variable(tf.random.normal([10]))
            self.logits = tf.matmul(L4, W5) + b5

        self.cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.cost)

        correct_prediction = tf.equal(
            tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predict(self, x_test, keep_prob=1.0):
        return self.sess.run(self.logits, feed_dict={
            self.X: x_test, self.keep_prob: keep_prob
        })

    def get_accuracy(self, x_test, y_test, keep_prob=1.0):
        return self.sess.run(self.accuracy, feed_dict={
            self.X: x_test, self.Y: y_test, self.keep_prob: keep_prob
        })

    def train(self, x_data, y_data, keep_prob=0.7, learning_rate=0.001):
        self.learning_rate = learning_rate
        return self.sess.run([self.cost, self.optimizer], feed_dict={
            self.X: x_data, self.Y: y_data, self.keep_prob: keep_prob
        })

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

sess = tf.compat.v1.Session()
m1 = MnistCnnModel(sess, "m1")
sess.run(tf.compat.v1.global_variables_initializer())

print('Learning Started!')
training_epochs = 15
batch_size = 100
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        c, _ = m1.train(batch_xs, batch_ys)
        avg_cost += c / total_batch

print('Learning Finished!')

# predict = m1.predict(mnist.test[0])
print('Accuracy:', m1.get_accuracy(mnist.test.images, mnist.test.labels))