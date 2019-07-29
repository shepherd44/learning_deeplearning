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

W1 = tf.Variable(tf.random.normal([784,1000]), name="weight1")
b1 = tf.Variable(tf.random.normal([1000]), name="bias1")
layer1 = tf.nn.softmax(tf.matmul(X,W1) + b1)

W2 = tf.Variable(tf.random.normal([1000,1000]), name="weight2")
b2 = tf.Variable(tf.random.normal([1000]), name="bias2")
layer2 = tf.nn.softmax(tf.matmul(layer1,W2) + b2)

W_last = tf.Variable(tf.random.normal([1000,nb_classes]), name="weight_last")
b_last = tf.Variable(tf.random.normal([nb_classes]), name="bias_last")
hypothesis = tf.nn.softmax(tf.matmul(layer2,W_last) + b_last)

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

is_correct = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(Y,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

training_epochs = 100
batch_size = 100

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, optimizer], feed_dict = {X:batch_xs, Y:batch_ys})
            avg_cost += c/ total_batch

        print("Epoch:", '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

    print("Accuracy:", accuracy.eval(session=sess, feed_dict={X:mnist.test.images, Y:mnist.test.labels}))

    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label:", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
    print("Prediction:", sess.run(tf.argmax(hypothesis, 1), feed_dict={X:mnist.test.images[r:r+1]}))

    plt.imshow(mnist.test.images[r:r+1].reshape(28,28), cmap='Greys', interpolation='nearest')
    plt.show()


