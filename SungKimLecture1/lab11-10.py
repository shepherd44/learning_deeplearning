from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np


class MnistCnnModel:

    """
    MnistCnnModel
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
            self.training = tf.compat.v1.placeholder(tf.bool)

            conv1 = tf.layers.conv2d(inputs=X_img, filters=32, kernel_size=[3, 3],padding="SAME", activation=tf.nn.relu)
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2],padding="SAME", strides=2)
            drop1 = tf.layers.dropout(inputs=pool1, rate=0.3, training=self.training)

            conv2 = tf.layers.conv2d(inputs=drop1, filters=64, kernel_size=[3, 3],padding="SAME", activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], padding="SAME", strides=2)
            drop2 = tf.layers.dropout(inputs=pool2, rate=0.3, training=self.training)

            conv3 = tf.layers.conv2d(inputs=drop2, filters=128, kernel_size=[3, 3],padding="SAME", activation=tf.nn.relu)
            pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2,2], padding="SAME", strides=2)
            drop3 = tf.layers.dropout(inputs=pool3, rate=0.3, training=self.training)

            flat = tf.reshape(drop3, [-1, 4*4*128])
            dense4 = tf.layers.dense(inputs=flat, units=625, activation=tf.nn.relu)
            drop4 = tf.layers.dropout(inputs=dense4, rate=0.3, training=self.training)

            W5 = tf.get_variable(
                "W5", shape=[625, 10], initializer=tf.contrib.layers.xavier_initializer())
            b5 = tf.Variable(tf.random.normal([10]))
            self.logits = tf.matmul(drop4, W5) + b5

        self.cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.cost)

        correct_prediction = tf.equal(
            tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predict(self, x_test, keep_prob=1.0, training=False):
        return self.sess.run(self.logits, feed_dict={
            self.X: x_test, self.keep_prob: keep_prob, self.training: training
        })

    def get_accuracy(self, x_test, y_test, keep_prob=1.0, training = False):
        return self.sess.run(self.accuracy, feed_dict={
            self.X: x_test, self.Y: y_test, self.keep_prob: keep_prob,
            self.training: training
        })

    def train(self, x_data, y_data, keep_prob=0.7, learning_rate=0.001, training = True):
        self.learning_rate = learning_rate
        return self.sess.run([self.cost, self.optimizer], feed_dict={
            self.X: x_data, self.Y: y_data, self.keep_prob: keep_prob,
            self.training: training
        })

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

sess = tf.compat.v1.Session()
models = []
num_models = 7
for m in range(num_models):
    models.append(MnistCnnModel(sess, "model" + str(m)))
sess.run(tf.compat.v1.global_variables_initializer())

print('Learning Started!')
training_epochs = 15
batch_size = 100
for epoch in range(training_epochs):
    avg_cost_list = np.zeros(len(models))
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        for m_idx, m in enumerate(models):
            c, _ = m.train(batch_xs, batch_ys)
            avg_cost_list[m_idx] += c / total_batch

    print('Epoch:', '%04d'%(epoch + 1), 'cost =', avg_cost_list)

print('Learning Finished!')

test_size = len(mnist.test.labels)
predictions = np.zeros(test_size * 10).reshape(test_size, 10)

for m_idx, m in enumerate(models):
    print(m_idx, 'Accuracy:', m.get_accuracy(mnist.test.images, mnist.test.labels))
    p = m.predict(mnist.test.images)
    predictions += p

ensemble_correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(mnist.test.labels, 1))
ensemble_accuracy = tf.reduce_mean(tf.cast(ensemble_correct_prediction, tf.float32))
print('Ensemble accuracy:', sess.run(ensemble_accuracy))

# predict = m1.predict(mnist.test[0])
# print('Accuracy:', m.get_accuracy(mnist.test.images, mnist.test.labels))
