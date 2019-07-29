import tensorflow as tf

x_data = [[1,2,1,1], [2,1,3,2], [3,1,3,4],[4,1,5,5], [1,7,5,5],[1,2,5,6],[1,6,6,6],[1,7,7,7]]
y_data = [[0,0,1],[0,0,1],[0,0,1],[0,1,0],[0,1,0],[0,1,0],[1,0,0],[1,0,0]]

xy_dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))
train_dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data)).batch(10).repeat()
test_dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data)).batch(10)

X = tf.placeholder("float", [None,4])
Y = tf.placeholder("float", [None,3])
nb_classes = 3

W = tf.Variable(tf.random.normal([4, nb_classes]), name='weight')
b = tf.Variable(tf.random.normal([nb_classes]), name='bias')

hypothesis = tf.nn.softmax(tf.matmul(X,W) + b)
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis),axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):

        for item in train_dataset:
            print(item)
        
        sess.run(optimizer, feed_dict={X:x_data, Y:y_data})
        # sess.run(optimizer, feed_dict={X:,Y:})
        if step % 200 == 0:
            print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}))

    a = sess.run(hypothesis, feed_dict={X:[[1,11,7,9]]})
    print(a, sess.run(tf.arg_max(a,1)))

    all = sess.run(hypothesis, feed_dict={X:[[1,11,7,9],[1,3,4,3],[1,1,0,1]]})
    print(all, sess.run(tf.arg_max(all,1)))
