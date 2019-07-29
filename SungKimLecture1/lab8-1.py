# tensor manipulation

import tensorflow as tf
import numpy as np

sess = tf.Session()

t = np.array([[1.,2.,3.], [4.,5.,6.], [7.,8.,9.],[10.,11.,12.]])

tt = tf.constant([1,2,3,4])
t1 = tf.shape(tt).eval(session=sess)
print(t1)

tt = tf.constant([[1,2],[3,4]])
t1 = tf.shape(tt).eval(session=sess)
print(t1)

x = [[0,1,2], [2,1,0]]
t1 = tf.argmax(x, axis=0).eval(session=sess)
print(t1)

x = [[0,1,2], [2,1,0]]
t1 = tf.argmax(x, axis=1).eval(session=sess)
print(t1)

t = np.array([[[0,1,2],[3,4,5]],[[6,7,8],[9,10,11]]])
print(t.shape)

# reshape
ret = tf.reshape(t, shape=[-1, 3]).eval(session=sess)
print(ret, ret.shape)
ret = tf.reshape(t, shape=[-1,1,3]).eval(session=sess)
print(ret, ret.shape)
print(tf.squeeze([[0],[1],[2]]).eval(session=sess))
print(tf.expand_dims([0,1,2], 1).eval(session=sess))

# onehot
t=tf.one_hot([[0],[1],[2],[0]], depth=3).eval(session=sess)
print(t)
t = tf.reshape(t, shape=[-1,3]).eval(session=sess)
print(t)
print(tf.one_hot([[0],[1],[2],[0]], depth=4).eval(session=sess))

# casting
t = tf.cast([1.8,2.2,3.3,4.9], tf.int32).eval(session=sess)
print(t)
t = tf.cast([True, False, 1 == 1, 0 == 1], tf.int32).eval(session=sess)
print(t)

# stack
x = [1,4]
y=[2,5]
z=[3,6]
t = tf.stack([x,y,z]).eval(session=sess)
print(t)
t = tf.stack([x,y,z], axis=1).eval(session=sess)
print(t)

# ones and zeros like
x = [[0,1,2],[2,1,0]]
t = tf.ones_like(x).eval(session=sess)
print(t)
t = tf.zeros_like(x).eval(session=sess)
print(t)

# zip
for x, y in zip([1,2,3], [4,5,6]):
    print(x,y)
for x,y,z in zip([1,2,3],[4,5,6],[7,8,9]):
    print(x,y,z)