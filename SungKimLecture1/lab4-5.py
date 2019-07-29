import tensorflow as tf
import numpy as np

filelists = ['DeepLearningZeroToAll/data-01-test-score.csv']

# dataset = tf.data.Dataset.from_tensor_slices(filelists)
record_defaults = [0.] * 4
dataset = tf.data.experimental.CsvDataset(filelists, record_defaults = record_defaults, header=False)
dataset = dataset.batch(10)
one_iter = dataset.make_one_shot_iterator()
next_element = one_iter.get_next()


print (dataset)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(next_element))
