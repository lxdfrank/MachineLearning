import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets ("H:/test/MachineLearning/MNIST_data/", one_hot=True)
train_X, train_Y = mnist.train.next_batch(5000)
test_X, test_Y = mnist.test.next_batch(200)   # 200 for testing

xtr = tf.placeholder(tf.float32,[None,784])
xte = tf.placeholder(tf.float32,[784])

distance = tf.sqrt(tf.reduce_sum(tf.pow(tf.add(xtr,tf.neg(xte)),2),reduction_indices=1))

pred = tf.argmin(distance,0)
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

file = open("1.txt",'w')
array = []
for i in range(28):
    print(i)
    for j in range(28):
        if test_X[1][28*i+j] >0.5:
            array.append(1)
        else:
            array.append(0)
        if j == 27:
            file.write(str(array))
            file.write('\r\n')
            array.clear()
print(train_Y[1])
file.close()


