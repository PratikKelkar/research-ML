import tensorflow as tf
import numpy as np

m = 1
n = 4

x = tf.placeholder(tf.float32, [m,n])
W = tf.Variable(tf.zeros([n,1]))
b = tf.Variable(tf.zeros([1]))
model = tf.matmul(x, W) + b
y = tf.placeholder(tf.float32)

loss = tf.reduce_sum(tf.square(model - y))
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

xtrain = [[1, 2, 4, 8]]
ytrain = 2


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1000):
    sess.run(train, {x:xtrain, y:ytrain})

curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x:xtrain, y:ytrain})

print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
