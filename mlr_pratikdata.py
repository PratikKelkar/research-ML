import tensorflow as tf
import numpy as np
import scipy.io as io

def loaddata(wordtosend):

    flatdata = {}
    
    data = io.loadmat('pratik_mat_smaller')
    for key, val in data.items():
        if "W" in key:
            flatdata[key] = tf.reshape(data[key].flatten()[201:851], [1, 650])

    print(flatdata[wordtosend].shape)
'''    
def learn():

    m = 1
    n = 650
    
    x = tf.placeholder(tf.float32, [m,n])
    W = tf.Variable(tf.zeros([n,1]))
    b = tf.Variable(tf.zeros([1]))
    model = tf.matmul(x, W) + b
    y = tf.placeholder(tf.float32)

    loss = tf.reduce_sum(tf.square(model - y))
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)

    xtrain = loaddata('W023_Segment_001')
    ytrain = 0


    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    for i in range(1000):
        sess.run(train, {x:xtrain, y:ytrain})

    curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x:xtrain, y:ytrain})

    print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))

learn()
'''
