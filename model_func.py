import tensorflow as tf
from utils import *
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple, GRUCell
from tensorflow.contrib import rnn

def primary_model(x, is_training=True):
    output = x
    #output = preprocess(output)
    
    output = tf.layers.conv2d(output,filters=64,kernel_size=4,strides=(1, 1),padding='same', activation=None,use_bias=False,name='Conv_1')
    #for i in range(1,7):
    #	output = s_attention(output,64,i)
    
    output = s_attention(output,64,1)
    output = rg(output,64,1,b=12)

    for i in range(1,3):
        output = tf.layers.conv2d_transpose(output,filters=64,kernel_size=4,strides=(2, 2),padding='same',use_bias=False,name='deconv_'+str(i))
        output = batch_norm(output,is_training,i)        
        output = tf.nn.relu(output)
    
    output = rg(output,64,2,b=6)
    #filter_list = [64,32,16,8,4,2]
    #output = tf.layers.conv2d(output,filters=64,kernel_size=4,strides=(1, 1),padding='same', activation=None,use_bias=False,name='Conv_ext')
    #output = batch_norm(output,is_training,0)        
    #output = tf.nn.relu(output)

    filter_list = [64,32,16,8,4,2]
    for i in range(3,9):
        output = tf.layers.conv2d(output,filters=filter_list[i-3],kernel_size=4,strides=(1, 1),padding='same', activation=None,use_bias=False,name='Conv_'+str(i))
        output = batch_norm(output,is_training,i)        
        output = tf.nn.relu(output)


    output = tf.layers.conv2d(output,filters=1,kernel_size=4,strides=(1, 1),padding='same', activation=None,use_bias=False,name='Conv_2')
    output = tf.nn.tanh(output)
    output += 1.0
    output /= 2.0
    return output

def discriminator(x, is_training=True):    
    noise = tf.random.normal(tf.shape(x),stddev=0.1)
    output = x + noise
    output = tf.layers.conv2d(output,filters=64,kernel_size=4,strides=(2, 2),padding='same', activation=None,use_bias=False,name='Conv_D1')
    output = tf.nn.leaky_relu(output)
    for i in range(2,6):
        noise = tf.random.normal(tf.shape(output),stddev=0.1)
        output = output + noise
        output = tf.layers.conv2d(output,filters=64*2,kernel_size=4,strides=(2, 2),padding='same', activation=None,use_bias=False,name='Conv_D'+str(i))
        output = batch_norm(output,is_training,99+i)
        output = tf.nn.leaky_relu(output)
    noise = tf.random.normal(tf.shape(output),stddev=0.1)
    output = output + noise
    output = tf.layers.conv2d(output,filters=64*32,kernel_size=4,strides=(2, 2),padding='same', activation=None,use_bias=False,name='Conv_D6')
    return output
