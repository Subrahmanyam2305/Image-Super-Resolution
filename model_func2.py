import tensorflow as tf
from utils import *

def primary_model(x, is_training=True):
    output = preprocess(x)
    output = tf.layers.conv2d(output,filters=96,kernel_size=3,strides=(1, 1),padding='same', activation=None,use_bias=False,name='Conv_1')
    output = s_attention(output,96,1)
 
    filter_list = [256,128]
    for i in range(1,3):
        output = tf.layers.conv2d_transpose(output,filters=filter_list[i-1],kernel_size=3,strides=(2, 2),padding='same',use_bias=False,name='deconv_'+str(i))
        output = batch_norm(output,is_training,i)        
        output = tf.nn.relu(output)
    
    output = tf.layers.conv2d(output,filters=64,kernel_size=3,strides=(1, 1),padding='same', activation=None,use_bias=False,name='Conv_2')
    output = batch_norm(output,is_training,3)        
    output = tf.nn.relu(output)

    for i in range(4):
        output = rg(output,64,i,b=5)

    filter_list = [32,8,8,8,4,2]
    for i in range(3,9):
        output = tf.layers.conv2d(output,filters=filter_list[i-3],kernel_size=3,strides=(1, 1),padding='same', activation=None,use_bias=False,name='Conv_'+str(i))
        output = batch_norm(output,is_training,200+i)        
        output = tf.nn.relu(output)
    
    output = tf.layers.conv2d(output,filters=1,kernel_size=3,strides=(1, 1),padding='same', activation=None,use_bias=False,name='Conv_9')
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
