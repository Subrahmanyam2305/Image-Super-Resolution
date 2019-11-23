import tensorflow as tf
from tensorflow.python.layers.core import Dense
import numpy as np

wt_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
wt_reg = None


def get_d_real_loss(real_logits):
    d_real_labels = tf.ones_like(real_logits)
    d_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logits,labels=d_real_labels))
    return d_real_loss

def get_d_fake_loss(fake_logits):
    d_fake_labels = tf.zeros_like(fake_logits)
    d_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits,labels=d_fake_labels))
    return d_fake_loss

def get_g_loss(fake_logits):
    g_labels = tf.ones_like(fake_logits)
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits,labels=g_labels))
    return g_loss   

###############################

"""
def get_d_fake_loss(real_logits, fake_logits):
    return tf.contrib.gan.losses.wargs.wasserstein_discriminator_loss(real_logits,fake_logits,real_weights=1.0,generated_weights=1.0)
def get_g_loss(fake_logits):
    return tf.contrib.gan.losses.wargs.wasserstein_generator_loss(fake_logits,weights=1.0)
"""  
################################
'''
def get_d_real_loss_KL(discriminator_on_data_logits):
    loss = tf.nn.softplus(-discriminator_on_data_logits)
    return tf.reduce_mean(loss)

def get_d_fake_loss_KL(discriminator_on_generator_logits):
    return tf.reduce_mean(tf.nn.softplus(discriminator_on_generator_logits))

def get_g_loss_KL(discriminator_on_generator_logits):
    return tf.reduce_mean(-discriminator_on_generator_logits)
'''
################################

"""
def get_d_real_loss(discriminator_on_data_logits):
    loss = tf.nn.relu(1.0 - discriminator_on_data_logits)
    return tf.reduce_mean(loss)

def get_d_fake_loss(discriminator_on_generator_logits):
    return tf.reduce_mean(tf.nn.relu(1 + discriminator_on_generator_logits))

def get_g_loss(discriminator_on_generator_logits):
    return -tf.reduce_mean(discriminator_on_generator_logits)

def resnet_block(x,channels,is_training,i):
    output = conv2d(x, channels=channels, kernel=3, stride=(1,1), pad=1, pad_type='reflect', use_bias=True, sn=True, scope='rb_conv1_'+str(i))
    output = tf.nn.leaky_relu(batch_norm(output,is_training,100+i))
    output = conv2d(output, channels=channels, kernel=3, stride=(1,1), pad=1, pad_type='reflect', use_bias=True, sn=True, scope='rb_conv2_'+str(i))
    output = tf.nn.leaky_relu(batch_norm(output,is_training,200+i))
    output += x 
    return output

"""
def preprocess(x,scale=4):
    # Resizing x using bicubic interpolation
    n,h,w,c = x.get_shape()
    #new_h = scale*int(h)
    #new_w = scale*int(w)
    new_h = scale*tf.shape(x)[1]
    new_w = scale*tf.shape(x)[2]
    img_cubic = tf.image.resize_images(x,[new_h,new_w],method=tf.image.ResizeMethod.BICUBIC,align_corners=True)
    # Reducing the size using space to depth
    img_new = tf.space_to_depth(img_cubic,block_size=scale)
    # final
    out = tf.concat([x,img_new],axis=-1)
    return out

def batch_norm(X,is_training,i):
    output = tf.contrib.layers.batch_norm(X,
                              updates_collections=None,
                              decay=0.9,
                              center=True,
                              scale=True,
                              is_training=is_training,
                              trainable=is_training,
                              scope='BN_'+str(i),
                              reuse=tf.AUTO_REUSE,
                              fused=True,
                              zero_debias_moving_mean=True,
                              adjustment = lambda shape: ( tf.random_uniform(shape[-1:], 0.93, 1.07), tf.random_uniform(shape[-1:], -0.1, 0.1)),
                              renorm=False)
    return output

def s_attention(X,ch,i):
    f = conv2d(X, ch // 8, kernel=1, stride=(1,1), pad=0, pad_type='reflect', use_bias=True, sn=False, scope='sa_conv1_'+str(i))
    g = conv2d(X, ch // 8, kernel=1, stride=(1,1), pad=0, pad_type='reflect', use_bias=True, sn=False, scope='sa_conv2_'+str(i))
    h = conv2d(X, ch, kernel=1, stride=(1,1), pad=0, pad_type='reflect', use_bias=True, sn=False, scope='sa_conv3_'+str(i))
    s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True) # # [bs, N, N] N is the value obtained after merging middle dims
    beta = tf.nn.softmax(s, axis=-1)  # attention map [bs, N, N]
    o = tf.matmul(beta, hw_flatten(h)) # [bs, N, Ch]
    gamma = tf.get_variable("gamma"+str(i), [1], initializer=tf.constant_initializer(0.0))
    o = tf.reshape(o, shape=tf.shape(X)) # [bs, h, w, C]
    #X = tf.nn.sigmoid(o)*X
    X  = gamma*o + X
    return X

def hw_flatten(x) :
    return tf.reshape(x, shape=[tf.shape(x)[0], -1, tf.shape(x)[-1]]) #keep first and last dim as it is and merge all middle dims

def c_attention(x,ch,i,j,r):

    #out = tf.layers.max_pooling2d(inputs=x,pool_size=(x.get_shape()[1],x.get_shape()[2]),strides=1)
    #out = tf.keras.layers.GlobalMaxPool2D()(x)
    #out = tf.expand_dims(out,axis=1)
    #out = tf.expand_dims(out,axis=1)
    out = tf.get_variable("attn_"+str(i)+'_'+str(j),[x.get_shape()[0],1,1,ch], initializer=tf.truncated_normal_initializer())
    #out = conv2d(out, ch, kernel=1, stride=(1,1), pad=0, pad_type='zero', use_bias=True, sn=False, scope='ca_conv1_'+str(i)+'_'+str(j))
    out = tf.layers.conv2d(out, int(ch/r),kernel_size=1, strides=(1,1),padding='SAME',name='ca_conv5_'+str(i)+'_'+str(j))
    out = tf.nn.relu(out)
    #out = conv2d(out, ch, kernel=1, stride=(1,1), pad=0, pad_type='zero', use_bias=True, sn=False, scope='ca_conv2_'+str(i)+'_'+str(j))
    out = tf.layers.conv2d(out, ch,kernel_size=1, strides=(1,1),padding='SAME',name='ca_conv6_'+str(i)+'_'+str(j))
    out = tf.nn.sigmoid(out)
    out = tf.tile(out,[1,tf.shape(x)[1],tf.shape(x)[2],1])
    out = out*x
    return out


def rcab(x,ch,i,j,r):
    # Residual Channel Attention Block
    out = tf.layers.conv2d(x, ch,kernel_size=3, strides=(1,1),padding='SAME',name='rcab_conv1_'+str(i)+'_'+str(j))
    out = tf.nn.relu(out)
    out = tf.layers.conv2d(out, ch, kernel_size=3, strides=(1,1),padding='SAME',name='rcab_conv2_'+str(i)+'_'+str(j))
    out = c_attention(out,ch,i,j,r)
    out+=x
    return out

def rg(x,ch,i,b=20,r=16):
    out = x
    for j in range(b):
            out = rcab(out,ch,i,j,r)
    out = tf.layers.conv2d(out,ch,kernel_size=3,strides=(1,1),padding='SAME',name='rg_conv_'+str(i))
    out+=x
    return out

def conv2d(x, channels, kernel=4, stride=(1,1), pad=0,padding='VALID', pad_type='zero', use_bias=True, sn=False, scope='conv_0'):
    with tf.variable_scope(scope):
        if pad_type == 'zero' :
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
        if pad_type == 'reflect' :
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode='REFLECT')
        if sn :
            w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], initializer=wt_init,
                                regularizer=wt_reg)
            x = tf.nn.conv2d(input=x, filter=spectral_norm(w),
                             strides=[1, stride[0], stride[1], 1], padding=padding)
            if use_bias :
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)
        else :
            x = tf.layers.conv2d(inputs=x, filters=channels,
                                 kernel_size=kernel, kernel_initializer=wt_init,
                                 kernel_regularizer=wt_reg,
                                 strides=stride, use_bias=use_bias)
        return x

def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = l2_norm(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = l2_norm(u_)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    w_norm = w / sigma

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm


def deconv2d(x, channels, kernel=4, stride=(1,1), padding='SAME', use_bias=True, sn=False, scope='deconv_0'):
    with tf.variable_scope(scope):
        x_shape = x.get_shape().as_list()

        if padding == 'SAME':
            output_shape = [tf.shape(x)[0], tf.shape(x)[1] * stride[0], tf.shape(x)[2] * stride[1], channels]

        else:
            output_shape =[tf.shape(x)[0], x_shape[1] * stride[0] + max(kernel - stride[0], 0), x_shape[2] * stride[1] + max(kernel - stride[1], 0), channels]

        if sn :
            w = tf.get_variable("kernel", shape=[kernel, kernel, channels, x.get_shape()[-1]], initializer=wt_init, regularizer=wt_reg)
            x = tf.nn.conv2d_transpose(x, filter=spectral_norm(w), output_shape=output_shape, strides=[1, stride[0], stride[1], 1], padding=padding)

            if use_bias :
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)

        else :
            x = tf.layers.conv2d_transpose(inputs=x, filters=channels,
                                           kernel_size=kernel, kernel_initializer=wt_init, kernel_regularizer=wt_reg,
                                           strides=stride, padding=padding, use_bias=use_bias)

        return x

def l2_norm(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

def data_augment(image_rgb, image_y):
    opt = tf.random_uniform((),minval=0,maxval=4,dtype=tf.int32)
    image_rgb = tf.cond(tf.equal(opt,0), lambda: tf.image.rot90(image_rgb,k=1), lambda: image_rgb)
    image_rgb = tf.cond(tf.equal(opt,1), lambda: tf.image.rot90(image_rgb,k=2), lambda: image_rgb)
    image_rgb = tf.cond(tf.equal(opt,2), lambda: tf.image.rot90(image_rgb,k=3), lambda: image_rgb)
    image_y = tf.cond(tf.equal(opt,0), lambda: tf.image.rot90(image_y,k=1), lambda: image_y)
    image_y = tf.cond(tf.equal(opt,1), lambda: tf.image.rot90(image_y,k=2), lambda: image_y)
    image_y = tf.cond(tf.equal(opt,2), lambda: tf.image.rot90(image_y,k=3), lambda: image_y)
    opt = tf.random_uniform((),minval=0,maxval=2,dtype=tf.int32)
    image_rgb = tf.cond(tf.equal(opt,0), lambda: tf.image.flip_left_right(image_rgb), lambda: image_rgb)
    image_y = tf.cond(tf.equal(opt,0), lambda: tf.image.flip_left_right(image_y), lambda: image_y)
    return image_rgb, image_y
        
def DownSample(x, h, scale=4):
    #ds_x = x.get_shape()
    #x = tf.reshape(x, [ds_x[0]*ds_x[1], ds_x[2], ds_x[3], 3])
    W = tf.constant(h)
    filter_height, filter_width = 13, 13
    pad_height = filter_height - 1
    pad_width = filter_width - 1
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    pad_array = [[0,0], [pad_top, pad_bottom], [pad_left, pad_right], [0,0]]    
    depthwise_F = tf.tile(W, [1, 1, 3, 1])
    y = tf.nn.depthwise_conv2d(tf.pad(x, pad_array, mode='REFLECT'), depthwise_F, [1, scale, scale, 1], 'VALID')
    ds_y = y.get_shape()
    #y = tf.reshape(y, [ds_x[0], ds_x[1], ds_y[1], ds_y[2], 3])
    return y

def gkern(kernlen=13, nsig=1.6):
    import scipy.ndimage.filters as fi
    inp = np.zeros((kernlen, kernlen))
    inp[kernlen//2, kernlen//2] = 1
    return fi.gaussian_filter(inp, nsig)
