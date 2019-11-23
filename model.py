import tensorflow as tf
#from model_func import *
from model_func import *

def model_fn(features, labels, mode, params):

    image_rgb = features['image_rgb']
    image_y = features['image_y']
    lrate = params['lrate']
    batch_size = params['batch_size'] 
    model_dir =  params['logdir']  
    

    #### Modify the block below ####
    with tf.variable_scope("main_model",initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02)):
        predicted_y = primary_model(image_rgb,is_training=True)
        l1_loss = tf.norm(predicted_y - image_y, ord=1) * 0.2
    with tf.variable_scope("discriminator",initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02),reuse=tf.AUTO_REUSE):
        real_logits = discriminator(image_y,is_training=True)
        fake_logits = discriminator(predicted_y,is_training=True)
        real_d_loss = get_d_real_loss(real_logits)
        fake_d_loss = get_d_fake_loss(fake_logits)
        g_loss      = get_g_loss(fake_logits)
        d_loss      = (0.5*real_d_loss) + (0.5*fake_d_loss)
        #d_loss      = get_d_fake_loss(real_logits, fake_logits)
        
	#### Do not modify below this ####   



    vars1 = tf.trainable_variables()
    g_params = [v for v in vars1 if v.name.startswith('main_model/')]
    d_params = [v for v in vars1 if v.name.startswith('discriminator/')]
    total_loss = l1_loss + g_loss + d_loss
    global_step = tf.train.get_or_create_global_step()   
    optimizer = tf.train.AdamOptimizer(lrate,beta1=0.5, beta2=0.9)
    update_step_l1 = optimizer.minimize(l1_loss,var_list=g_params,name='l1_loss_minimize',global_step=global_step)
    update_step_g = optimizer.minimize(g_loss,var_list=g_params,name='g_loss_minimize',global_step=None)
    update_step_d = optimizer.minimize(d_loss,var_list=d_params,name='d_loss_minimize',global_step=None)
    update_step = tf.group(*[update_step_l1,update_step_g,update_step_d])

    predicted_y = ((predicted_y)) * 255.0
    predicted_y = tf.cast(predicted_y,tf.uint8)
    image_y = ((image_y)) * 255.0
    image_y = tf.cast(image_y,tf.uint8)
    psnr = tf.reduce_mean(tf.image.psnr(image_y,predicted_y,255,name='PSNR'))
    ssim = tf.reduce_mean(tf.image.ssim(image_y,predicted_y,255))
    tf.summary.scalar('losses/l1_loss', l1_loss)
    tf.summary.scalar('losses/G_loss', g_loss)
    tf.summary.scalar('losses/D_loss', d_loss)
    tf.summary.scalar('accuracy/PSNR', psnr)
    tf.summary.scalar('accuracy/SSIM', ssim)
    tf.summary.scalar('parameter/L_Rate', lrate)
    tf.summary.image('GT/Images', image_y)
    tf.summary.image('Predicted/Images', predicted_y)
    tf.summary.image('Input/Images', image_rgb)
    return tf.estimator.EstimatorSpec(mode=mode,loss=total_loss,train_op=update_step)