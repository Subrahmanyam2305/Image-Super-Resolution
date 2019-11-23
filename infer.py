import tensorflow as tf
#tf.enable_eager_execution()
import argparse
from scipy.misc import imsave
from model import *
import os
import math
import numpy as np
np.set_printoptions(threshold=np.nan)
CUDA_DEV = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_DEV
NUM_PARALLEL_EXEC_UNITS = 7
os.environ["OMP_NUM_THREADS"] = "NUM_PARALLEL_EXEC_UNITS"
os.environ["KMP_BLOCKTIME"] = "30"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"
inter_op_parallelism_threads = 5
################# Global Parameter Values#########################

parser = argparse.ArgumentParser(description='Custom')
parser.add_argument('--eval', type=str, default='set5',
                    help='evalutaion set')
args = parser.parse_args()
#file_name = args.eval
file_names = ['set5','set14','b100','urban100']
scale = 4
checkpoint_file = tf.train.latest_checkpoint('log/')#'log_train/model.ckpt-15750'
with tf.Graph().as_default(),tf.device('/gpu:0') as graph:
    tf.logging.set_verbosity(tf.logging.INFO) 
    image_data = tf.placeholder(tf.float32, shape=(None,None,3))
    height = tf.placeholder(tf.int32, shape=())
    width = tf.placeholder(tf.int32, shape=())

    image_data1 = tf.reshape(image_data, [height, width,3])
    image_rgb = ((tf.expand_dims(image_data1,axis=0) / 255.0))
    image_rgb = tf.pad(image_rgb,paddings=[[0,0],[8,8],[8,8],[0,0]],mode='REFLECT',name='Pad')
    h = gkern(13, 1.6)
    h = h[:,:,np.newaxis,np.newaxis].astype(np.float32)
    image_rgb = DownSample(image_rgb, h, scale)
    image_rgb = image_rgb[:,2:-2,2:-2,:]
    with tf.variable_scope("main_model"):
        predicted_y = tf.cast(  ((primary_model(image_rgb,is_training=False)[0,:,:,0])) * 255.0   ,tf.uint8)
        
    ###########################################################
    gpu_options = gpu_options = tf.GPUOptions(allow_growth=True)#tf.GPUOptions(per_process_gpu_memory_fraction=1.0)#
    saver = tf.train.Saver()
    with tf.Session(config=tf.ConfigProto(log_device_placement=False,intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS, inter_op_parallelism_threads=inter_op_parallelism_threads, allow_soft_placement=True, device_count = {'CPU': NUM_PARALLEL_EXEC_UNITS})) as sess:
        saver.restore(sess, checkpoint_file) 
        for file_name in file_names:
            print(file_name," inference starts")
            loaded_file  = np.load('../benchmark/'+file_name+'.npz')
            data_rgb = loaded_file['data_rgb']
            data_y = loaded_file['data_y']
            for i in range(data_rgb.shape[0]):
                input_data_rgb = data_rgb[i]
                input_data_y = data_y[i]
                h, w, c = input_data_rgb.shape
                new_h = (math.floor(h/4)*4)
                new_w = (math.floor(w/4)*4)
                pad_input_data_rgb = input_data_rgb[0:new_h,0:new_w,:]
                pad_input_data_y= input_data_y[0:new_h,0:new_w]
                result = sess.run(predicted_y,{image_data:pad_input_data_rgb,height:pad_input_data_rgb.shape[0],width:pad_input_data_rgb.shape[1]})


                imsave('results/'+file_name+'/predicted/'+str(i+1)+'.png', result)
                imsave('results/'+file_name+'/gt/'+str(i+1)+'.png', pad_input_data_y)
            print(file_name," inference complete")
        