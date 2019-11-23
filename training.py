import tensorflow as tf
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging
import os
from utils import *
from model import *
import glob
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

"""CUDA_DEV = "3"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_DEV"""

NUM_PARALLEL_EXEC_UNITS = 4
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["KMP_BLOCKTIME"] = "30"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"
################# Global Parameter Values #########################
lr = 0.0001
scale = 4
batch_size = 15
crop_size = 200
################# Do not change below this line #########################



tfrecord_parallel_read = 4
inter_op_parallelism_threads = 4
save_summary_steps=10
save_checkpoints_steps=250
keep_checkpoint_max=5
keep_checkpoint_every_n_hours=10000
log_step_count_steps=50
max_steps_to_execute = 10000
warm_start = False
num_samples = 800

filenames = glob.glob("./data/*.tfrecord")
model_dir = 'log/'
checkpoint_file=''
params = {'lrate': lr,
        "logdir": model_dir,
        'batch_size': batch_size}
ws_scopes_to_load = ['','']

with tf.device('/gpu:0'):
    if warm_start:
        ws = tf.estimator.WarmStartSettings(ckpt_to_initialize_from=checkpoint_file,vars_to_warm_start=ws_scopes_to_load)
    else:
        ws = None
    tf.logging.set_verbosity(tf.logging.INFO)
    def parser(serialized_example):
        features = tf.parse_single_example(serialized_example,
        features={'image/encoded_rgb': tf.FixedLenFeature([], tf.string),
                  'image/encoded_y': tf.FixedLenFeature([], tf.string),
                  'image/height': tf.FixedLenFeature([], tf.int64),
                  'image/width': tf.FixedLenFeature([], tf.int64)})
        image_rgb = tf.decode_raw(features['image/encoded_rgb'], tf.uint8)
        image_y = tf.decode_raw(features['image/encoded_y'], tf.uint8)
        height = features['image/height']
        width = features['image/width']
        image_rgb = tf.reshape(image_rgb, [height, width,3])
        image_y = tf.reshape(image_y, [height, width,1])
        image_rgb = ((tf.cast(image_rgb, tf.float32) / 255.0))
        image_y = ((tf.cast(image_y, tf.float32) / 255.0))
        maxh = height - crop_size
        maxw = width - crop_size
        rx = tf.random_uniform((),minval=0,maxval=maxh+1,dtype=tf.int64)
        ry = tf.random_uniform((),minval=0,maxval=maxw+1,dtype=tf.int64)
        image_rgb = image_rgb[rx:rx+crop_size,ry:ry+crop_size,:]
        image_y = image_y[rx:rx+crop_size,ry:ry+crop_size,:]
        image_rgb , image_y = data_augment(image_rgb, image_y)
        image_y = tf.reshape(image_y, [crop_size, crop_size,1])
        image_rgb = tf.reshape(image_rgb, [crop_size, crop_size,3])
        image_rgb = tf.expand_dims(image_rgb,axis=0)
        image_rgb = tf.pad(image_rgb,paddings=[[0,0],[8,8],[8,8],[0,0]],mode='REFLECT',name='Pad')
        h = gkern(13, 1.6)
        h = h[:,:,np.newaxis,np.newaxis].astype(np.float32)
        image_rgb = DownSample(image_rgb, h, scale)
        image_rgb = image_rgb[0,2:-2,2:-2,:]
        label = 0
        return image_rgb, image_y, label
    def train_input_fn(params):
        dataset = tf.data.TFRecordDataset(filenames=filenames,compression_type=None,buffer_size=None,num_parallel_reads=tfrecord_parallel_read)
        dataset = dataset.map(parser, num_parallel_calls=tfrecord_parallel_read)
        dataset = dataset.prefetch(buffer_size=batch_size * 4)
        dataset = dataset.cache()
        dataset = dataset.shard(1,0)
        dataset = dataset.repeat()
        dataset = dataset.shuffle(batch_size * 4,reshuffle_each_iteration=True)
        dataset = dataset.batch(batch_size,drop_remainder=True)
        iterator = dataset.make_one_shot_iterator()
        image_rgb, image_y, label = iterator.get_next()
        features={"image_rgb": image_rgb,"image_y": image_y}
        return features, label
    conf1 = tf.ConfigProto(log_device_placement=False,intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS, inter_op_parallelism_threads=inter_op_parallelism_threads, allow_soft_placement=True, device_count = {'CPU': NUM_PARALLEL_EXEC_UNITS})
    conf1.gpu_options.allow_growth=True
    my_gpu_run_config = tf.estimator.RunConfig(
        save_summary_steps=save_summary_steps,
        save_checkpoints_steps=save_checkpoints_steps,
        session_config = conf1,
        keep_checkpoint_max=keep_checkpoint_max,
        keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours,
        log_step_count_steps=log_step_count_steps)
    model = tf.estimator.Estimator(model_fn=model_fn,
            params=params,
            model_dir=model_dir,
            config=my_gpu_run_config,
            warm_start_from=ws)
    model.train(input_fn=train_input_fn, steps=max_steps_to_execute)