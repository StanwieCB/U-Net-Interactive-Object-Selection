from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

SEED = 0
import numpy as np

np.random.seed(SEED)
import tensorflow as tf

tf.set_random_seed(SEED)

import os, shutil, glob
from skimage import transform, io
from model import UNet
from utils import VIS, mean_IU
# configure args
from opts import *
import cv2

# save and compute metrics
print('num_class = %d' % opt.num_class)

# configuration session
config = tf.ConfigProto(
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5),
    device_count={'GPU': 0}
)
# config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

img_shape = [280, 500]  # Resized to be the same.

# define input holders
test_list = glob.glob(opt.data_path + '/test/img/0/*.png')
test_num = len(test_list)
for idx, fname in enumerate(test_list):
    _, test_list[idx] = os.path.split(fname)

label = tf.placeholder(tf.int32, shape=[None] + img_shape)
# define model
with tf.name_scope('unet'):
    model = UNet().create_model(img_shape=img_shape + [4], num_class=opt.num_class)
    img = model.input
    pred = model.output

# define saver
saver = tf.train.Saver(max_to_keep=20)  # must be added in the end

''' Main '''
tot_iter = opt.iter_epoch * opt.epoch
init_op = tf.global_variables_initializer()

with tf.Session().as_default() as sess:  # .as_default():
    # restore from a checkpoint if exists
    # the name_scope can not change
    sess.run(init_op)
    if opt.load_from_checkpoint != '':
        try:
            # saver = tf.train.import_meta_graph('./checkpoints/unetxPascal/model-0.meta')
            saver.restore(sess, tf.train.latest_checkpoint(opt.load_from_checkpoint))
            print('--> load from checkpoint ' + opt.load_from_checkpoint)
        except:
            print('unable to load checkpoint ...')

    x_batch = np.zeros([1] + img_shape + [4])
    y_batch = np.zeros([1] + img_shape)

    sum = 0

    for cnt, i in enumerate(xrange(opt.batch_size)):
        # print('batch_iter = %d' % cnt)
        ori_name, __ = os.path.splitext(test_list[i])
        x = io.imread(opt.data_path + '/test/img/0/' + test_list[i])
        x = transform.resize(x, img_shape) # Turned into [0, 1] float32 automatically
        map_choice = np.random.randint(1, 16)
        selection_rect_img = io.imread(opt.data_path + '/my-VOC2012/rects/R' +
                                       ori_name[:-2] + '_' + str(map_choice) + '.png')
        ori_shape = selection_rect_img.shape
        selection_rect_img = transform.resize(selection_rect_img, output_shape=img_shape) > 0
        nonzero_list = np.nonzero(selection_rect_img)
        left, right, up, down = nonzero_list[1][0], nonzero_list[1][-1], \
                                nonzero_list[0][0], nonzero_list[0][-1]
        inside_mask = np.zeros(selection_rect_img.shape, dtype=np.uint8)
        inside_mask[up:down + 1, left:right + 1] = 1

        selection_rect_img = (selection_rect_img == 0).astype(np.uint8)
        dis_map = cv2.distanceTransform(selection_rect_img, cv2.cv.CV_DIST_L2, cv2.cv.CV_DIST_MASK_PRECISE)
        dis_map = dis_map - 2 * inside_mask * dis_map + 128

        dis_map = np.reshape(dis_map, dis_map.shape + (1,))
        x_batch[0] = np.concatenate((x, dis_map), axis=2)

        y = io.imread(opt.data_path + '/test/gt/0/' + ori_name[:-2] + '.png')
        y = transform.resize(y, img_shape) > 0
        y = y.astype(np.int32)
        y_batch[0] = y

        feed_dict = {img: x_batch,
                     label: y_batch
                     }
        pred_logits = sess.run(pred, feed_dict=feed_dict)

        pred_map = np.argmax(pred_logits, axis=3)

        score, _ = mean_IU(pred_map[0], y_batch[0])
        print(score)

        sum += score
    score = sum / opt.batch_size

    print('mean_IU = %f' % score)
