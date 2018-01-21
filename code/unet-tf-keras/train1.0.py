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
vis = VIS(save_path=opt.checkpoint_path)
print('num_class = %d' % opt.num_class)

# configuration session
config = tf.ConfigProto(
    device_count={'GPU': 0}
)
# config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

img_shape = [280, 500]  # Resized to be the same.

# define input holders
train_list = glob.glob(opt.data_path + '/train/img/0/*.png')
train_num = len(train_list)
test_list = glob.glob(opt.data_path + '/val/img/0/*.png')
test_num = len(test_list)
for idx, fname in enumerate(train_list):
    _, train_list[idx] = os.path.split(fname)
for idx, fname in enumerate(test_list):
    _, test_list[idx] = os.path.split(fname)

opt.iter_epoch = int(train_num / opt.batch_size)

label = tf.placeholder(tf.int32, shape=[None] + img_shape)
# define model
with tf.name_scope('unet'):
    model = UNet().create_model(img_shape=img_shape + [4], num_class=opt.num_class)
    img = model.input
    pred = model.output
# define loss
with tf.name_scope('weighted_cross_entropy'):
    cross_entropy_loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=pred))
# define optimizer
global_step = tf.Variable(0, name='global_step', trainable=False)
with tf.name_scope('learning_rate'):
    learning_rate = tf.train.exponential_decay(opt.learning_rate, global_step,
                                               opt.iter_epoch, opt.lr_decay, staircase=True)
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy_loss,
                                                                          global_step=global_step)
''' Tensorboard visualization '''
# cleanup pervious info
if opt.load_from_checkpoint == '':
    cf = os.listdir(opt.checkpoint_path)
    for item in cf:
        if 'event' in item:
            os.remove(os.path.join(opt.checkpoint_path, item))
# define summary for tensorboard
tf.summary.scalar('cross_entropy_loss', cross_entropy_loss)
tf.summary.scalar('learning_rate', learning_rate)
summary_merged = tf.summary.merge_all()
# define saver
train_writer = tf.summary.FileWriter(opt.checkpoint_path, sess.graph)
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

    # debug
    start = global_step.eval()
    for it in range(start, tot_iter):
        # print('Iter = %d' % it)
        if (it + 1) % opt.iter_epoch == 0 or it == start:

            saver.save(sess, opt.checkpoint_path + 'model', global_step=global_step)
            print('save a checkpoint at ' + opt.checkpoint_path + 'model-' + str(it))
            print('start testing {} samples...'.format(test_num))
            for ti in range(50):
                ori_name, ext = os.path.splitext(test_list[ti])
                x = io.imread(opt.data_path + '/val/img/0/' + test_list[ti])
                x = transform.resize(x, img_shape)
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
                # cv2.imwrite('dmap.png', dis_map)

                # dmap = io.imread(opt.data_path+'/dis_map/R'+
                #                          ori_name[:-2]+'_'+str(map_choice)+'.png')
                dis_map = np.reshape(dis_map, dis_map.shape + (1,))
                x_batch = np.concatenate((x, dis_map), axis=2)
                x_batch = np.reshape(x_batch, (1,) + x_batch.shape)

                y = io.imread(opt.data_path + '/val/gt/0/' + ori_name[:-2] + '.png')
                y = transform.resize(y, img_shape) > 0
                y = y.astype(np.int32)
                y_batch = np.reshape(y, (1,) + y.shape)

                # tensorflow wants a different tensor order
                feed_dict = {
                    img: x_batch,
                    label: y_batch,
                }
                loss, pred_logits = sess.run([cross_entropy_loss, pred], feed_dict=feed_dict)
                pred_map_batch = np.argmax(pred_logits, axis=3)
                # import pdb; pdb.set_trace()
                for pred_map, y in zip(pred_map_batch, y_batch):
                    score = vis.add_sample(pred_map, y)
            vis.compute_scores(suffix=it)

        # x_batch, y_batch = next(train_generator)

        batch_index = np.random.randint(0, train_num, opt.batch_size)
        x_batch = np.zeros([opt.batch_size] + img_shape + [4])
        y_batch = np.zeros([opt.batch_size] + img_shape)
        for cnt, i in enumerate(batch_index):
            # print('batch_iter = %d' % cnt)
            ori_name, __ = os.path.splitext(train_list[i])
            x = io.imread(opt.data_path + '/train/img/0/' + train_list[i])
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
            x_batch[cnt] = np.concatenate((x, dis_map), axis=2)

            y = io.imread(opt.data_path + '/train/gt/0/' + ori_name[:-2] + '.png')
            y = transform.resize(y, img_shape) > 0
            y = y.astype(np.int32)
            y_batch[cnt] = y

        feed_dict = {img: x_batch,
                     label: y_batch
                     }
        _, loss, summary, lr, pred_logits = sess.run([train_step,
                                                      cross_entropy_loss,
                                                      summary_merged,
                                                      learning_rate,
                                                      pred
                                                      ], feed_dict=feed_dict)
        global_step.assign(it).eval()

        train_writer.add_summary(summary, it)
        if (it + 1) % 20 == 0:
            pred_map = np.argmax(pred_logits[0], axis=2)
            score, _ = mean_IU(pred_map, y_batch[0])
            print('max_pred = %d, max_gt = %d ' % (
                np.max(pred_map, (0, 1)), np.max(y_batch[0], (0, 1))))  # ('backup/'+str(it)+'.png',

            print(
                '[iter %d, epoch %.3f]: lr=%f loss=%f, mean_IU=%f' % (
                    it + 1, float(it + 1) / opt.iter_epoch, lr, loss, score))
