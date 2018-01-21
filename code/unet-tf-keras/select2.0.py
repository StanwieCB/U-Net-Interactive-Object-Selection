from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2
import numpy as np

import tensorflow as tf
import os, sys
from model import UNet

# configure args
from opts import *
from skimage import io, transform
from glob import glob

drawing = False
start = False

image_list = glob(os.path.join(opt.image_dir, '*.'+opt.image_ext))

image = cv2.imread(image_list[0])
sp = image.shape
w = sp[0]
h = sp[1]
output = np.zeros((w, h, 3), np.uint8)
left = 0xFFF
right = 0
up = 0xFFF
down = 0

# mouse callback function
def interactive_drawing(event, x, y, flags, param):
    global xs, ys, ix, iy, drawing, image, output, left, right, up, down

    if event == cv2.EVENT_LBUTTONDOWN:
        print('down')
        drawing = True
        ix, iy = x, y
        xs, ys = x, y
        left = min(left, x)
        right = max(right, x)
        up = min(up, y)
        down = max(down, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing is True:
            cv2.line(image, (ix, iy), (x, y), (0, 0, 255), 2)
            cv2.line(output, (ix, iy), (x, y), (255, 255, 255), 1)
            ix = x
            iy = y
            left = min(left, x)
            right = max(right, x)
            up = min(up, y)
            down = max(down, y)
            print(left)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(image, (ix, iy), (x, y), (0, 0, 255), 2)
        cv2.line(output, (ix, iy), (x, y), (255, 255, 255), 1)
        ix = x
        iy = y
        cv2.line(image, (ix, iy), (xs, ys), (0, 0, 255), 2)
        cv2.line(output, (ix, iy), (xs, ys), (255, 255, 255), 1)
    return x, y


def main():
    assert(opt.load_from_checkpoint != '')

    # configuration session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # define data loader
    img_shape = [280, 500]
    with tf.name_scope('unet'):
        model = UNet().create_model(img_shape=img_shape + [4], num_class=opt.num_class)
        img = model.input
        pred = model.output

    saver = tf.train.Saver()  # must be added in the end

    ''' Main '''
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    try:
        saver.restore(sess, tf.train.latest_checkpoint(opt.load_from_checkpoint))
        print('--> load from checkpoint ' + opt.load_from_checkpoint)
    except:
        print('unable to load checkpoint ...')
        sys.exit(0)

    global image, output
    cv2.namedWindow('draw', flags=cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('draw', interactive_drawing)

    image_idx = 0
    while(1):
        cv2.imshow('draw', image)
        k = cv2.waitKey(1) & 0xFF
        if k != 255:
            print(k)
        if k == 100: # D
            cv2.imwrite('results/' + str(image_idx) + 'out.png', image)
        if k == 115:
            global left, right, up, down
            left = 0xFFF
            right = 0
            up = 0xFFF
            down = 0
            drawing = False  # true if mouse is pressed
            image = cv2.imread(image_list[image_idx])
            sp = image.shape
            w = sp[0]
            h = sp[1]
            output = np.zeros((w, h, 3), np.uint8)
            while (1):
                cv2.imshow('draw', image)
                k = cv2.waitKey(1) & 0xFF
                if k == 32:
                    break
                if k == 27:
                    image = cv2.imread(image_list[image_idx])
                    output = np.zeros((w, h, 3), np.uint8)

            output = (output[:, :, 0] > 0).astype(np.uint8)
            print("output_type = ", type(output))
            print("output_size = ", np.shape(output))
            fill_mask = np.ones((output.shape[0] + 2, output.shape[1] + 2))
            fill_mask[1:-1, 1:-1] = output
            fill_mask = fill_mask.astype(np.uint8)
            # print(left, right, up, down)
            cv2.floodFill(output.copy(), fill_mask, (int((left + right) / 2), int((up + down) / 2)), 1)
            fill_mask = fill_mask.astype(np.float32)
            fill_mask = transform.resize(fill_mask, output_shape=img_shape) # Automatically turned into float32 in [0, 1]
            print("fill_mask_max = ", np.max(fill_mask, axis=(0, 1)))


            output = transform.resize(output, output_shape=img_shape)
            output = (output == 0).astype(np.uint8)
            print("output_max = ", np.max(output, (0, 1)))


            dis_map = cv2.distanceTransform(output, cv2.cv.CV_DIST_L2, cv2.cv.CV_DIST_MASK_PRECISE)
            dis_map = dis_map - 2 * fill_mask * dis_map + 128

            ori_image = io.imread(image_list[image_idx])
            ori_shape = np.shape(ori_image)[0:2]
            # print('Original Shape = ', ori_shape)
            resized_image = transform.resize(ori_image, output_shape=img_shape)
            dis_map = np.reshape(dis_map, np.shape(dis_map) + (1,))
            merge_input = np.concatenate((resized_image, dis_map), axis=2)
            x_batch = np.reshape(merge_input, newshape=(1,) + np.shape(merge_input))
            feed_dict = {
                img: x_batch
                # label: y_batch
            }
            pred_logits = sess.run(pred, feed_dict=feed_dict)
            pred_map = np.argmax(pred_logits[0], axis=2).astype(np.float32)
            pred_map = transform.resize(pred_map, ori_shape)
            red_mask = np.zeros(np.shape(ori_image))
            red_mask[:, :] = (0, 0, 255)
            display_mask = 0.4 * np.reshape(pred_map, newshape=pred_map.shape + (1,))
            image = (image * (1 - display_mask) + red_mask * display_mask).astype(np.uint8)

            # io.imsave('demo/' + 'out.png', pred_map)

        if k == 99:
            break

        if k == 110:
            image_idx += 1
            if image_idx >= len(image_list):
                print('Already the last image. Starting from the beginning.')
                image_idx = 0
            image = cv2.imread(image_list[image_idx])
            sp = image.shape
            w = sp[0]
            h = sp[1]
            output = np.zeros((w, h, 3), np.uint8)

        if k == 112:
            image_idx -= 1
            if image_idx < 0:
                print('Reached the first image. Starting from the end.')
                image_idx = len(image_list)-1
            image = cv2.imread(image_list[image_idx])
            sp = image.shape
            w = sp[0]
            h = sp[1]
            output = np.zeros((w, h, 3), np.uint8)


    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
