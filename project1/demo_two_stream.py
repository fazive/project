# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Hangyan Jiang, based on code from Ross Girshick
# --------------------------------------------------------


"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from lib.config import config as cfg
from lib.utils.nms_wrapper import nms
from lib.utils.test import im_detect
from lib.nets.resnet_v1 import resnetv1
from lib.nets.vgg16 import vgg16
from lib.utils.timer import Timer

CLASSES = ('__background__',
           'copy_move','tamper')

# PLEASE specify weight files dir for vgg16
NETS = {'vgg16': ('vgg16_faster_rcnn_iter_250.ckpt',), 'res50': ('resnet_v1_50_faster_rcnn_iter_10000_3_class.ckpt',)}
DATASETS = {'pascal_voc': ('voc_2007_trainval',), 'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}


def vis_detections(im, class_name, dets, image_name, thresh=0.0):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return
    im = im[:, :, (2, 1, 0)]
    im_copy = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    width = im.shape[0]
    height = im.shape[1]
    mask = np.zeros([width, height, 3], np.uint8)
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
#        print(bbox[0],bbox[1],bbox[3],bbox[2])
        mask[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])] = im_copy[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
#        cv2.rectangle(im, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(0,255,0),3)
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
        )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    #cv2.imwrite(r'static/images/mask/'+image_name, mask)
#    ax.set_title(('{} detections with '
#                  'p({} | box) >= {:.1f}').format(class_name, class_name,
#                                                  thresh),
#                 fontsize=14)
    # plt.axis('off')
    # plt.tight_layout()
    # plt.draw()
#    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
#    cv2.imwrite('lib/layer_utils/data/box/' + image_name, im)
#    plt.imshow(im)
#    plt.show()
    plt.axis('off')
    plt.savefig(r'static/images/box/' + image_name)
def demo(sess, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(r'static/images/pic/', image_name)
    im = cv2.imread(im_file)
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.6
    NMS_THRESH = 0.1
#    im = im[:, :, (2, 1, 0)]
#    fig, ax = plt.subplots(figsize=(12, 12))
#    ax.imshow(im, aspect='equal')
#    print(scores)
    for cls_ind, cls in enumerate(CLASSES[1:]):
#        print(cls_ind,cls)
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets, image_name, thresh=CONF_THRESH)
    # plt.imshow(im)
    # plt.show()


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='res50')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc_0712')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    tf.reset_default_graph()

    args = parse_args()
    # model path
    demonet = args.demo_net
    dataset = args.dataset
    tfmodel = os.path.join('default', 'DIY_dataset', 'default', NETS[demonet][0])

    # if not os.path.isfile(tfmodel + '.meta'):
    #     print(tfmodel)
    #     raise IOError(('{:s} not found.\nDid you download the proper networks from '
    #                    'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        net = vgg16(batch_size=1)
    elif demonet == 'res50':
         net = resnetv1(batch_size=1, num_layers=50)
    else:
        raise NotImplementedError
#    print('生成net')
    net.create_architecture(sess, "TEST", 3,
                            tag='default', anchor_scales=[8, 16, 32])

    saver = tf.train.Saver()
    last_ckpt = saver.last_checkpoints
#    print(last_ckpt)
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))
    for file in os.listdir(r"./static/images/pic"):
        if file.endswith(".jpg"):
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print('Demo for static/images/pic{}'.format(file))
            demo(sess, net, file)