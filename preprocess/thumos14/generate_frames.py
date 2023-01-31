# import os
# from preprocess.thumos14.util import *
#
# fps = 25
# ext = '.mp4'
# VIDEO_DIR = 'F:/TH14/video/validation'
# FRAME_DIR = 'F:/TH14/frame'
#
# META_DIR = os.path.join(FRAME_DIR, 'annotation_')
#
# def generate_frame(split):
#     SUB_FRAME_DIR = os.path.join(FRAME_DIR, split)
#     os.mkdir(SUB_FRAME_DIR)   # 创建文件
#     segment = dataset_label_parser(META_DIR+split, split, use_ambiguous=False)
#     video_list = segment.keys()
#     for vid in video_list:
#         filename = os.path.join(VIDEO_DIR, split, vid+ext)
#         outpath = os.path.join(FRAME_DIR, split, vid)
#         outfile = os.path.join(outpath, "image_%5d.jpg")
#         mkdir(outpath)
#         ffmpeg(filename, outfile, fps)
#         for framename in os.listdir(outpath):
#             resize(os.path.join(outpath, framename))
#         frame_size = len(os.listdir(outpath))
#         print(filename, fps, frame_size)
#
# generate_frame('val')
# # for split in ['val', 'test']:
# #     generate_frame(split)


#coding=utf-8
# --------------------------------------------------------
# R-C3D
# Copyright (c) 2017 Boston University
# Licensed under The MIT License [see LICENSE for details]
# Written by Huijuan Xu
# --------------------------------------------------------

import os
from preprocess.thumos14.util import *
import json
import glob

# fps = 10  #25
# ext = '.mp4'
# VIDEO_DIR = 'F:/TH14/video/validation'
# FRAME_DIR = 'F:/TH14/frame1'
#
# META_DIR = os.path.join(FRAME_DIR, 'annotation_')
#
# def generate_frame(split):
#   SUB_FRAME_DIR = os.path.join(FRAME_DIR, split)  # F:/TH14/frame/val
#   mkdir(SUB_FRAME_DIR)  # 创建目录
#   segment = dataset_label_parser(META_DIR+split, split, use_ambiguous=False)
#   video_list = segment.keys()
#
#   for vid in video_list:
#     filename = os.path.join(VIDEO_DIR, vid+ext)
#     outpath = os.path.join(FRAME_DIR, split, vid)
#     outfile = os.path.join(outpath, "image_%5d.jpg")
#     mkdir(outpath)
#     ffmpeg(filename, outfile, fps)
#     for framename in os.listdir(outpath):
#       resize(os.path.join(outpath, framename))
#     frame_size = len(os.listdir(outpath))
#     print (filename, fps, frame_size)
#
# generate_frame('val')


import os
from preprocess.thumos14.util import *
import json
import glob

fps = 10  #25
ext = '.mp4'
VIDEO_DIR = r'G:\Gesture Recognition\TH14\validation'
FRAME_DIR = r'G:\Gesture Recognition\TH14\frame'

META_DIR = os.path.join(FRAME_DIR, 'annotation_')

def generate_frame(split):
  SUB_FRAME_DIR = os.path.join(FRAME_DIR, split)  # F:/TH14/frame/val
  mkdir(SUB_FRAME_DIR)  # 创建目录
  segment = dataset_label_parser(META_DIR+split, split, use_ambiguous=False)
  video_list = segment.keys()

  for vid in video_list:
    filename = os.path.join(VIDEO_DIR, vid+ext)
    outpath = os.path.join(FRAME_DIR, split, vid)
    outfile = os.path.join(outpath, "image_%5d.jpg")
    mkdir(outpath)
    ffmpeg(filename, outfile, fps)
    for framename in os.listdir(outpath):
      resize(os.path.join(outpath, framename))
    frame_size = len(os.listdir(outpath))
    print(filename, fps, frame_size)

generate_frame('val')

