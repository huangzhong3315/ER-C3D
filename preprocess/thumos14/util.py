import os
import errno
import numpy as np
from collections import defaultdict

import shutil
import subprocess
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# segment
def dataset_label_parser(meta_dir, split, use_ambiguous=False):
    class_id = defaultdict(int)
    with open(os.path.join(meta_dir, 'detclasslist.txt'), 'r') as f:
        lines = f.readlines()
        for l in lines:
            cname = l.strip().split()[-1]  # 保留最后一段 BaseballPitch
            cid = int(l.strip().split()[0])  # 1
            class_id[cname] = cid
            if use_ambiguous:
                class_id['Ambiguous'] = 21
        segment = {}

    for cname in class_id.keys():
        tmp = '{}_{}.txt'.format(cname, split)
        with open(os.path.join(meta_dir, tmp)) as f:
            lines = f.readlines()
            for l in lines:
                vid_name = l.strip().split()[0]  # video_test_0000278
                start_t = float(l.strip().split()[1])  # 0.0
                end_t = float(l.strip().split()[2])  # 1.4

                # initionalize at the first time  [0.0, 29.3, 12]
                if not vid_name in segment.keys():
                    segment[vid_name] = [[start_t, end_t, class_id[cname]]]
                else:
                    segment[vid_name].append([start_t, end_t, class_id[cname]])

    # sort segments by start_time
    for vid in segment :
        segment[vid].sort(key= lambda x: x[0])

    if True:
        keys = list(segment.keys())
        keys.sort()
        with open('segment.txt', 'w') as f:
            for k in keys:
                f.write("{}\n{}\n\n".format(k, segment[k]))

    return segment

def get_segment_len(segment):
    segment_len = []
    for vid_seg in segment.values():
        for seg in vid_seg:
            l = seg[1]-seg[0]
            assert l > 0
            segment_len.append(l)
    return segment_len

def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def rm(path):
    try:
        shutil.rmtree(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

# 视频数据的处理转换
def ffmpeg(filename, outfile, fps):
    command = ["D:/program files/ffmpeg4.4/bin/ffmpeg", "-i", filename, "-q:v", "1", "-r", str(fps), outfile]
    pipe = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    pipe.communicate()

# def resize(filename, size = (171, 128)):
def resize(filename, size = (171, 128)):
    img = cv2.imread(filename, 100)
    img2 = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(filename, img2, [100])

# get segs_len from segments by: segs_len = [ s[1]-s[0] for v in segments.values() for s in v ]
def kmeans(seg_len, K = 5, vis = False):
    X = np.array(seg_len).reshape(-1, 1)
    cls = KMeans(K).flt(X)
    print("the cluster centers are: ")
    print(cls.cluster_centers_)
    if vis:
        markers = ['^','x','o','*','+']
        for i in range(K):
            members = cls.labels_ == i
            plt.scatter(X[members, 0], X[members, 0], s=60, marker=markers[min(i, K - 1)], c='b', alpha=0.5)
            plt.title(' ')
            plt.show()










