from __future__ import absolute_import

import numpy as np
import torch
import matplotlib.pyplot as plt
import  os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def nms_cpu(dets, thresh):
    # dets = dets.cpu().numpy()
    # dets = dets.numpy()
    x1 = dets[:, 0]
    x2 = dets[:, 1]
    scores = dets[:, 2]

    length = (x2 - x1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order.item(0)
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        #yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        #yy2 = np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0.0, xx2 - xx1 + 1)
        ovr = inter / (length[i] + length[order[1:]] - inter)

        inds = np.where(ovr < thresh)[0]
        order = order[inds+1]

    return keep

    # return torch.IntTensor(keep)
# def softnms_cpu(boxes, sigma=0.5, threshold1=0.7, threshold2=0.1, method=1):
#     '''
#     paper:Improving Object Detection With One Line of Code
#     '''
#     N = boxes.shape[0]       # 边界个数
#     pos = 0
#     maxscore = 0
#     maxpos = 0
#
#     # 遍历所有边界框
#     for i in range(N):
#         maxscore = boxes[i, 2]
#         maxpos = i
#
#         tx1 = boxes[i, 0]
#         tx2 = boxes[i, 1]
#         ts = boxes[i, 2]
#
#         pos = i + 1
#         # 得到评分最高的box
#         while pos < N:
#             if maxscore < boxes[pos, 2]:
#                 maxscore = boxes[pos, 2]
#                 maxpos = pos
#             pos = pos + 1
#
#         # 交换第i个box和评分最高的box,将评分最高的box放到第i个位置
#         boxes[i, 0] = boxes[maxpos, 0]
#         boxes[i, 1] = boxes[maxpos, 1]
#         boxes[i, 2] = boxes[maxpos, 2]
#
#         boxes[maxpos, 0] = tx1
#         boxes[maxpos, 1] = tx2
#         boxes[maxpos, 2] = ts
#
#         tx1 = boxes[i, 0]
#         tx2 = boxes[i, 1]
#         ts = boxes[i, 2]
#
#         pos = i + 1
#         # softNMS迭代
#         while pos < N:
#             x1 = boxes[pos, 0]
#             x2 = boxes[pos, 1]
#             s = boxes[pos, 2]
#
#             # area = (x2 - x1 + 1) * (y2 - y1 + 1)
#             length = (x2 - x1 + 1)
#             xx1 = np.maximum(x1[i], x1[pos[1:]])
#             xx2 = np.minimum(x2[i], x2[pos[1:]])
#
#             inter = np.maximum(0.0, xx2 - xx1 + 1)
#             iou = inter / (length[i] + length[pos[1:]] - inter)
#             iw = (min(tx2, x2) - max(tx1, x1) + 1)
#             if iw > 0:
#                 ih = (min(ty2, y2) - max(ty1, y1) + 1)
#                 if ih > 0:
#                     uinon = float((tx2 - tx1 + 1) *
#                                   (ty2 - ty1 + 1) + area - iw * ih)
#                     iou = iw * ih / uinon  # 计算iou
#                     if method == 1:  # 线性更新分数
#                         if iou > threshold1:
#                             weight = 1 - iou
#                         else:
#                             weight = 1
#                     elif method == 2:  # 高斯权重
#                         weight = np.exp(-(iou * iou) / sigma)
#                     else:  # 传统 NMS
#                         if iou > threshold1:
#                             weight = 0
#                         else:
#                             weight = 1
#
#                     boxes[pos, 4] = weight * boxes[pos, 4]  # 根据和最高分数box的iou来更新分数
#
#                     # 如果box分数太低，舍弃(把他放到最后，同时N-1)
#                     if boxes[pos, 4] < threshold2:
#                         boxes[pos, 0] = boxes[N - 1, 0]
#                         boxes[pos, 1] = boxes[N - 1, 1]
#                         boxes[pos, 2] = boxes[N - 1, 2]
#                         boxes[pos, 3] = boxes[N - 1, 3]
#                         boxes[pos, 4] = boxes[N - 1, 4]
#                         N = N - 1  # 注意这里N改变
#                         pos = pos - 1
#
#             pos = pos + 1
#
#     keep = [i for i in range(N)]
#     return keep

def softnms_cpu(dets, iou_thresh, sigma=0.5, thresh=0.001, method=1):
    # dets = dets.numpy()
    N = dets.shape[0]  # the size of bboxes
    x1 = dets[:, 0]
    x2 = dets[:, 1]

    length = (x2 - x1 + 1)
    scores = dets[:, 2]

    for i in range(N):

        temp_box = dets[i, :2]
        temp_score = scores[i]
        temp_area = length[i]
        pos = i + 1

        # m <---- argmax S
        if i != N - 1:
            maxscore = np.max(scores[pos])
            maxpos = np.argmax(scores[pos])
        else:
            maxscore = scores[-1]
            maxpos = -1

        #
        if temp_score < maxscore:
            dets[i, :2] = dets[maxpos + i + 1, :2]  # M <--- b_m
            dets[maxpos + i + 1, :2] = temp_box  # swap position
            # temp_box = dets[i, :4]

            scores[i] = scores[maxpos + i + 1]
            scores[maxpos + i + 1] = temp_score  # swap position
            # temp_score = scores[i]

            length[i] = length[maxpos + i + 1]
            length[maxpos + i + 1] = temp_area  # swap position
            # temp_area = areas[i]

        # calculate IoU  iou(M, b_i)
        xx1 = np.maximum(x1[i], x1[pos:])
        xx2 = np.minimum(x2[i], x2[pos:])

        inter = np.maximum(0.0, xx2 - xx1 + 1)
        ious = inter / (length[i] + length[pos:] - inter)


        # Three methods, linear, gaussian, original
        # f(iou(M, b_i))
        if method == 1:
            weight = np.ones(ious.shape)
            weight[ious > iou_thresh] = weight[ious > iou_thresh] - ious[ious > iou_thresh]
        elif method == 2:
            weight = np.exp(-ious * ious / sigma)
        else:
            weight = np.ones(ious.shape)
            weight[ious > iou_thresh] = 0
        scores[pos:] = scores[pos:] * weight  # s_i <---- s_i * f(iou(M, b_i))

    inds = np.where(scores >= thresh)
    keep = inds

    return keep


def plot_dets(dets,c):
    x1 = dets[:, 0]
    x2 = dets[:, 1]

    plt.plot([x1,x2],c)
    # plt.plot([x1,x1], [x2,x2], [x1,x2], c)




if __name__ == '__main__':
    # dets = np.array([[156.8, 162.2641609,0.10728362],
    #                  [51.0079712,54.4,0.12332062],
    #                  [12.8,25.075143,0.19384557],
    #                  [184.517537,191.6281876, 0.29564312],
    #                  [51.2,56.927982, 0.36355543],
    #                  [22.576532745,29.07037811, 0.442815334],
    #                  [148.2746906, 159.382387, 0.44486701],
    #                  [86.4, 93.23976135000001, 0.456036985],
    #                  [112.91391945000001, 123.503044, 0.51111805],
    #                  [189.23909721, 200.68448800000002, 0.6122672]
    #                  ])
    dets = np.array([[100, 210, 0.1],
                  [250, 420, 0.8],
                  [220, 330, 0.92],
                  [100, 240, 0.72],
                  [230, 330, 0.81],
                  [220, 315, 0.9]])



    plt.figure()
    ax1 = plt.subplot(131)
    ax2 = plt.subplot(132)
    ax3 = plt.subplot(133)

    plt.sca(ax1)
    plot_dets(dets, 'k')

    res = nms_cpu(dets, 0.7)
    plt.sca(ax2)
    plot_dets(dets[res], 'r')

    res1 = softnms_cpu(dets, 0.7)
    plt.sca(ax3)
    plot_dets(dets[res1], 'b')

    plt.show()





