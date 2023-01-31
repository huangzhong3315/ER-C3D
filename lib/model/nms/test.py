import numpy as np
import matplotlib.pyplot as plt


def plot_bbox(dets, c='k', title='None'):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    plt.plot([x1, x2], [y1, y1], c)
    plt.plot([x1, x1], [y1, y2], c)
    plt.plot([x1, x2], [y2, y2], c)
    plt.plot([x2, x2], [y1, y2], c)
    plt.title(title)


def cpu_softnms(dets, iou_thresh, sigma=0.5, thresh=0.001, method=2):
    N = dets.shape[0]  # the size of bboxes
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    scores = dets[:, 4]
    for i in range(N):

        temp_box = dets[i, :4]
        temp_score = scores[i]
        temp_area = areas[i]
        pos = i + 1

        # m <---- argmax S
        if i != N - 1:
            maxscore = np.max(scores[pos:])
            maxpos = np.argmax(scores[pos:])
        else:
            maxscore = scores[-1]
            maxpos = -1

        #
        if temp_score < maxscore:
            dets[i, :4] = dets[maxpos + i + 1, :4]  # M <--- b_m
            dets[maxpos + i + 1, :4] = temp_box  # swap position
            # temp_box = dets[i, :4]

            scores[i] = scores[maxpos + i + 1]
            scores[maxpos + i + 1] = temp_score  # swap position
            # temp_score = scores[i]

            areas[i] = areas[maxpos + i + 1]
            areas[maxpos + i + 1] = temp_area  # swap position
            # temp_area = areas[i]

        # calculate IoU  iou(M, b_i)
        xx1 = np.maximum(x1[i], x1[pos:])
        xx2 = np.minimum(x2[i], x2[pos:])
        yy1 = np.maximum(y1[i], y1[pos:])
        yy2 = np.minimum(y2[i], y2[pos:])

        w = np.maximum(xx2 - xx1 + 1.0, 0.)
        h = np.maximum(yy2 - yy1 + 1.0, 0.)

        inters = w * h
        ious = inters / (areas[i] + areas[pos:] - inters)

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

    inds = np.argwhere(scores >= thresh)
    keep = inds.reshape(1, inds.shape[0])[0]  # [0 1 2 3 4]

    return keep


# boxes = np.array([[100, 100, 210, 210, 0.72],
#                   [250, 250, 420, 420, 0.80],
#                   [220, 220, 320, 330, 0.92],
#                   [120, 120, 210, 210, 0.72],
#                   [230, 240, 325, 330, 0.81],
#                   [220, 230, 315, 330, 0.90],
#                   [50, 50, 150, 150, 0.60]
#                   ])
boxes = np.array([[100, 100, 210, 210, 0.1],
                  [250, 250, 420, 420, 0.8],
                  [220, 220, 320, 330, 0.92],
                  [100, 100, 240, 240, 0.72],
                  [230, 240, 325, 330, 0.81],
                  [220, 230, 315, 340, 0.9]])


plt.figure(1)
ax1 = plt.subplot(1, 2, 1)
ax2 = plt.subplot(1, 2, 2)

plt.sca(ax1)
plot_bbox(boxes, 'k', title='original')

plt.sca(ax2)
keep = cpu_softnms(boxes, iou_thresh=0.9, thresh=0.001, method=2)
plot_bbox(boxes[keep], 'r', title='soft-NMS')
plt.show()

