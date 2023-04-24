# adapted from: https://github.com/colincsl/TemporalConvolutionalNetworks/blob/master/code/metrics.py

import numpy as np
import xlsxwriter
import matplotlib.pyplot as plt
import os
import torch


def read_file(path):
    with open(path, 'r') as f:
        content = f.read()
        f.close()
    return content


def get_labels_start_end_time(frame_wise_labels, bg_class=["background"]):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i+1)
    return labels, starts, ends


def levenstein(p, y, norm=False):
    m_row = len(p)    
    n_col = len(y)
    D = np.zeros([m_row+1, n_col+1], np.float)
    for i in range(m_row+1):
        D[i, 0] = i
    for i in range(n_col+1):
        D[0, i] = i

    for j in range(1, n_col+1):
        for i in range(1, m_row+1):
            if y[j-1] == p[i-1]:
                D[i, j] = D[i-1, j-1]
            else:
                D[i, j] = min(D[i-1, j] + 1,
                              D[i, j-1] + 1,
                              D[i-1, j-1] + 1)
    
    if norm:
        score = (1 - D[-1, -1]/max(m_row, n_col)) * 100
    else:
        score = D[-1, -1]

    return score


def edit_score(recognized, ground_truth, norm=True, bg_class=["background"]):
    P, _, _ = get_labels_start_end_time(recognized, bg_class)
    Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return levenstein(P, Y, norm)


def f_score(recognized, ground_truth, overlap, bg_class=["background"]):
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)

    tp = 0
    fp = 0

    hits = np.zeros(len(y_label))

    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0*intersection / union)*([p_label[j] == y_label[x] for x in range(len(y_label))])
        # Get the best scoring segment
        idx = np.array(IoU).argmax()

        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn)


def evaluate(dataset, split, time_data):
    print("Evaluate dataset {} in split {} for single stamp supervision".format(dataset, split))

    bz_stages = '/margin_map_both' + time_data
    recog_path = "./results/" + dataset + bz_stages + "_split_" + split + '/'
    ground_truth_path = "./data/" + dataset+"/groundTruth/"
    file_list = "./data/" + dataset + "/splits/test.split" + split + ".bundle"

    list_of_videos = read_file(file_list).split('\n')[:-1]

    overlap = [.1, .25, .5]
    tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)

    file_name = './result/' + time_data + '.xlsx'
    workbook = xlsxwriter.Workbook(file_name)
    worksheet = workbook.add_worksheet()
    metrics = ['F1@10', 'F1@25', 'F1@50', 'Edit', 'Acc']
    row = 0
    col = 0
    for m in range(len(metrics)):
        worksheet.write(row, col, metrics[m])
        col += 1

    row += 1
    col = 0

    correct = 0
    total = 0
    edit = 0

    for vid in list_of_videos:
        gt_file = ground_truth_path + vid
        gt_content = read_file(gt_file).split('\n')[0:-1]
        
        recog_file = recog_path + vid.split('.')[0]
        recog_content = read_file(recog_file).split('\n')[1].split()

        for i in range(len(gt_content)):
            total += 1
            if gt_content[i] == recog_content[i]:
                correct += 1
        
        edit += edit_score(recog_content, gt_content)

        for s in range(len(overlap)):
            tp1, fp1, fn1 = f_score(recog_content, gt_content, overlap[s])
            tp[s] += tp1
            fp[s] += fp1
            fn[s] += fn1

    for s in range(len(overlap)):
        precision = tp[s] / float(tp[s]+fp[s])
        recall = tp[s] / float(tp[s]+fn[s])
    
        f1 = 2.0 * (precision*recall) / (precision+recall)
        f1 = np.nan_to_num(f1)*100
        print('F1@%0.2f: %.4f' % (overlap[s], f1))

        worksheet.write(row, col, round(f1, 4))
        col += 1

    edit = (1.0 * edit) / len(list_of_videos)
    acc = 100 * float(correct) / total

    worksheet.write(row, col, round(edit, 4))
    worksheet.write(row, col + 1, round(acc, 4))

    print('Edit: %.4f' % edit)
    print("Acc: %.4f" % acc)

    workbook.close()


def evaluate_pseudo_labels(dataset):
    sample_rate = 1
    if dataset == "50salads":
        sample_rate = 2

    random_index = np.load("data/" + dataset + "_annotation_all.npy", allow_pickle=True).item()
    
    recog_path = "./data/I3D_merge/" + dataset + '/'
    ground_truth_path = "./data/" + dataset+"/groundTruth/"

    overlap = [.1, .25, .5]
    tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)

    correct = 0
    label_num = 0
    total = 0
    edit = 0

    for vid in random_index.keys():
        gt_file = ground_truth_path + vid
        gt_content = read_file(gt_file).split('\n')[0:-1:sample_rate]
        
        recog_file = recog_path + vid
        recog_content = read_file(recog_file).split('\n')[0:-1]

        for i in range(len(gt_content)):
            total += 1
            if gt_content[i] == recog_content[i]:
                correct += 1
            if recog_content[i] != 'no_label':
                label_num += 1
        
    #     edit += edit_score(recog_content, gt_content)

    #     for s in range(len(overlap)):
    #         tp1, fp1, fn1 = f_score(recog_content, gt_content, overlap[s])
    #         tp[s] += tp1
    #         fp[s] += fp1
    #         fn[s] += fn1

    # for s in range(len(overlap)):
    #     precision = tp[s] / float(tp[s]+fp[s])
    #     recall = tp[s] / float(tp[s]+fn[s])
    
    #     f1 = 2.0 * (precision*recall) / (precision+recall)
    #     f1 = np.nan_to_num(f1)*100
    #     print('F1@%0.2f: %.4f' % (overlap[s], f1))

    # edit = (1.0 * edit) / len(random_index)
    acc = 100 * float(correct) / label_num
    label_rate = 100 * float(label_num) / total

    print('Edit: %.4f' % edit)
    print("Acc: %.4f" % acc)
    print("Label rate: %.4f" % label_rate)




def segment_bars_with_confidence(save_path, confidence, *labels):
    num_pics = len(labels) + 1
    color_map = plt.get_cmap('turbo').copy()
    color_map.set_under('w')

    axprops = dict(xticks=[], yticks=[], frameon=False)
    barprops = dict(aspect='auto', cmap=color_map, interpolation='nearest', vmin=0)
    fig = plt.figure(figsize=(15, num_pics * 1.5))

    interval = 1 / (num_pics + 1)
    for i, label in enumerate(labels):
        i = i + 1
        ax1 = fig.add_axes([0, 1 - i * interval, 1, interval])
        ax1.imshow([label], **barprops)

    ax4 = fig.add_axes([0, interval, 1, interval])
    ax4.set_xlim(0, len(confidence))
    ax4.set_ylim(0, 1)
    ax4.plot(range(len(confidence)), confidence)
    # ax4.plot(range(len(confidence)), [0.3] * len(confidence), color='red', label='0.5')

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
    
    fig.clear()
    plt.close()


def plot_pseudo_labels(save_path, num_classes, *labels):
    num_pics = len(labels) + 1
    color_map = plt.get_cmap('turbo').copy()
    color_map.set_under('w')
    fig = plt.figure(figsize=(15, num_pics*1.5))
    
    barprops = dict(aspect='auto', cmap=color_map, interpolation='nearest', vmin=0, vmax=num_classes-1)
    
    # dic = {1:'(a)', 2:'(b)', 3:'(c)', 4:'(d)', 5:'(e)', 6:'(f)'}
    for i, label in enumerate(labels):
        plt.subplot(num_pics, 1, i + 1)
        plt.xticks([])
        plt.yticks([])
        # plt.ylabel(dic[i+1]+'      ', rotation = 0, size=20)
        plt.imshow([label], **barprops)

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

    fig.clear()
    plt.close()
