import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

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

    plt.close()


def eval_pseudo_labels(pseudo_labels, gt_labels):
    true_num = np.sum(pseudo_labels == gt_labels)
    pseudo_num = len(gt_labels) - np.sum(pseudo_labels == -1)
    return true_num, pseudo_num


def intersection_labels(*labels):
    assert len(labels) >= 2
    out_labels = np.zeros_like(labels[0]) - 1
    out_labels[labels[0]==labels[1]] = labels[0][labels[0]==labels[1]]
    for i in range(2, len(labels)):
        out_labels[~(out_labels==labels[i])] = -1
    return out_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="50salads", help='three dataset: breakfast, 50salads, gtea')
    parser.add_argument('--type', default="all", help='all, energy_function, constrained_k_medoids, agglomerative_clustering, constrained_agens')

    args = parser.parse_args()

    dataset_name = args.dataset #'gtea' #'50salads'
    sample_rate = 1
    if dataset_name == "50salads":
        sample_rate = 2

    pseudo_label_dir1 = "data/I3D_1024/"+dataset_name+"/"
    pseudo_label_dir2 = "data/I3D_2048/"+dataset_name+"/"

    pseudo_label_dir = "data/I3D_merge/"+dataset_name+"/"
    if not os.path.exists(pseudo_label_dir):
        os.makedirs(pseudo_label_dir)
        
    # read action dict
    file_ptr = open("data/" + dataset_name + "/mapping.txt", 'r')
    actions = file_ptr.read().split('\n')[:-1]
    file_ptr.close()
    actions_dict = dict()
    reverse_dict = dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])
        reverse_dict[int(a.split()[0])] = a.split()[1]
    reverse_dict[-1] = 'no_label'


    random_index = np.load("data/" + dataset_name + "_annotation_all.npy", allow_pickle=True).item()

    total_true = 0
    total_pseudo = 0
    total_length = 0

    # process each video
    for vid, stamp in random_index.items():

        # read labels
        file_ptr = open("data/" + dataset_name + "/groundTruth/" + vid, 'r')  
        content = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        classes = np.zeros(len(content))
        for i in range(len(classes)):
            classes[i] = actions_dict[content[i]]
        classes = classes[::sample_rate]

        file_ptr = open(pseudo_label_dir1 + vid, 'r')
        content = file_ptr.read().split('\n')[:-1]
        output_classes1 = np.zeros(len(content)) - 1
        for i in range(len(content)):
            output_classes1[i] = actions_dict.get(content[i], -1)
        file_ptr.close()

        file_ptr = open(pseudo_label_dir2 + vid, 'r')
        content = file_ptr.read().split('\n')[:-1]
        output_classes2 = np.zeros(len(content)) - 1
        for i in range(len(content)):
            output_classes2[i] = actions_dict.get(content[i], -1)
        file_ptr.close()

        output_classes = intersection_labels(output_classes1, output_classes2)
        true_num, pseudo_num = eval_pseudo_labels(output_classes, classes)

        print(vid.split('.')[0] + "    true num: {}, pseudo labels num: {}, length: {}, stop iter: {}".format(true_num, pseudo_num, len(classes), 0))
        
        total_true += true_num
        total_pseudo += pseudo_num
        total_length += len(classes)

        # save pseudo label
        file_ptr = open(pseudo_label_dir+vid, 'w')
        for each in output_classes:
            file_ptr.write(reverse_dict[each] + '\n')
        file_ptr.close()

        plot_pseudo_labels(pseudo_label_dir+vid.split('.')[0]+'.pdf', len(actions_dict), classes, output_classes)
        # plot_pseudo_labels(pseudo_label_dir+vid.split('.')[0]+'.pdf', len(actions_dict), classes, ts_only(stamp,features,classes)[0], output_classes1, output_classes2, output_classes3, output_classes)

    print("label rate: {}".format(total_pseudo/float(total_length)))
    print("label acc: {}".format(total_true/float(total_pseudo)))
