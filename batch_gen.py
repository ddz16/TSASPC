
import torch
import torch.nn.functional as F
import numpy as np
import random


class BatchGenerator(object):
    def __init__(self, num_classes, actions_dict, gt_path, features_path, sample_rate, pseudo_path):
        self.list_of_examples = list()
        self.index = 0
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.gt_path = gt_path
        self.features_path = features_path
        self.sample_rate = sample_rate
        self.gt = {}
        self.confidence_mask = {}
        self.pseudo_path = pseudo_path
        dataset_name = gt_path.split('/')[-3]
        assert dataset_name in ['50salads', 'breakfast', 'gtea']
        self.random_index = np.load('./data/' + dataset_name + "_annotation_all.npy", allow_pickle=True).item()

    def reset(self, shuffle=False):
        self.index = 0
        if shuffle:
            random.shuffle(self.list_of_examples)

    def has_next(self):
        if self.index < len(self.list_of_examples):
            return True
        return False
    
    def get_len(self):
        return len(self.list_of_examples)
    
    def get_num_classes(self):
        return self.num_classes

    def read_data(self, vid_list_file):
        file_ptr = open(vid_list_file, 'r')
        self.list_of_examples = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        random.shuffle(self.list_of_examples)
        self.generate_confidence_mask()

    def generate_confidence_mask(self):
        for vid in self.list_of_examples:
            file_ptr = open(self.gt_path + vid, 'r')
            content = file_ptr.read().split('\n')[:-1]
            classes = np.zeros(len(content))
            for i in range(len(classes)):
                classes[i] = self.actions_dict[content[i]]
            classes = classes[::self.sample_rate]
            self.gt[vid] = classes
            num_frames = classes.shape[0]

            random_idx = self.random_index[vid]

            # Generate mask for confidence loss. There are two masks for both side of timestamps
            left_mask = np.zeros([self.num_classes, num_frames - 1])
            right_mask = np.zeros([self.num_classes, num_frames - 1])
            for j in range(len(random_idx) - 1):
                left_mask[int(classes[random_idx[j]]), random_idx[j]:random_idx[j + 1]] = 1
                right_mask[int(classes[random_idx[j + 1]]), random_idx[j]:random_idx[j + 1]] = 1

            self.confidence_mask[vid] = np.array([left_mask, right_mask])

    def next_batch(self, batch_size):
        batch = self.list_of_examples[self.index:self.index + batch_size]
        self.index += batch_size

        batch_input = []
        batch_target = []
        batch_confidence = []
        for vid in batch:
            features = np.load(self.features_path + vid.split('.')[0] + '.npy')
            batch_input.append(features[:, ::self.sample_rate])
            batch_target.append(self.gt[vid])
            batch_confidence.append(self.confidence_mask[vid])

        length_of_sequences = list(map(len, batch_target))
        batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0], max(length_of_sequences), dtype=torch.float)
        batch_target_tensor = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long)*(-100)
        mask = torch.zeros(len(batch_input), self.num_classes, max(length_of_sequences), dtype=torch.float)
        for i in range(len(batch_input)):
            batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i])
            batch_target_tensor[i, :np.shape(batch_target[i])[0]] = torch.from_numpy(batch_target[i])
            mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(self.num_classes, np.shape(batch_target[i])[0])

        # batch_input_tensor: (B,D,L)       batch_target_tensor: (B,L)       mask: (B,C,L) [1 is gt; 0 is padding]
        # batch_confidence: a list (length=B) where each element is numpy array (2,C,L-1), L is each video length in the mini-batch 
        return batch_input_tensor, batch_target_tensor, mask, batch_confidence, batch

    def get_single_random(self, batch_size, max_frames):
        # Generate target for only timestamps. Do not generate pseudo labels at first 30 epochs.
        batch = self.list_of_examples[self.index - batch_size:self.index]
        boundary_target_tensor = torch.ones(len(batch), max_frames, dtype=torch.long) * (-100)
        mask = torch.zeros(len(batch), self.num_classes, max_frames, dtype=torch.float)
        for b, vid in enumerate(batch):
            single_frame = self.random_index[vid]
            gt = self.gt[vid]
            frame_idx_tensor = torch.from_numpy(np.array(single_frame))
            gt_tensor = torch.from_numpy(gt.astype(int))
            boundary_target_tensor[b, frame_idx_tensor] = gt_tensor[frame_idx_tensor]
            mask[b, :, frame_idx_tensor] = 1
        
        return boundary_target_tensor, mask  # (B,L), only the timestamps have labels, other positions are -100

    
    def get_pseudo_boundary_dic(self):
        left_indices_dic = {}
        right_indices_dic = {}

        for vid in self.list_of_examples:
            left = [0]
            right = []
            flag = True
            file_ptr = open(self.pseudo_path + vid, 'r')
            content = file_ptr.read().split('\n')[:-1]
            file_ptr.close()
            for i in range(1, len(content)):
                if content[i] != content[i-1] and flag and content[i] == 'no_label':
                    right.append(i)
                    flag = False
                elif content[i] != content[i-1] and flag and content[i] != 'no_label':
                    right.append(i)
                    left.append(i)
                elif content[i] != content[i-1] and not flag:
                    left.append(i)
                    flag = True
            right.append(len(content))
            left_indices_dic[vid] = left
            right_indices_dic[vid] = right

        return left_indices_dic, right_indices_dic


    def smooth_mask(self, batch_size, max_frames):
        batch = self.list_of_examples[self.index - batch_size:self.index]

        mask = torch.zeros(len(batch), self.num_classes, max_frames, dtype=torch.float)
        for b, vid in enumerate(batch):
            single_frame = list(map(int, self.random_index[vid]))
            gt = self.gt[vid].astype(int)
            mask[b, gt[single_frame[0]], 0:single_frame[1]] = 1
            for i in range(1, len(single_frame)-1):
                mask[b, gt[single_frame[i]], single_frame[i-1]:single_frame[i+1]] = 1
            mask[b, gt[single_frame[-1]], single_frame[-2]:] = 1
            
        return mask  # (B,C,L)


    def get_boundary(self, batch_size, pred):
        # This function is to generate pseudo labels

        batch = self.list_of_examples[self.index - batch_size:self.index]
        num_video, _, max_frames = pred.size()
        boundary_target_tensor = torch.ones(num_video, max_frames, dtype=torch.long) * (-100)

        bounds = []
        stamp_labels = []
        truth_label_total_num = []
        for b, vid in enumerate(batch):
            single_idx = self.random_index[vid]
            vid_gt = self.gt[vid]
            features = pred[b]
            boundary_target = np.ones(vid_gt.shape) * (-100)
            boundary_target[:single_idx[0]] = vid_gt[single_idx[0]]  # frames before first single frame has same label
            left_bound = [0]

            # Forward to find action boundaries
            for i in range(len(single_idx) - 1):
                start = single_idx[i]
                end = single_idx[i + 1] + 1
                left_score = torch.zeros(end - start - 1, dtype=torch.float)
                for t in range(start + 1, end):
                    center_left = torch.mean(features[:, left_bound[-1]:t], dim=1)
                    diff_left = features[:, start:t] - center_left.reshape(-1, 1)
                    score_left = torch.mean(torch.norm(diff_left, dim=0))

                    center_right = torch.mean(features[:, t:end], dim=1)
                    diff_right = features[:, t:end] - center_right.reshape(-1, 1)
                    score_right = torch.mean(torch.norm(diff_right, dim=0))

                    left_score[t-start-1] = ((t-start) * score_left + (end - t) * score_right)/(end - start)

                cur_bound = torch.argmin(left_score) + start + 1
                left_bound.append(cur_bound.item())

            # Backward to find action boundaries
            right_bound = [vid_gt.shape[0]]
            for i in range(len(single_idx) - 1, 0, -1):
                start = single_idx[i - 1]
                end = single_idx[i] + 1
                right_score = torch.zeros(end - start - 1, dtype=torch.float)
                for t in range(end - 1, start, -1):
                    center_left = torch.mean(features[:, start:t], dim=1)
                    diff_left = features[:, start:t] - center_left.reshape(-1, 1)
                    score_left = torch.mean(torch.norm(diff_left, dim=0))

                    center_right = torch.mean(features[:, t:right_bound[-1]], dim=1)
                    diff_right = features[:, t:end] - center_right.reshape(-1, 1)
                    score_right = torch.mean(torch.norm(diff_right, dim=0))

                    right_score[t-start-1] = ((t-start) * score_left + (end - t) * score_right)/(end - start)

                cur_bound = torch.argmin(right_score) + start + 1
                right_bound.append(cur_bound.item())

            # Average two action boundaries for same segment and generate pseudo labels
            left_bound = left_bound[1:]
            right_bound = right_bound[1:]
            middle_bound_list = []
            stamp_label = []
            num_bound = len(left_bound)
            for i in range(num_bound):
                temp_left = left_bound[i]
                temp_right = right_bound[num_bound - i - 1]
                middle_bound = int((temp_left + temp_right)/2)
                middle_bound_list.append(middle_bound)
                boundary_target[single_idx[i]:middle_bound] = vid_gt[single_idx[i]]
                boundary_target[middle_bound:single_idx[i + 1] + 1] = vid_gt[single_idx[i + 1]]
                stamp_label.append(vid_gt[single_idx[i]])

            boundary_target[single_idx[-1]:] = vid_gt[single_idx[-1]]  # frames after last single frame has same label
            boundary_target_tensor[b, :vid_gt.shape[0]] = torch.from_numpy(boundary_target)
            stamp_label.append(vid_gt[single_idx[-1]])

            truth_num = np.sum(boundary_target == vid_gt)
            truth_label_total_num.append((truth_num, vid_gt.shape[0], vid_gt.shape[0]))

            middle_bound_list.insert(0, 0)
            middle_bound_list.append(vid_gt.shape[0])
            bounds.append(middle_bound_list)
            stamp_labels.append(stamp_label)
        
        for i in range(len(bounds)):
            assert len(bounds[i]) - 1 == len(stamp_labels[i])

        return boundary_target_tensor, bounds, stamp_labels, truth_label_total_num


    def get_boundary_lp(self, batch_size, pred, confs, left_indices_dic, right_indices_dic):
        """ generate pseudo labels by temporal label propagation
        Args:
            batch_size: batch size
            pred (B, D, L): middle representation
            left_indices_dic: ({vid: []}), the value list contains each segment's left boundaries
            right_indices_dic: ({vid: []}), the value list contains each segment's right boundaries
        """

        batch = self.list_of_examples[self.index - batch_size:self.index]
        num_video, _, max_frames = pred.size()
        boundary_target_tensor = torch.ones(num_video, max_frames, dtype=torch.long) * (-100)

        truth_label_total_num = []

        for b, vid in enumerate(batch):
            label_num = 0
            truth_num = 0
            single_idx = self.random_index[vid]
            vid_gt = self.gt[vid]
            confidence = confs[b]
            features = pred[b]  # (D, L)
            features = F.normalize(features, p=2, dim=0)
            boundary_target = np.ones(vid_gt.shape) * (-100)
            boundary_target[:single_idx[0]] = vid_gt[single_idx[0]]  # frames before first single frame has same label
            label_num += single_idx[0]
            truth_num += single_idx[0]
            
            # compute and combine the mean vector of each segment
            seg_mean_vec_list = []

            for i in range(len(left_indices_dic[vid])):
                seg_start = left_indices_dic[vid][i]
                seg_end = right_indices_dic[vid][i]
                # print("begin2end", seg_start, seg_end, single_idx[i])
                seg_mean_vec = torch.mean(features[:, seg_start:seg_end], dim=1)
                # seg_mean_vec = features[:, single_idx[i]]
                seg_mean_vec_list.append(seg_mean_vec)

            seg_mean_vecs = torch.stack(seg_mean_vec_list, dim=0)  # (n, D), n is segment num

            # compute the distance between any pair of frame feature and segment mean vector
            # features = F.avg_pool1d(features, kernel_size=5, stride=1, padding=2, count_include_pad=False)
            dis_matrix = torch.cdist(features.T, seg_mean_vecs, p=2)  # (L, n)
            # min_indices = torch.argmin(dis_matrix, dim=1).tolist()  # (L)
            # target_list = [vid_gt[single_idx[each]] for each in min_indices]  # (L)

            # label propagation
            new_left_indices = [0]
            new_right_indices = []


            # for i in range(len(single_idx)):
            #     # print("timestamp: ", )
            #     confidence[single_idx[i]] = confidence[single_idx[i]] * 100
            #     print(confidence[left_indices_dic[vid][i]:right_indices_dic[vid][i]])

            for i in range(len(single_idx)-1):
                # l, r = single_idx[i], single_idx[i+1]
                l, r = right_indices_dic[vid][i], left_indices_dic[vid][i+1]-1

                while l < r and dis_matrix[l][i] <= dis_matrix[l][i+1] and confidence[l] >= 0.0:
                    l += 1
                new_right_indices.append(l)

                while l <= r and dis_matrix[r][i] >= dis_matrix[r][i+1] and confidence[r] >= 0.0:
                    r -= 1
                new_left_indices.append(r+1)

            new_right_indices.append(vid_gt.shape[0])

            boundary_target[single_idx[-1]:] = vid_gt[single_idx[-1]]  # frames after last single frame has same label

            left_indices_dic[vid] = new_left_indices
            right_indices_dic[vid] = new_right_indices

            for i in range(len(new_left_indices)):
                boundary_target[new_left_indices[i]:new_right_indices[i]] = vid_gt[single_idx[i]]
           
            # for i in range(len(new_left_indices)-1):
            #     if new_left_indices[i+1] > new_right_indices[i]:
            #         before = torch.cumsum(dis_matrix[new_right_indices[i]:new_left_indices[i+1], i], dim=0)
            #         reverse_idx = [idx for idx in range(new_left_indices[i+1]-new_right_indices[i]-1, -1, -1)]
            #         reverse_idx = torch.LongTensor(reverse_idx).to(before.device)
            #         after = torch.cumsum(dis_matrix[new_right_indices[i]:new_left_indices[i+1], i+1].index_select(0, reverse_idx), dim=0)
            #         after = after.index_select(0, reverse_idx)
            #         bound_offset = torch.argmin(before+after)
            #         boundary_target[new_right_indices[i]:new_right_indices[i]+bound_offset] = vid_gt[single_idx[i]]
            #         boundary_target[new_right_indices[i]+bound_offset:new_left_indices[i+1]] = vid_gt[single_idx[i+1]]

            truth_num = np.sum(boundary_target == vid_gt)
            label_num = vid_gt.shape[0] - np.sum(boundary_target == -100)
            truth_label_total_num.append((truth_num, label_num, vid_gt.shape[0]))

            boundary_target_tensor[b, :vid_gt.shape[0]] = torch.from_numpy(boundary_target)
            
        return boundary_target_tensor, truth_label_total_num, left_indices_dic, right_indices_dic
    

    def get_cur_boundary(self, batch_size, left_indices_dic, right_indices_dic):
        batch = self.list_of_examples[self.index - batch_size:self.index]
        cur_left = []
        cur_right = []
        timestamp_list = []
        cur_label_num = 0
        for b, vid in enumerate(batch):
            timestamp_list.append(self.random_index[vid])
            cur_left.append(left_indices_dic[vid])
            cur_right.append(right_indices_dic[vid])
            for i in range(len(left_indices_dic[vid])):
                cur_label_num += right_indices_dic[vid][i] - left_indices_dic[vid][i]
        
        return cur_left, cur_right, cur_label_num, timestamp_list
