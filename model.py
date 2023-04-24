
from random import shuffle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import os
import copy
import numpy as np
from eval import segment_bars_with_confidence


class MultiStageModel(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes):
        super(MultiStageModel, self).__init__()
        self.dropout2d = nn.Dropout2d(p=0.2)
        self.tower_stage = TowerModel(num_layers, num_f_maps, dim, num_classes)
        self.single_stages = nn.ModuleList([copy.deepcopy(SingleStageModel(num_layers, num_f_maps, num_classes, num_classes, 3))
                                     for s in range(num_stages-1)])

    def forward(self, x, mask):
        x = x.unsqueeze(2)
        x = self.dropout2d(x)
        x = x.squeeze(2)
        middle_out, out = self.tower_stage(x, mask)
        outputs = out.unsqueeze(0)
        for s in self.single_stages:
            middle_out, out = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return middle_out, outputs


class TowerModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(TowerModel, self).__init__()
        self.stage1 = SingleStageModel(num_layers, num_f_maps, dim, num_classes, 3)
        self.stage2 = SingleStageModel(num_layers, num_f_maps, dim, num_classes, 5)

    def forward(self, x, mask):
        out1, final_out1 = self.stage1(x, mask)
        out2, final_out2 = self.stage2(x, mask)

        return out1 + out2, final_out1 + final_out2


class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes, kernel_size):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps, kernel_size))
                                     for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        final_out = self.conv_out(out) * mask[:, 0:1, :]
        return out, final_out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, kernel_size):
        super(DilatedResidualLayer, self).__init__()
        padding = int(dilation + dilation * (kernel_size - 3) / 2)
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x, mask):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]


class Trainer:
    def __init__(self, num_blocks, num_layers, num_f_maps, dim, num_classes):
        self.model = MultiStageModel(num_blocks, num_layers, num_f_maps, dim, num_classes)
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.ce_soft = nn.CrossEntropyLoss(reduction='none')
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = num_classes

    def confidence_loss(self, pred, confidence_mask, device):
        batch_size = pred.size(0)
        pred = F.log_softmax(pred, dim=1)
        loss = 0
        for b in range(batch_size):
            num_frame = confidence_mask[b].shape[2]
            m_mask = torch.from_numpy(confidence_mask[b]).type(torch.float).to(device)
            left = pred[b, :, 1:] - pred[b, :, :-1]
            left = torch.clamp(left[:, :num_frame] * m_mask[0], min=0)
            left = torch.sum(left) / torch.sum(m_mask[0])
            loss += left

            right = (pred[b, :, :-1] - pred[b, :, 1:])
            right = torch.clamp(right[:, :num_frame] * m_mask[1], min=0)
            right = torch.sum(right) / torch.sum(m_mask[1])
            loss += right

        return loss


    def cluster_loss(self, cur_left, cur_right, timestamp_list, label_num, middle_pred):
        """compute cluster loss

        Args:
            bounds ([[],[],...,[]]): a list whose length is batch size, each element is also a list which contains each video's boundaries
            middle_pred (B, D, L): middle representation
        """

        loss = 0.0
        middle_pred = F.normalize(middle_pred, p=2, dim=1)

        for vidx in range(len(cur_left)):
            for i in range(len(cur_left[vidx])):
                center_vec = middle_pred[vidx, :, timestamp_list[vidx][i]]
                # center_vec = torch.mean(middle_pred[vidx, :, cur_left[vidx][i]:cur_right[vidx][i]], dim=1)
                diff_left = middle_pred[vidx, :, cur_left[vidx][i]:cur_right[vidx][i]] - center_vec.reshape(-1, 1)
                loss += torch.sum(torch.norm(diff_left, dim=0))

        return loss / label_num


    def train(self, save_dir, batch_gen, writer, num_epochs, batch_size, learning_rate, device):
        self.model.train()
        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        start_epochs = 50
        print('start epoch of single supervision is:', start_epochs)

        epoch_list = []
        truth_rate = []
        label_rate = []
        
        left_indices_dic, right_indices_dic = batch_gen.get_pseudo_boundary_dic()
        for epoch in range(1, num_epochs+1):

        # self.model.load_state_dict(torch.load('model_firststage_stamp_split1.pkl'))
        # optimizer.load_state_dict(torch.load('optim_firststage_stamp_split1.pth'))
        # for epoch in range(start_epochs+1, num_epochs+1):
            
            truth_num, label_num, total_num = 0, 0, 0
            epoch_loss = 0
            correct = 0
            total = 0
            while batch_gen.has_next():
                batch_input, batch_target, mask, batch_confidence, _ = batch_gen.next_batch(batch_size)
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                optimizer.zero_grad()
                middle_pred, predictions = self.model(batch_input, mask)

                # Generate pseudo labels after training 30 epochs for getting more accurate labels
                if epoch <= start_epochs:
                    batch_boundary, _ = batch_gen.get_single_random(batch_size, batch_input.size(-1))
                    batch_boundary = batch_boundary.to(device)
                else:
                    confs, _ = torch.max(F.softmax(predictions[-1], dim=1).data, 1)
                    batch_boundary, truth_label_total_num, left_indices_dic, right_indices_dic = batch_gen.get_boundary_lp(batch_size, middle_pred.detach(), confs.detach(), left_indices_dic, right_indices_dic)
                    truth_num += sum([each[0] for each in truth_label_total_num])
                    label_num += sum([each[1] for each in truth_label_total_num])
                    total_num += sum([each[2] for each in truth_label_total_num])
                    batch_boundary = batch_boundary.to(device)

                loss = 0.0

                cur_left, cur_right, cur_label_num, timestamp_list = batch_gen.get_cur_boundary(batch_size, left_indices_dic, right_indices_dic)
                loss += 0.15 * self.cluster_loss(cur_left, cur_right, timestamp_list, cur_label_num, middle_pred)

                for p in predictions:
                    if epoch <= start_epochs:
                        loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_boundary.view(-1))
                    else:
                        # loss += 0.2 * self.DEC_loss(bounds, middle_pred, stamp_labels)  # 权重要改为1试试
                        loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_boundary.view(-1))
                        # loss += torch.sum(self.ce_soft(p, batch_boundary) * mask[:, 0, :]) / torch.sum(mask[:, 0, :])
                        # print((torch.sum(self.ce_soft(p, batch_boundary) * p_mask[:, 0, :]) / torch.sum(p_mask[:, 0, :])).item())
                        # print(self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_boundary_info.view(-1)).item())

                    loss += 0.15 * torch.mean(torch.clamp(
                        self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0,
                        max=16) * mask[:, :, 1:])

                    # loss += 2 * self.smooth_loss(p, batch_gen.smooth_mask(batch_size, batch_input.size(-1)).to(device))
                    loss += 0.075 * self.confidence_loss(p, batch_confidence, device)
                
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(F.softmax(predictions[-1], dim=1).data, 1)
                correct += ((predicted == batch_target).float()*mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()

            batch_gen.reset(shuffle=True)
            torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
            torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")
            writer.add_scalar('trainLoss', epoch_loss / len(batch_gen.list_of_examples), epoch + 1)
            writer.add_scalar('trainAcc', float(correct)/total, epoch + 1)

            print("[epoch %d]: epoch loss = %f, acc = %f" % (epoch, epoch_loss / len(batch_gen.list_of_examples),
                                                            float(correct)/total))

            if epoch > start_epochs:
                print("[epoch {}]: truth: {}, label: {}, total: {}, truth_rate: {}, label_rate: {}".format(
                    epoch, truth_num, label_num, total_num, truth_num / float(label_num), label_num / float(total_num)
                ))
                epoch_list.append(epoch)
                truth_rate.append(float(truth_num / float(label_num)))
                label_rate.append(float(label_num / float(total_num)))


    def predict(self, model_dir, results_dir, features_path, batch_gen_tst, epoch, actions_dict, device, sample_rate):
        self.model.eval()
        with torch.no_grad():
            self.model.to(device)
            self.model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))
            batch_gen_tst.reset()

            while batch_gen_tst.has_next():
                batch_input, batch_target, mask, batch_confidence, vids = batch_gen_tst.next_batch(1)
                vid = vids[0]
                # print(vid)
                features = np.load(features_path + vid.split('.')[0] + '.npy')
                features = features[:, ::sample_rate]
                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)
                _, predictions = self.model(input_x, torch.ones(input_x.size(), device=device))
                # _, predicted = torch.max(predictions[-1].data, 1)
                # predicted = predicted.squeeze()

                for i in range(len(predictions)):
                    confidence, predicted = torch.max(F.softmax(predictions[i], dim=1).data, 1)
                    confidence, predicted = confidence.squeeze(), predicted.squeeze()
 
                    batch_target = batch_target.squeeze()
                    confidence, predicted = confidence.squeeze(), predicted.squeeze()
 
                    segment_bars_with_confidence(results_dir + '/{}_stage{}.png'.format(vid, i),
                                                 confidence.tolist(),
                                                 batch_target.tolist(), predicted.tolist())

                recognition = []
                for i in range(len(predicted)):
                    index = list(actions_dict.values()).index(predicted[i].item())
                    recognition = np.concatenate((recognition, [list(actions_dict.keys())[index]]*sample_rate))

                f_name = vid.split('/')[-1].split('.')[0]
                f_ptr = open(results_dir + "/" + f_name, "w")
                f_ptr.write("### Frame level recognition: ###\n")
                f_ptr.write(' '.join(recognition))
                f_ptr.close()
