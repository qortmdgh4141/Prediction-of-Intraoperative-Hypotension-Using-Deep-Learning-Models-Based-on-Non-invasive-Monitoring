from src.utils.att_map import savefig, draw_graph
from src.train import train_function
import numpy as np
import pandas as pd
import os
import random
import torch

# #
# ## Class that handle attention records and save them label by label
# ## test on dummy model?
class Attn_Saver:
    def __init__(self, log_path: str):
        #memories
        self.attns = []
        self.gattns = []

        #self.x = []
        self.y = []
        self.py = []

        #path for data save
        self.log_path = log_path

        #index by case
        self.classified_index = [[[] for i in range(2)] for j in range(2)]

    # hook function which will save attention map
    # self.model.inter_chunk_block1.multihead_attn.register_forward_hook(hook)
    # actual access of intermediate output
    # if hook has trouble getting intermediate output, just use set function to save on the class list
    def hook(self):
        def hookin(module, input, output):
            att_list = [output[1].detach().cpu().unsqueeze(0) for i in range(len(output[1]))]
            self.attns += att_list
            print('x')
            print(len(output[1]))
            print(len(output[1].shape))
        return hookin


    ##############################################################################################
    # initially collects data ####################################################################
    ##############################################################################################
    def set_att(self, att_list):
        att = [att_list[i].detach().cpu().unsqueeze(0) for i in range(len(att_list))]
        self.attns += att

    def set_gatt(self,gatt_list):
        gatt = [gatt_list[i].detach().cpu().unsqueeze(0) for i in range(len(gatt_list))]
        self.gattns += gatt

    def set_raw_data(self, x_list):
        x_list = [x_list[i].detach().cpu() for i in range(len(x_list))]
        self.x += x_list

    def set_labels(self, y, py):
        self.y += y
        self.py += py

    ##index groups : (actual_label, predicted_label)
    def classify(self):
        labels = [0, 1]
        for i in range(len(self.y)):
            for j in labels:
                for k in labels:
                    if labels[j] == self.y[i] and labels[k] == self.py[i]:
                        self.classified_index[j][k].append(i)

    # intermediate function
    def pick_random_attn(self, y, pred_y):
        index = self.classified_index[y][pred_y]

        rand_i = random.randrange(0, len(index))
        index = index[rand_i]  ## random index
        raw_data = self.x[index]
        attention = self.attns[index]
        attention = torch.mean(attention, dim=1)
        gattention = self.gattns[index]
        gattention = torch.mean(gattention, dim=1)

        return gattention, attention, raw_data

    ##save random attention image with given label
    def save_average_attn(self):
        labels = [0, 1]
        for j in labels:  # #real label
            for k in labels:  # #predicted label
                attn_array_index = self.classified_index[j][k]
                if len(self.classified_index[j][k]) == 0:
                    continue
                if len(self.classified_index[j][k]) == 1:
                    average = self.attns[attn_array_index[j][k]]
                else:
                    attn_array = torch.concat([self.attns[i] for i in attn_array_index], dim=0)
                    average = torch.mean(attn_array, dim=1)
                savefig(average, self.log_path, 0, (8, 1), 'avg_att_{}_'.format(j) + '{}'.format(k))

    def save_average_gattn(self):
        labels = [0, 1]
        for j in labels:  # #real label
            for k in labels:  # #predicted label
                attn_array_index = self.classified_index[j][k]
                if len(self.classified_index[j][k]) == 0:
                    continue
                if len(self.classified_index[j][k]) == 1:
                    average = self.gattns[attn_array_index[j][k]]
                else:
                    attn_array = torch.concat([self.gattns[i] for i in attn_array_index], dim=0)
                    average = torch.mean(attn_array, dim=1)
                savefig(average, self.log_path, 0, (40, 10), 'avg_gatt_{}_'.format(j) + '{}'.format(k), True)
                savefig(average, self.log_path, 0, (40, 10), 'avg_gatt_h_{}_'.format(j) + '{}'.format(k))

    def save_random_attn(self, y, pred_y, index):
        if len(self.classified_index[y][pred_y]) == 0:
            return
        gattn, attn, data = self.pick_random_attn(y, pred_y)
        savefig(attn, self.log_path, index, (8, 1), 'att_{}_'.format(y) + '{}'.format(pred_y))
        savefig(gattn, self.log_path, index, (40, 10), 'gatt_{}_'.format(y) + '{}'.format(pred_y), True)
        draw_graph(data, self.log_path, index, 'data_{}_'.format(y) + '{}'.format(pred_y))

    def save_attn(self):

        # total 8 files are made
        labels = [0,1]
        for j in labels:  # #real label
            for k in labels:  # #predicted label
                attn_array_index = self.classified_index[j][k]
                attn_array = np.array([self.attns[i].numpy() for i in attn_array_index],dtype=object)
                print(attn_array.shape)
                gattn_array = np.array([self.gattns[i].numpy() for i in attn_array_index],dtype=object)
                print(gattn_array.shape)
                #data_array = np.array([self.x[i] for i in attn_array_index])

                np.save('matching2_gattn{}'.format(j) + '{}'.format(k) + '.npy', gattn_array)
                np.save('matching2_attn{}'.format(j) + '{}'.format(k) + '.npy', attn_array)

                #np.save('raw_data{}'.format(j) + '{}'.format(k) + '.npy', data_array)

    #called during train or test
    def preparing(self, y, py, x_list, gatt, att):
        #self.set_raw_data(x_list)
        self.set_labels(y, py)
        self.set_gatt(gatt)
        self.set_att(att)

    #called after train or test
    def whole_process(self):
        self.classify()
        self.save_attn()

        self.save_average_attn()
        self.save_average_gattn()

        #for i in range(3):
        #    self.save_random_attn(1, 1, i)
        #    self.save_random_attn(0, 0, i)
        #    self.save_random_attn(0, 1, i)
        #    self.save_random_attn(1, 0, i)
        
