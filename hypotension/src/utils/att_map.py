import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import os
import matplotlib as mpl
import pickle

def save_attn(data, log_path, epoch, shape, tag='', gatt = False):
    return 0

def savefig(data, log_path, epoch, shape, tag='', gatt = False):

    data = torch.mean(data, dim=0)

    data1 = data
    data = data.unsqueeze(0)

    s = data.cpu().detach().numpy()
    s1 = data1.cpu().detach().numpy()

    plt.clf()
    plt.rc('font', size=20)



    ##데이터 나누는 방식.
    c_dist = pd.DataFrame(s)

    array = {'attention': s1}
    c_dist1 = pd.DataFrame(array)

    # Round for data division convenience.
    c_dist = c_dist.round(2)

    # To plot
    #sns.set(rc={'figure.figsize': shape})
    plt.figure(figsize=shape)



    ax = sns.heatmap(c_dist, cmap='YlGnBu', annot=True, cbar=False)
    ax.set(xlabel='feature', ylabel='att_score')
    figure = ax.get_figure()


    # Save to Image
    if not gatt :
        figure.savefig(os.path.join(log_path,'{}'.format(epoch) + tag+'_attention.png'), format='png')

    if gatt :
        plt.clf()
        sns.lineplot(data=c_dist1)
        plt.ylim([0.0, 0.04])
        for i in range(31):
            plt.axvline(x=1.0 * i -0.5, color='r', linestyle='--', linewidth=1)
        plt.savefig(os.path.join(log_path,'{}'.format(epoch) + tag+'_gattention.png'))

    return 0

'''

xData = np.array([1, 2, 3, 4, 5])
yData = np.array([1, 4, 9, 16, 25])
array = {'Number': xData, 'Square': yData}
 
DataFrame_lineplot = pd.DataFrame(array)
 
sns.lineplot(data=DataFrame_lineplot)
pltsw.show()
'''

###remake
## draw graph of data.
def draw_graph(data,log_path,epoch ,tag):
    #data = torch.mean(data, dim=0)
    #data = data.unsqueeze(0)
    #s = data
    s = data.cpu().detach().numpy()
    plt.clf()

    dd = [s[:, i] for i in range(8)]


    array = {'{}'.format(i): dd[i] for i in range(8) if i!=5 and i!= 0 and i!=7}

    array1 = {'0': dd[0] }

    array2 = {'5': dd[5] }

    array3 = {'7': dd[7] }

    fig, axe1 = plt.subplots()




    c_dist = pd.DataFrame(array)
    c_dist1 = pd.DataFrame(array1)
    c_dist2 = pd.DataFrame(array2)
    c_dist3 = pd.DataFrame(array3)

    ##데이터 나누는 방식.
    #c_dist = pd.DataFrame(s)

    # Round for data division convenience.
    #c_dist = c_dist.round(2)
    plt.figure(figsize=(40, 10))

    sns.lineplot(data=c_dist)
    for i in range(31):
        plt.axvline(x=100*i, color='r', linestyle='--', linewidth=3)




    # To plot

    #sns.lineplot([i for i in range(3000)])
    plt.savefig(os.path.join(log_path,'{}'.format(epoch) + tag+ '_raw_data_{}.png'.format(epoch)))
    plt.clf()
    plt.figure(figsize=(40,10))
    sns.lineplot(data=c_dist1)
    for i in range(31):
        plt.axvline(x=100*i, color='r', linestyle='--', linewidth=1)

    plt.savefig(os.path.join(log_path,'{}'.format(epoch) + tag+  '_raw_data_ecg{}.png'.format(epoch)))

    plt.clf()
    plt.figure(figsize=(40, 10))

    sns.lineplot(data=c_dist2)
    for i in range(31):
        plt.axvline(x=100*i, color='r', linestyle='--', linewidth=1)

    plt.savefig(os.path.join(log_path,'{}'.format(epoch) + tag+ '_raw_data_mac{}.png'.format(epoch)))
    plt.clf()
    plt.figure(figsize=(50, 5))

    sns.lineplot(data=c_dist3)
    for i in range(31):
        plt.axvline(x=100*i, color='r', linestyle='--', linewidth=1)

    plt.savefig(os.path.join(log_path, '{}'.format(epoch) + tag+ '_raw_data_bis{}.png'.format(epoch)))

    #figure = ax.get_figure()

    # Save to Image
    #figure.savefig(os.path.join(log_path, tag+'_raw_data_{}.png'.format(epoch)), format='png')

    return 0

