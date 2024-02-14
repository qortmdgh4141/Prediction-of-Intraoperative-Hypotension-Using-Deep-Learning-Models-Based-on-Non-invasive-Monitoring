import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import random
import sys


#  Solar8000/NIBP_SBP < 90

if __name__ == "__main__":

    #1 clinical file list
    #2 test file list

    #file_lists = glob.glob(os.path.join('./data/clinical_dataset/original/', "*.csv"))
    #group_name = 'clinic'
    outliers=[]
    with open("./outliers.txt") as f:
        for line in f:
            outliers.append(line.strip())
    print(outliers)
    sys.exit()
    Test = False

    file_lists = glob.glob(os.path.join('./data/dataset2/original/', "*.csv")) ######
    group_name = 'original'

    if Test:
        random.seed(42)
        random.shuffle(file_list_o)
        n_of_test = int(len(file_list_o) * 0.2)
        file_list_o = file_list_o[:n_of_test]

    print('total_' + group_name + '_people:{}'.format(len(file_lists)))

    total_df = []
    columns = []

    count = 0
    for file in file_lists:
        df = pd.read_csv(file) ## dataframe
        columns = df.columns[1:]
        count += 1

        fi = -1
        li = float('inf')

        #check the rage where total column is valid
        for column in columns :
            first_idx = df[column].first_valid_index()
            last_idx =df[column].last_valid_index()
            if li > last_idx:
                li = last_idx
            if fi < first_idx:
                fi = first_idx

        df = df.loc[fi:li]
        print('current individual..:{}'.format(count))
        print('item count..:{}'.format(len(df)))

        mdf = df.mean()

        dict = {k: [mdf[k]] for k in columns }
        dict['group'] = group_name
        dict['pname'] = file

        print('calculated mean...')
        print(dict)
        total_df.append(pd.DataFrame(data=dict))###

    count = 0

    all_data = pd.concat(total_df, axis=0, ignore_index=True)
    print(all_data)



    part = True
    part_data = all_data
    sub_name = ''

    if part:
        filt = (all_data['BIS/BIS'] <= 20.0)
        # filt = (all_data['Primus/MAC'] >= 0.4)
        part_data = all_data.loc[filt]

        sub_name = 'b_shared_'
        print(part_data)

    outliers = part_data['pname']
    with open("outliers.txt", "w") as f:
        for i in outliers:
            print(i)
            f.write(str(i) + "\n")

    sys.exit()

    part_data.plot.box(by='group', figsize=(40,10))
    plt.savefig(os.path.join('./logs', sub_name+'part_box'))
    plt.clf()

    for column in columns :
        sns.histplot(part_data, x=column, hue='group', bins=200, stat='count', common_norm=False)
        plt.title(sub_name + 'Histogram_'+column)
        plt.savefig(os.path.join('./logs', sub_name+'Histogram_'+column[-3:]))
        plt.clf()

    print("check the result")
