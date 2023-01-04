import csv
import pandas as pd


def read_csv():
    data = []
    with open('./cluster_results/mapped_index.csv', 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for line in reader:
            data.append(line)
    return data


def sampling(FLAG_POSITIVE_SAMPLE, FLAG_NEGATIVE_SAMPLE,
             positive_sampling_number, negative_sampling_number, data, dataset_name):
    total_sample = data
    total_sample_list = ['bug_id', 'mapped_vector_bugId', 'owner_id']
    total_sample = pd.DataFrame(total_sample, columns=total_sample_list)
    total_sample = total_sample.sort_values(by=['owner_id'], ascending=True)
    pn_sample_list = ['owner_id', 'bug_id', 'rating']
    pn_sample = pd.DataFrame(columns=pn_sample_list)
    n1 = len(total_sample)
    flag = 0
    res = 0
    for i in range(n1):
        pn_sample.loc[res] = (total_sample.loc[i][2], total_sample.loc[i][0], 4)
        res += 1
        if i % (n1 // 10) == 0:
            print('resampling：' + str(i / (n1 // 100)) + '%')
        if FLAG_POSITIVE_SAMPLE == 1:
            np = positive_sampling_number
            tmp = total_sample[total_sample['owner_id'] == total_sample.loc[i][2]][['bug_id']]
            while np > 0:

                tmp1 = total_sample[total_sample['mapped_vector_bugId'] == total_sample.loc[i][1]][['bug_id']]
                p = tmp1.sample(n=1, axis=0).values
                positive = p[0][0]
                while positive in tmp['bug_id'].values or positive in \
                        pn_sample.loc[pn_sample.owner_id == total_sample.loc[i][2]]['bug_id'].values:

                    tmp1 = tmp1.drop(index=tmp1[tmp1['bug_id'] == positive].index[0])

                    if len(tmp1) == 0:
                        flag = 1
                        break

                    p = tmp1.sample(n=1, axis=0).values
                    positive = p[0][0]

                if flag == 0:
                    pn_sample.loc[res] = (total_sample.loc[i][2], positive, 3)
                    tmp.loc[len(tmp)] = positive
                    res += 1
                    np -= 1
                else:
                    flag = 0
                    break
        if FLAG_NEGATIVE_SAMPLE == 1:
            nn = negative_sampling_number
            tmp = total_sample[total_sample['owner_id'] == total_sample.loc[i][2]][['bug_id']]
            while nn > 0:

                tmp2 = total_sample[total_sample['mapped_vector_bugId'] != total_sample.loc[i][1]][['bug_id']]
                n = tmp2.sample(n=1, axis=0).values
                negative = n[0][0]
                while negative in tmp['bug_id'].values or negative in \
                        pn_sample.loc[pn_sample.owner_id == total_sample.loc[i][2]]['bug_id'].values:

                    tmp2 = tmp2.drop(index=tmp2[tmp2['bug_id'] == negative].index[0])
                    if len(tmp2) == 0:
                        flag = 1
                        break
                    p = tmp2.sample(n=1, axis=0).values
                    negative = p[0][0]

                if flag == 0:
                    pn_sample.loc[res] = (total_sample.loc[i][2], negative, 1)
                    tmp.loc[len(tmp)] = negative
                    res += 1
                    nn -= 1
                else:
                    flag = 0
                    break
    if dataset_name == 'google_chromium':
        pre = 'chrome'
    elif dataset_name == 'mozilla_core':
        pre = 'core'
    elif dataset_name == 'mozilla_firefox':
        pre = 'firefox'
    name = ('./sample_results/' + pre + '-' + 'augmented' + '-' + str(FLAG_POSITIVE_SAMPLE)
            + '-' + str(FLAG_NEGATIVE_SAMPLE) + '-' + str(positive_sampling_number)
            + '-' + str(negative_sampling_number) + '.csv')
    pn_sample.to_csv(name, index=None, header=True)
