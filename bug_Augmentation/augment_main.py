"""
the implementation of prototype clustering-based augmentation
key parameters：
FLAG_POSITIVE_SAMPLE：whether the positive sampling is conducted
FLAG_NEGATIVE_SAMPLE：whether the negative sampling is conducted
"""
import bug_vector
import cluster
import sample
import numpy as np
import pandas as pd
import os


def augment_main(dataset_name, min_train_samples_per_class, num_cv,
                 FLAG_POSITIVE_SAMPLE, FLAG_NEGATIVE_SAMPLE, positive_sampling_number, negative_sampling_number):
    # bug vector loading
    bug, owner = bug_vector.bug_vector(dataset_name, min_train_samples_per_class, num_cv)
    print("bug shape:", bug.shape)
    print("owner shape:", owner.shape)
    update_bug = bug.reshape(bug.shape[0], -1)
    y = 0
    x = 10000
    update_bug = update_bug[y:x]
    update_owner = owner[y:x]
    print(update_bug.shape, update_owner.shape)
    np.savetxt('./bug_vector_results/bug_vectors.csv', update_bug, delimiter=',')
    list1 = [[i] for i in range(1, x + 1)]
    names = ['bug_id']
    test = pd.DataFrame(columns=names, data=list1)
    test.to_csv('test.csv', index=False, header=False)
    f1 = pd.read_csv('test.csv', header=None)
    f2 = pd.read_csv('./bug_vector_results/bug_vectors.csv', header=None)
    file = [f1, f2]
    train = pd.concat(file, axis=1)
    train.to_csv("./bug_vector_results/bug_vectors.csv", index=False, header=False)
    np.savetxt('./bug_vector_results/owner.csv', update_owner, delimiter=',', header='owner_id')
    os.remove('test.csv')

    # bug embedding clustering
    cluster.run_kmeans()
    f1 = pd.read_csv('./cluster_results/mapped_index.csv')
    f2 = pd.read_csv('./bug_vector_results/owner.csv')
    file = [f1, f2]
    train = pd.concat(file, axis=1)
    train.to_csv("./cluster_results/mapped_index.csv", index=False)

    # bug resampling augmentation
    data = sample.read_csv()
    sample.sampling(FLAG_POSITIVE_SAMPLE, FLAG_NEGATIVE_SAMPLE,
                    positive_sampling_number, negative_sampling_number, data, dataset_name)


if __name__ == '__main__':
    augment_main(dataset_name="google_chromium", min_train_samples_per_class=20, num_cv=10,
                 FLAG_POSITIVE_SAMPLE=1, FLAG_NEGATIVE_SAMPLE=1, positive_sampling_number=5, negative_sampling_number=5)
