import numpy as np

from BugInitialize import load_data, filter_with_vocabulary, filter_with_labels, embedding


def bug_vector(dataset_name, min_train_samples_per_class, num_cv):
    wordvec_model, sentences, labels = load_data(dataset_name, min_train_samples_per_class)

    vocabulary = wordvec_model.wv.vocab
    splitLength = len(sentences) // (num_cv + 1)

    for i in range(1, num_cv + 1):
        train_data = sentences[: i * splitLength - 1]
        test_data = sentences[i * splitLength: (i + 1) * splitLength - 1]
        train_owner = labels[: i * splitLength - 1]
        test_owner = labels[i * splitLength: (i + 1) * splitLength - 1]

        updated_train_data, updated_train_owner = filter_with_vocabulary(
            train_data, train_owner, vocabulary
        )
        final_test_data, final_test_owner = filter_with_vocabulary(
            test_data, test_owner, vocabulary
        )

        updated_test_data, updated_test_owner = filter_with_labels(
            final_test_data, final_test_owner, updated_train_owner
        )

        unique_train_label = list(set(updated_train_owner))

        X, Y = embedding(
            updated_test_data,
            updated_test_owner,
            unique_train_label,
            wordvec_model,
            vocabulary,
        )
        if (i == 1):
            bug = X
            owner = Y
        else:
            bug = np.concatenate((bug, X), axis=0)
            owner = np.concatenate((owner, Y), axis=0)
            if (bug.shape[0] >= 30000 or owner.shape[0] >= 30000):
                break
    return bug, owner
