import numpy as np
from gensim.models import Word2Vec

np.random.seed(1337)


def filter_with_vocabulary(sentences, labels, vocabulary, min_sentence_length=15):
    updated_sentences = []
    updated_labels = []
    for label_index, sentence in enumerate(sentences):
        current_train_filter = [word for word in sentence if word in vocabulary]
        if len(current_train_filter) >= min_sentence_length:
            updated_sentences.append(current_train_filter)
            updated_labels.append(labels[label_index])

    return updated_sentences, updated_labels


def filter_with_labels(sentences, labels, known_labels):
    known_labels_unique = set(known_labels)
    labels_unique = set(labels)
    unwanted_labels = list(labels_unique - known_labels_unique)
    updated_sentences = []
    updated_labels = []
    for j in range(len(labels)):
        if labels[j] not in unwanted_labels:
            updated_sentences.append(sentences[j])
            updated_labels.append(labels[j])

    return updated_sentences, updated_labels


def load_data(dataset_name, min_train_samples_per_class):
    wordvec_model = Word2Vec.load("./data/{0}/word2vec.model".format(dataset_name))
    all_data = np.load(
        "./data/{0}/all_data_{1}.npy".format(dataset_name, min_train_samples_per_class),
        allow_pickle=True,
    )
    all_owner = np.load(
        "./data/{0}/all_owner_{1}.npy".format(
            dataset_name, min_train_samples_per_class
        ),
        allow_pickle=True,
    )

    return wordvec_model, all_data, all_owner


def embedding(
        sentences,
        labels,
        unique_labels,
        wordvec_model,
        vocabulary,
        max_sentence_len=50,
        embed_size_word2vec=200,
):
    X = np.empty(
        shape=[len(sentences), max_sentence_len, embed_size_word2vec], dtype="float32"
    )
    Y = np.empty(shape=[len(labels), 1], dtype="int32")
    for j, curr_row in enumerate(sentences):
        sequence_cnt = 0
        for item in curr_row:
            if item in vocabulary:
                X[j, sequence_cnt, :] = wordvec_model[item]
                sequence_cnt = sequence_cnt + 1
                if sequence_cnt == max_sentence_len - 1:
                    break
        for k in range(sequence_cnt, max_sentence_len):
            X[j, k, :] = np.zeros((1, embed_size_word2vec))
        Y[j, 0] = unique_labels.index(labels[j])

    return X, Y
