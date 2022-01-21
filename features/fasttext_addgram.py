# coding: utf-8

import numpy as np


def create_ngram_set(input_list, ngram_value=2):
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))


def add_ngram(sequences, token_indice, ngram_range=2):
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for ngram_value in range(2, ngram_range + 1):
            for i in range(len(new_list) - ngram_value + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)

    return new_sequences


def addgram(ngram_range, x_train, x_test, max_features=20000):
    if ngram_range > 1:
        print('Adding {}-gram features'.format(ngram_range))
        # Create set of unique n-gram from the training set.
        ngram_set = set()
        for input_list in x_train:
            for i in range(2, ngram_range + 1):
                set_of_ngram = create_ngram_set(input_list, ngram_value=i)
                ngram_set.update(set_of_ngram)

        # Dictionary mapping n-gram token to a unique integer.
        # Integer values are greater than max_features in order
        # to avoid collision with existing features.
        start_index = max_features + 1
        token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
        indice_token = {token_indice[k]: k for k in token_indice}

        # max_features is the highest integer that could be found in the dataset.
        max_features = np.max(list(indice_token.keys())) + 1

        # Augmenting x_train and x_test with n-grams features
        x_train = add_ngram(x_train, token_indice, ngram_range)
        x_test = add_ngram(x_test, token_indice, ngram_range)
        return x_train, x_test


def seq_padding(*X, padding=0):
    L = [len(x) for x in X[0]] + [len(x) for x in X[1]]
    ML = max(L)
    return (ML, np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X[0]
    ]), np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X[1]
    ]))
def seq_padding_ML(*X, ML = 500, padding=0):

    return (ML, np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x[:ML] for x in X[0]
    ]), np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x[:ML] for x in X[1]
    ]))