"""
Learning on Sets and Graph Generative Models - ALTEGRAD - Nov 2023
"""

import numpy as np


def create_train_dataset():
    n_train = 100000
    max_train_card = 10

    ############## Task 1
    
    X_train = np.zeros((n_train, max_train_card), dtype=int)
    y_train = np.zeros(n_train, dtype=int)

    for i in range(n_train):
        card = np.random.randint(1, max_train_card + 1)
        X_train[i, -card:] = np.random.randint(1, 11, size=card)
        y_train[i] = np.sum(X_train[i, :])

    print(f'The first training sample is {X_train[0, :]}')
    print(f'The first label is {y_train[0]}')

    return X_train, y_train


def create_test_dataset():
    
    ############## Task 2
    
    n_test = 200000
    min_test_card = 5
    max_test_card = 100
    step_test_card = 5
    cards = range(min_test_card, max_test_card + 1, step_test_card)
    n_samples_per_card = n_test // len(cards)

    X_test = list()
    y_test = list()

    for card in cards:
        X = np.random.randint(1, 11, size=(n_samples_per_card, card))
        y = np.sum(X, axis=1)

        X_test.append(X)
        y_test.append(y)

    return X_test, y_test
