from numpy import *
import operator
import os


def knn_classify(vec_input, vecs_train, labels_train, k=5):
    num_samples = vecs_train.shape[0]

    # step 1: calculate Euclidean distance
    # tile(A, reps): Construct an array by repeating A reps times
    # the following copy num_samples rows for vecs_train
    diff = tile(vec_input, (num_samples, 1)) - vecs_train  # Subtract element-wise
    squared_diff = diff ** 2  # squared for the subtract
    squared_dist = sum(squared_diff, axis=1)  # sum is performed by row
    distance = squared_dist ** 0.5

    # step 2: sort the distance
    # argsort() returns the indices that would sort an array in a ascending order
    sorted_dist_indices = argsort(distance)

    class_count = {}  # define a dictionary (can be append element)
    for i in range(k):
        # step 3: choose the min k distance
        vec_label = labels_train[sorted_dist_indices[i]]

        # step 4: count the times labels occur
        # when the key voteLabel is not in dictionary classCount, get()
        # will return 0
        class_count[vec_label] = class_count.get(vec_label, 0) + 1

    # step 5: the max voted class will return
    max_count = 0
    for key, value in class_count.items():
        if value > max_count:
            max_count = value
            max_index = key

    return max_index


def knn_classify_list(vecs_input, vecs_train, labels_train, k=5):
    test_pred = []
    for i, vec_input in enumerate(vecs_input):
        pred = knn_classify(vec_input, vecs_train, labels_train, k)
        test_pred.append(pred)
        print(i)
    return np.array(test_pred)