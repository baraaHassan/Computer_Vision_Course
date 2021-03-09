import numpy as np
import time

import utils
import task1
import task2

hands_orig_train = 'data/hands_orig_train.txt.new'
hands_aligned_test = 'data/hands_aligned_test.txt.new'
hands_aligned_train = 'data/hands_aligned_train.txt.new'

def get_keypoints(path):
    data_info = utils.load_data(path)

    data_dim = data_info['data_dim']
    num_points = data_info['num_data']
    samples = data_info['samples']
    kpts = utils.convert_samples_to_xy(samples)

    return kpts

def task_1():
    # Loading Trainig Data
    kpts = get_keypoints(hands_orig_train)

    # calculate mean
    reference_mean = task1.calculate_mean_shape(kpts)

    # we want to visualize the data first
    shapes_ax = utils.visualize_hands(kpts, "Training Shapes", 0.1)
    mean_ax = utils.visualize_hands(reference_mean[np.newaxis, :, :],
                                    "Mean Shape", 0.1)

    task1.procrustres_analysis(kpts)

    time.sleep(20)


def task_2_1():
    # ============= Load Data =================
    kpts = get_keypoints(hands_aligned_train)
    kpts = kpts.reshape([kpts.shape[0], -1])

    mean, pcs, pc_weights = task2.train_statistical_shape_model(kpts)

    return mean, pcs, pc_weights

def task_2_2(mean, pcs, pc_weights):
    # ============= Load Data =================
    kpts = get_keypoints(hands_aligned_test)
    # shapes_ax = utils.visualize_hands(kpts, 0.1)
    print(kpts.shape)
    kpts = kpts.reshape([kpts.shape[0], -1])

    task2.reconstruct_test_shape(kpts, mean, pcs, pc_weights)

    time.sleep(20)

if __name__ == '__main__':
    print("Running Task 1")
    task_1()

    print("Running Task 2")
    mean, pcs, pc_weights = task_2_1()

    task_2_2(mean, pcs, pc_weights)
