import cv2
import numpy as np
import matplotlib.pylab as plt


def main():
    # Load the images
    file_path = "data/exercise3/mountain1.png"
    file_path2 = "data/exercise3/mountain2.png"
    img1 = cv2.imread(file_path)
    img_gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.imread(file_path2)
    img_gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # extract sift keypoints and descriptors
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img_gray1, None)
    kp2, des2 = sift.detectAndCompute(img_gray2, None)
    img1 = cv2.drawKeypoints(img_gray1, kp1, img1)
    img2 = cv2.drawKeypoints(img_gray2, kp2, img2)
    h_stack = np.hstack([img1, img2])
    display_image("img1 keypoints + img2 keypoints", h_stack)
    # your own implementation of matching
    # To generate candidate matches Euclidean distance
    height = len(des1)
    width = len(des2)
    distance_matrix = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            dist = np.sum((des1[i] - des2[j]) ** 2) ** 0.5
            distance_matrix[i, j] = dist
    th = 0.4
    # Store the best match for the first image
    best_match_i_idx = []
    # Store the best match for the second image
    best_match_j_idx = []
    # Store the best match
    best_match = []
    # sort
    match_sort_i_idx = np.argsort(distance_matrix, axis=1)
    match_sort_j_idx = np.argsort(distance_matrix, axis=0)
    best2match_i_idx = match_sort_i_idx[:, 0:2]
    best2match_j_idx = match_sort_j_idx[0:2, :]
    for i in range(len(best2match_i_idx)):
        best_ma_idx = best2match_i_idx[i, 0]
        second_ma_idx = best2match_i_idx[i, 1]
        if distance_matrix[i, best_ma_idx] / distance_matrix[i, second_ma_idx] <= th:
            best_match_i_idx.append([i, best_ma_idx])
    for j in range(len(best2match_j_idx[0])):
        best_ma_idx = best2match_j_idx[0, j]
        second_ma_idx = best2match_j_idx[1, j]
        if distance_matrix[best_ma_idx, j] / distance_matrix[second_ma_idx, j] <= th:
            best_match_j_idx.append([best_ma_idx, j])
    # print("best_match_i_idx:",best_match_i_idx)
    # print("best_match_j_idx:", best_match_j_idx)
    for k in range(len(best_match_i_idx)):
        if best_match_i_idx[k] in best_match_j_idx:
            best_match.append(best_match_i_idx[k])
    # display the matches
    # k_best
    k = 10
    good = []
    for m, n in best_match:
        good.append([cv2.DMatch(m, n, distance_matrix[m, n])])
    img_match = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good[0:k], None)
    display_image("img_match", img_match)


def display_image(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
