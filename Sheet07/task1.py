import numpy as np
import utils



# ========================== Mean =============================
def calculate_mean_shape(kpts):
    """
        calculates mean shape and fix it to unit variance
        kpts: num_samples x num_points x 2
    """

    mean_shape = np.mean(kpts, axis=0)
    mean_point = np.mean(mean_shape, axis=0)

    # calculate the variance
    var = np.sum(mean_point ** 2)

    # scale mean to unit variance
    scale = np.sqrt(1 / var)
    mean_shape *= scale

    return mean_shape



# ====================== Main Step ===========================
def procrustres_analysis_step(kpts, reference_mean):
    # get mean point
    reference_mean_pt = np.mean(reference_mean, axis=0)
    mean_pts = np.mean(kpts, axis=1)

    #  calculate variance of reference_shape
    centered_ref_mean = reference_mean - reference_mean_pt[None, :]
    centered_kpts = kpts - mean_pts[:, None]
    var_ref = np.sum((centered_ref_mean) ** 2)
    var_kpts = np.sum(np.sum(centered_kpts ** 2, axis=2), axis=1)

    # calculate the corresponding scale based on keypoint variance
    scales = np.sqrt(var_ref / var_kpts)
    centered_kpts = centered_kpts * scales[:, None, None]

    # calculate covariance matrices
    num_samples, num_points, dim = centered_kpts.shape

    temp = (centered_kpts.transpose([0, 2, 1])).reshape([-1, num_points])
    covariances = np.dot(temp, centered_ref_mean)
    covariances = covariances.reshape([num_samples, dim, dim])

    # get rotation matrices
    u, s, vT = np.linalg.svd(covariances)

    v = vT.transpose([0, 2, 1])
    uT = u.transpose([0, 2, 1])

    rotations = np.sum((uT[:, :, None, :] * v[..., None]), axis=1)

    # rotate keypoints
    rotated_kpts = centered_kpts.transpose([0, 2, 1])
    rotated_kpts = np.asarray([np.dot(rotations[i], rotated_kpts[i]).transpose() for i in range(rotations.shape[0])])

    # add mean pt
    rotated_kpts = rotated_kpts + reference_mean_pt[None, None, :]

    return rotated_kpts



# =========================== Error ====================================

def compute_avg_error(kpts, mean_shape):

    squared_diff = (kpts - mean_shape[None, :, :]) ** 2

    rmse = np.sqrt(np.mean(squared_diff))

    return rmse




# ============================ Procrustres ===============================

def procrustres_analysis(kpts, max_iter=int(1e3), min_error=1e-5):

    aligned_kpts = kpts.copy()

    for iter in range(max_iter):

        reference_mean = calculate_mean_shape(aligned_kpts)

        # align shapes to mean shape
        aligned_kpts = procrustres_analysis_step(aligned_kpts, reference_mean)

        # calculate new reference mean
        reference_mean = calculate_mean_shape(aligned_kpts)
        # calculate alignment error
        rmse = compute_avg_error(aligned_kpts, reference_mean)

        print("(%d) RMSE: %f" % (iter + 1, rmse))

        if rmse <= min_error:
            break

    # visualize
    utils.visualize_hands(aligned_kpts, "aligned keypoints", 0.1)

    # visualize mean shape
    utils.visualize_hands(reference_mean[np.newaxis, :, :], "reference mean")

    return aligned_kpts
