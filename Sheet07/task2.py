import numpy as np
import utils

# ======================= PCA =======================
def pca(covariance, preservation_ratio=0.9):

    U, s, VT = np.linalg.svd(covariance, full_matrices=True)
    eigenvectors = VT
    eigenvalues = s

    idxs = np.argsort(s)[::-1]
    eigenvectors = VT[idxs, :]
    eigenvalues = s[idxs]

    eigval_sum = np.sum(eigenvalues)
    cum = 0
    for i in range(len(eigenvalues)):
        cum += eigenvalues[i]
        if cum / eigval_sum >= 0.9:
            break

    print("We select %d PCs" % (i + 1))

    selected_eigenvalues = eigenvalues[:(i+1)]
    selected_pcs = eigenvectors[:(i+1)]

    return selected_pcs, selected_eigenvalues




# ======================= Covariance =======================

def create_covariance_matrix(kpts, mean_shape):
    aligned_kpts = kpts - mean_shape[np.newaxis, :]
    covariance = np.dot(aligned_kpts.transpose(), aligned_kpts)

    return covariance





# ======================= Visualization =======================

def visualize_impact_of_pcs(mean, pcs, pc_weights):
    for pc_idx in range(pc_weights.shape[0]):
        ax = utils.visualize_hands(mean[np.newaxis, :].reshape([1, -1, 2]),
                                   title="Variation of PC %d" % (pc_idx + 1),
                                   clear=True, delay=0.1)
        for itr in range(5):
            weights = np.linspace(-0.3, 0.3, 7)

            for weight in weights:
                tmp_shape = mean.copy()
                w = weight * np.sqrt(pc_weights[pc_idx])

                tmp_shape += pcs[pc_idx, :] * w
                ax = utils.visualize_hands(tmp_shape[np.newaxis, :].reshape([1, -1, 2]),
                                            title="Variation of PC %d" % (pc_idx + 1),
                                            clear=True, delay=0.1, ax=ax)





# ======================= Training =======================
def train_statistical_shape_model(kpts):

    # Mean Shape
    mean_shape = np.mean(kpts, axis=0)
    shapes_ax = utils.visualize_hands(mean_shape[np.newaxis, :].reshape([1, -1, 2]),
                                      "Initial Mean Shape", 0.1)

    # COVAR
    covariance = create_covariance_matrix(kpts, mean_shape)

    # PCA
    pcs, pc_weights = pca(covariance)

    # VIS
    visualize_impact_of_pcs(mean_shape, pcs, pc_weights)

    return mean_shape, pcs, pc_weights




# ======================= Reconstruct =======================
def reconstruct_test_shape(kpts, mean, pcs, pc_weight):

    # Align test shape
    aligned_kpts = kpts - mean[None, :]

    # Estimate weights
    H = pcs @ aligned_kpts[0]
    print("Estimated weights h_i: ", H)

    # reconstruction
    weighted_pcs = pcs * H[:, None]
    reconstruction = mean + np.sum(weighted_pcs, axis=0)

    # calculate reconstruction error
    error = np.sqrt(np.mean((kpts - reconstruction) ** 2))
    print("RMS:", error)

    # Visualize
    utils.visualize_hands(kpts[np.newaxis, :].reshape([1, -1, 2]),
                                title="Original Shape")

    utils.visualize_hands(reconstruction[np.newaxis, :].reshape([1, -1, 2]),
                                title="Reconstructed Shape")
