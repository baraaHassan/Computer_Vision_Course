import cv2
import numpy as np
import matplotlib.pylab as plt


def main():
    # Load the image
    img = cv2.imread("data/exercise2/building.jpeg")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 1st and 2nd question
    # Compute Structural Tensor
    D = 2
    size_w = 2 * D + 1
    dx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, 3)
    dy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, 3)
    Mxx = dx * dx
    Mxy = dx * dy
    Myy = dy * dy
    w = np.ones((size_w, size_w), dtype=np.float) / size_w ** 2
    Mxx_w = cv2.filter2D(Mxx, cv2.CV_64F, w)
    Mxy_w = cv2.filter2D(Mxy, cv2.CV_64F, w)
    Myy_w = cv2.filter2D(Myy, cv2.CV_64F, w)
    f = np.zeros_like(img_gray, dtype=np.float)
    height, width = np.shape(img_gray)
    k = 0.05
    for i in range(height):
        for j in range(width):
            Sij = np.array([[Mxx_w[i, j], Mxy_w[i, j]], [Mxy_w[i, j], Myy_w[i, j]]])
            e_value, e_vector = np.linalg.eig(Sij)
            lam1 = e_value[0]
            lam2 = e_value[1]
            f[i, j] = lam1 * lam2 - k * (lam1 + lam2) ** 2
    # normalize
    f = (f - np.min(f)) / (np.max(f) - np.min(f) + 0.0001)
    # threshold
    th = 0.4
    f_th = cv2.inRange(f, th, 1.0)
    # Harris Corner Detection
    # find the points whose surrounding window gave large corner response
    # nms
    f_ = f.copy()
    f_[f < th] = 0.0
    f_corner = nms(f_, D)
    h_stack_q2 = np.hstack([img_gray / 255.0, f, f_th, f_corner])
    display_image("Image gray + Response function + Threshold response function + detected corner", h_stack_q2)


    # The 3rd question
    w_th = 0.25
    q_th = 0.5
    # Forstner Corner Detection
    w_matrx = np.zeros_like(img_gray, dtype=np.float)
    q_matrx = np.zeros_like(img_gray, dtype=np.float)
    for i in range(height):
        for j in range(width):
            # numpy.linalg.LinAlgError: Singular matrix
            # Sij = np.array([[Mxx_w[i, j], Mxy_w[i, j]], [Mxy_w[i, j], Myy_w[i, j]]])
            # a = np.linalg.inv(Sij)
            # e_value, e_vector = np.linalg.eig(a)
            # lam1 = e_value[0]
            # lam2 = e_value[1]
            # w_matrx[i, j] = 1 / (lam1 + lam2)
            # q_matrx[i, j] = 1 - ((lam1 - lam2) / (lam1 + lam2)) ** 2
            Sij = np.array([[Mxx_w[i, j], Mxy_w[i, j]], [Mxy_w[i, j], Myy_w[i, j]]])
            det = Sij[0, 0] * Sij[1, 1] - Sij[0, 1] ** 2
            tr = Sij[0, 0] + Sij[1, 1]
            w_matrx[i, j] = det / (tr + 0.0000001)
            q_matrx[i, j] = 4 * det / ((tr + 0.0000001) ** 2)

    # normal
    w_matrx = (w_matrx - np.min(w_matrx)) / (np.max(w_matrx) - np.min(w_matrx) + 0.00000001)
    w_matrx_th = cv2.inRange(w_matrx, w_th, 1.0)
    w_matrx_ = w_matrx.copy()
    w_matrx_[w_matrx_ < w_th] = 0.0
    w_matrx_corner = nms(w_matrx_, D)
    h_stack_q31 = np.hstack([img_gray / 255.0, w_matrx, w_matrx_th, w_matrx_corner])
    display_image("Image gray + w_matrx function + Threshold w_matrx + Detected corner w_matrx", h_stack_q31)

    q_matrx = (q_matrx - np.min(q_matrx)) / (np.max(q_matrx) - np.min(q_matrx) + 0.00000001)
    q_matrx_th = cv2.inRange(q_matrx, q_th, 1.0)
    q_matrx_ = q_matrx.copy()
    q_matrx_[q_matrx_ < q_th] = 0.0
    q_matrx__corner = nms(q_matrx_, D)
    h_stack_q32 = np.hstack([img_gray / 255.0, q_matrx, q_matrx_th, q_matrx__corner])
    display_image("Image gray + w_matrx function + Threshold w_matrx + Detected corner w_matrx", h_stack_q32)

    return


def nms(f, D):
    f_corner = np.zeros_like(f)
    height, width = np.shape(f)
    for i in np.arange(D, height - D - 1):
        for j in np.arange(D, width - D - 1):
            arg_max = np.argmax(f[i - D:i + D + 1, j - 1:j + D + 1])
            # center
            if arg_max == (D + 1) ** 2:
                f_corner[i, j] = 1
    return f_corner


def display_image(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
