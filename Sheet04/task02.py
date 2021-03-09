import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)


def load_data():
    """ loads the data for this task
    :return:
    """
    fpath = 'images/ball.png'
    fx = 0.5
    radius = 140 * fx
    Im = cv2.imread(fpath, 0).astype('float32')/255  # 0 .. 1

    # we resize the image to speed-up the level set method
    Im = cv2.resize(Im, dsize=(0, 0), fx=fx, fy=fx)

    height, width = Im.shape

    centre = (width // 2, height // 2)
    Y, X = np.ogrid[:height, :width]
    phi = radius - np.sqrt((X - centre[0]) ** 2 + (Y - centre[1]) ** 2)

    return Im, phi


def get_contour(phi, eps=1):
    """ get all points on the contour
    :param phi:
    :param eps:
    :return: [(x, y), (x, y), ....]  points on contour
    """
    phi = np.round(phi)
    A = (phi > -eps) * 1
    B = (phi < eps) * 1
    D = (A - B).astype(np.int32)
    D = (D == 0) * 1
    Y, X = np.nonzero(D)
    return np.array([X, Y]).transpose()

# ===========================================
# FUNCTIONS
# ===========================================


def get_w(Im):
    """ yields w and other static objects
    :param Im: grayscale image
    :return:
    """
    dx = np.array([[-1 / 2, 0, 1 / 2]], 'float64')
    dxx = np.array([[1, -2, 1]], 'float64')

    gx = cv2.filter2D(Im, -1, dx)
    gy = cv2.filter2D(Im, -1, dx.T)

    M = np.sqrt(gx ** 2 + gy ** 2)  # magnitude
    w = 1 / (M + 1)
    min_w = np.min(w)
    max_w = np.max(w)
    w = (w - min_w) * (max_w / (max_w - min_w))  # 0 .. 1

    # estimate time step
    tau = 1 / (4 * max_w)

    return dx, dxx, gx, gy, w, tau


def get_derivatives(phi, dx, dxx):
    """ calculate all derivates for phi
    :param phi:
    :param dx:
    :param dxx:
    :return:
    """
    phi_x = cv2.filter2D(phi, -1, dx)
    phi_y = cv2.filter2D(phi, -1, dx.T)
    phi_xy = cv2.filter2D(phi_x, -1, dx.T)
    phi_xx = cv2.filter2D(phi, -1, dxx)
    phi_yy = cv2.filter2D(phi, -1, dxx.T)
    return phi_x, phi_y, phi_xy, phi_xx, phi_yy


def uphill_direction(w_x, w_y, phi):
    """ calculates the uphill direction \partial_t \phi
    :param w_x:
    :param w_y:
    :param phi:
    :return: \partial_t \phi
    """
    term1 = np.maximum(w_x, 0) * (np.roll(phi, -1, axis=1) - phi)
    term2 = np.minimum(w_x, 0) * (phi - np.roll(phi, 1, axis=1))
    term3 = np.maximum(w_y, 0) * (np.roll(phi, -1, axis=0) - phi)
    term4 = np.minimum(w_y, 0) * (phi - np.roll(phi, 1, axis=0))
    return term1 + term2 + term3 + term4


def curvature(phi, dx, dxx):
    """ calculates the curvature
    :param phi:
    :param dx:
    :param dxx:
    :return:
    """
    eps = 0.0001
    phi_x, phi_y, phi_xy, phi_xx, phi_yy = get_derivatives(
        phi, dx, dxx)
    phi_x_sq = phi_x**2
    phi_y_sq = phi_y**2

    term1 = phi_xx * phi_y_sq
    term2 = -2 * phi_x * phi_y * phi_xy
    term3 = phi_yy * phi_x_sq
    term4 = phi_x_sq + phi_y_sq + eps

    return (term1 + term2 + term3)/term4


def get_contour_img(phi):
    """ gets an image with all pixels of the contour = 1
        and all other pixels = 0
    :param phi:
    :return:
    """
    c = get_contour(phi)
    img = np.zeros(phi.shape, 'int32')
    img[c[:, 1], c[:, 0]] = 1
    return img


def step(phi, w, gx, gy, dx, dxx, tau):
    """
    :param phi:
    :param w:
    :param gx:
    :param gy:
    :param dx:
    :param dxx:
    :param tau:
    :return:
    """
    # so that we can see how many pixels have changed
    previous_contour = get_contour_img(phi)

    curv = (curvature(phi, dx, dxx) * w) * tau
    uphill = uphill_direction(gx, gy, phi.astype('float32'))
    #uphill = uphill_two(gx, gy, phi.astype('float32'))

    phi += curv + uphill

    contour = get_contour_img(phi)  # count changed pixels
    diff = np.abs(previous_contour - contour)
    n_pixels_changed = np.sum(diff)
    return phi, n_pixels_changed


# ===========================================
# RUNNING
# ===========================================


if __name__ == '__main__':

    n_steps = 20000
    plot_every_n_step = 100

    Im, phi = load_data()

    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    # ------------------------
    # your implementation here
    dx, dxx, gx, gy, w, tau = get_w(Im)
    consecutive_0 = 0
    # ------------------------

    for t in range(n_steps):

        # ------------------------
        # your implementation here
        phi, n_pixels_changed = step(phi, w, gx, gy, dx,dxx, tau)
        # ------------------------

        if t % plot_every_n_step == 0:
            ax1.clear()
            ax1.imshow(Im, cmap='gray')
            ax1.set_title('frame ' + str(t) +\
                          ', changed pixel=' + str(n_pixels_changed))

            contour = get_contour(phi, eps=1)
            if len(contour) > 0:
                ax1.scatter(contour[:, 0], contour[:, 1], color='red', s=1)

            ax2.clear()
            ax2.imshow(phi)
            ax2.set_title(r'$\phi$', fontsize=22)
            plt.pause(0.01)

        if n_pixels_changed == 0:
            consecutive_0 += 1
        else:
            consecutive_0 = 0
        if consecutive_0 > 20:
            print('converged after frame ', t)

            ax1.clear()
            ax1.imshow(Im, cmap='gray')
            ax1.set_title('converged at frame ' + str(t))
            contour = get_contour(phi, eps=1)
            if len(contour) > 0:
                ax1.scatter(contour[:, 0], contour[:, 1], color='red', s=20)

            ax2.clear()
            ax2.imshow(phi)
            ax2.set_title(r'$\phi$', fontsize=22)
            plt.pause(0.01)
            break

    plt.show()
