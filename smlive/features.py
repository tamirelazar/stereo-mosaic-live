import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import (generate_binary_structure, maximum_filter, convolve,
                           label, center_of_mass, map_coordinates)
from smlive import utils as sol4_utils


def harris_corner_detector(im):
    """
    Detects harris corners.
    Make sure the returned coordinates are x major!!!
    :param im: A 2D array representing an image.
    :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
    """
    # get the x der and y der using [1,0,-1] and its transpose respectively
    dx_vec = np.array([[0, 0, 0], [1, 0, -1], [0, 0, 0]])
    dy_vec = np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]])

    der_x = convolve(im, dx_vec)
    der_y = convolve(im, dy_vec)

    # blur the images der_x^2, der_y^2, der_x*der_y using spatial blur from utils with sernel 3
    der_x_sq = sol4_utils.blur_spatial(der_x ** 2, 3)
    der_y_sq = sol4_utils.blur_spatial(der_y ** 2, 3)
    der_x_y = sol4_utils.blur_spatial(der_x * der_y, 3)

    # for each pixel, find the size of both eigenvalues using det(M) - 0.04(trace(M))^2, place in response output image
    response = np.zeros(shape=im.shape)

    XSquaredYSquaredMat= np.multiply(der_x_sq, der_y_sq)
    XYSquaredMat = np.multiply(der_x_y, der_x_y)
    detM = XSquaredYSquaredMat - XYSquaredMat
    traceM = der_x_sq + der_y_sq

    response = detM - (0.04 * np.multiply(traceM, traceM))

    # use supplied non_maximum_supp function to get binary image of corners
    response = non_maximum_suppression(response)

    # return (x,y) of all corners (which is (col,row))
    indices = np.argwhere(response.T)

    return indices


def sample_descriptor(im, pos, desc_rad):
    """
    Samples descriptors at the given corners.
    :param im: A 2D array representing an image.
    :param pos: An array with shape (N,2), where pos[i,:] are the [x,y] coordinates of the ith corner point.
    :param desc_rad: "Radius" of descriptors to compute.
    :return: A 3D array with shape (N,K,K) containing the ith descriptor at desc[i,:,:].
    """
    #  get 3rd level of gauss pyramid of image, corners positions, sec_radius

    #  sample around each corner, assigning the descript matrix to an NxKxK array when K=1+2*desc_rad
    patch_s = 1 + 2 * desc_rad
    descript_mat = np.zeros(shape=(pos.shape[0], patch_s, patch_s))

    for idx, val in enumerate(pos):
        x = val[0]
        y = val[1]
        x_coords = np.linspace(x - desc_rad, x + desc_rad, patch_s)
        y_coords = np.linspace(y - desc_rad, y + desc_rad, patch_s)
        coords = np.meshgrid(y_coords, x_coords)

        descript_m = map_coordinates(im, coords, order=1, prefilter=False)
        mean = np.mean(descript_m)
        mean_normalized_descriptor = descript_m - mean
        if np.any(mean_normalized_descriptor):
            norm = np.linalg.norm(mean_normalized_descriptor)
            normalized_descriptor = np.true_divide(mean_normalized_descriptor, norm)
            descript_mat[idx] = normalized_descriptor
        else:
            descript_mat[idx] = mean_normalized_descriptor

    return descript_mat


def find_features(pyr):
    """
  Detects and extracts feature points from a pyramid.
  :param pyr: Gaussian pyramid of a grayscale image having 3 levels.
  :return: A list containing:
              1) An array with shape (N,2) of [x,y] feature location per row found in the image.
                 These coordinates are provided at the pyramid level pyr[0].
              2) A feature descriptor array with shape (N,K,K)
  """
    #  pyr is gauss pyr of >= 3 levels

    #  finds corners and extracts descriptions
    pos = spread_out_corners(pyr[0], 7, 7, 3)
    pos_for_samples = pos.astype(np.float64) * 0.25
    descriptor_mat = sample_descriptor(pyr[2], pos_for_samples, desc_rad=3)

    #  return list of corners, a sample_descriptor output 3D array
    return [pos, descriptor_mat]


def match_features(desc1, desc2, min_score):
    """
    Return indices of matching descriptors.
    :param desc1: A feature descriptor array with shape (N1,K,K).
    :param desc2: A feature descriptor array with shape (N2,K,K).
    :param min_score: Minimal match score.
    :return: A list containing:
              1) An array with shape (M,) and dtype int of matching indices in desc1.
              2) An array with shape (M,) and dtype int of matching indices in desc2.
    """
    desc1_num = desc1.shape[0]
    desc2_num = desc2.shape[0]

    #  calculate S matrix where S[k,j] is the match-score (dot prod) of desc1[k,:,:] and desc2[j,:,:]
    matches = np.einsum("ilk,jlk->ij", desc1, desc2)

    desc1_sorted_indices = np.argpartition(matches, -2)[:, -2:] # indexes of max and 2nd max in rows
    desc2_sorted_indices = np.argpartition(matches.T, -2)[:, -2:] # indexes of max and 2nd max in cols

    desc1_matches_idx = []
    desc2_matches_idx = []

    for idx, max_score_indices in enumerate(desc1_sorted_indices):
        best_match_idx = max_score_indices[1]
        second_best_idx = max_score_indices[0]
        best_match = matches[idx, best_match_idx]
        second_best_match = matches[idx, second_best_idx]
        best_match_best_idx = desc2_sorted_indices[best_match_idx, 1]
        best_match_second_idx = desc2_sorted_indices[best_match_idx, 0]
        second_match_best_idx = desc2_sorted_indices[second_best_idx, 1]
        second_match_second_idx = desc2_sorted_indices[second_best_idx, 0]

        assigned = False
        if best_match > min_score and idx in [best_match_best_idx, best_match_second_idx] \
                and best_match_idx not in desc2_matches_idx:
            desc1_matches_idx.append(idx)
            desc2_matches_idx.append(best_match_idx)
            assigned = True
        elif second_best_match > min_score and idx in [second_match_best_idx, second_match_second_idx] \
                and second_best_idx not in desc2_matches_idx and not assigned:
            desc1_matches_idx.append(idx)
            desc2_matches_idx.append(second_best_idx)

    return [np.array(desc1_matches_idx), np.array(desc2_matches_idx)]


def apply_homography(pos1, H12):
    """
  Apply homography to inhomogenous points.
  :param pos1: An array with shape (N,2) of [x,y] point coordinates.
  :param H12: A 3x3 homography matrix.
  :return: An array with the same shape as pos1 with [x,y] point coordinates obtained from transforming pos1 using H12.
  """
    #  multiply each point (after casting to (3,) shape) by H12, then dividing by the third coordinate to get return coord
    if H12[2, 2] != 0:
        H12 = np.true_divide(H12, H12[2, 2])
    pos2 = np.zeros(shape=pos1.shape)
    pos2deluxe = np.ones(shape=(pos1.shape[0], 3))
    pos2deluxe[:, 0] = pos1[:, 0]
    pos2deluxe[:, 1] = pos1[:, 1]
    pos2deluxe = np.einsum("...j, kj", pos2deluxe, H12)
    pos2[:, :] = pos2deluxe[:, :2]

    return pos2


def ransac_homography(pos1, pos2, num_iter, inlier_tol, translation_only=False):
    """
    Computes homography between two sets of points using RANSAC.
    :param pos1: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 1.
    :param pos2: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 2.
    :param num_iter: Number of RANSAC iterations to perform.
    :param inlier_tol: inlier tolerance threshold.
    :param translation_only: see estimate rigid transform
    :return: A list containing:
              1) A 3x3 normalized homography matrix.
              2) An Array with shape (S,) where S is the number of inliers,
                  containing the indices in pos1/pos2 of the maximal set of inlier matches found.
    """

    res_inliers = []
    for i in range(num_iter):

        # choose randomly 2 pairs of points
        random_pair_idx1 = np.random.choice(pos1.shape[0])
        random_pair_idx2 = np.random.choice(pos2.shape[0])
        if random_pair_idx2 == random_pair_idx1:
           random_pair_idx2 = pos2.shape[0] - random_pair_idx2 - 1

        if translation_only:
            random_pairs1 = np.array([pos1[random_pair_idx1]])
            random_pairs2 = np.array([pos2[random_pair_idx1]])
        else:
            random_pairs1 = pos1[[random_pair_idx1, random_pair_idx2]]
            random_pairs2 = pos2[[random_pair_idx1, random_pair_idx2]]


        # random_pairs1 = pos1[[random_pair_idx1, random_pair_idx2]]
        # random_pairs2 = pos2[[random_pair_idx1, random_pair_idx2]]

        #  use estimate_rigid transformation to get homography
        homography = estimate_rigid_transform(random_pairs1, random_pairs2, translation_only)
        #  is camera is not rotating, better to use translation_only = True, with one pair

        #  use apply homography to convert points1, then calculate the euclidean dist from points2
        points1_afterH = apply_homography(pos1, homography)

        new_inliers = []

        diff_points = points1_afterH - pos2

        #  store all inliers (distance less than inlier_tol)
        for idx, point in enumerate(diff_points):
            euclidian_dist = ((point[0] ** 2) + (point[1] ** 2))
            if euclidian_dist < inlier_tol:
                new_inliers.append(idx)

        if len(new_inliers) > len(res_inliers):
            res_inliers = new_inliers

    #  estimate homography again, from the largest subset of points found as inliers
    best_homo = estimate_rigid_transform(pos1[res_inliers], pos2[res_inliers], translation_only)

    #  return homography and array of inliers indices
    return [best_homo, np.array(res_inliers)]


def display_matches(im1, im2, points1, points2, inliers):

    """
    Dispalay matching points.
    :param im1: A grayscale image.
    :param im2: A grayscale image.
    :parma pos1: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im1.
    :param pos2: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im2.
    :param inliers: An array with shape (S,) of inlier matches.
    """
    #  displays  both images, the matched points on them as red dots, inliers connected by yellow lines and outliers as blue (page 8)
    image_pair = np.hstack((im1, im2))
    plt.imshow(image_pair, cmap='gray')
    for index, point in enumerate(points1):
        x = (points1[index, 0], points2[index, 0] + im1.shape[1])
        y = (points1[index, 1], points2[index, 1])
        if index in inliers:
            plt.plot(x, y, c='y', lw=.3, ms=10, marker='.', markerfacecolor='red')
        else:
            plt.plot(x, y, c='b', lw=.1, linestyle='dashed', marker='.', markerfacecolor='blue')
    plt.show()


def accumulate_homographies(H_succesive, m):
    """
    Convert a list of succesive homographies to a
    list of homographies to a common reference frame.
    :param H_successive: A list of M-1 3x3 homography
    matrices where H_successive[i] is a homography which transforms points
    from coordinate system i to coordinate system i+1.
    :param m: Index of the coordinate system towards which we would like to
    accumulate the given homographies.
    :return: A list of M 3x3 homography matrices,
    where H2m[i] transforms points from coordinate system i to coordinate system m
    """
    #  multiply homographies to get homographies from all frames to middle one
    H2m = np.zeros(shape=(len(H_succesive ) + 1, 3, 3))
    H2m[:] = np.eye(3)
    for i in range(m):  # all smaller than m
        H2m[:(m - i)] = np.matmul(H_succesive[m - i - 1], H2m[:(m - i)])

    for i in range(m + 1, len(H_succesive)):
        H2m[i:] = np.matmul(H2m[i:], np.linalg.inv(H_succesive[i]))

    #  return a list of homograpies to i
    return list(H2m)


def compute_bounding_box(homography, w, h):
    """
  computes bounding box of warped image under homography, without actually warping the image
  :param homography: homography
  :param w: width of the image
  :param h: height of the image
  :return: 2x2 array, where the first row is [x,y] of the top left corner,
   and the second row is the [x,y] of the bottom right corner
  """
    #  get the corners of each frames' transform to middle frame
    points = np.array([[0, 0], [0, h], [w, 0], [w, h]])
    transformed = apply_homography(points, homography)

    minx = min(transformed[:, 0])
    miny = min(transformed[:, 1])
    maxx = max(transformed[:, 0])
    maxy = max(transformed[:, 1])

    #  return 2x2 mat of max and min of x,y each
    return np.array([[minx, miny], [maxx, maxy]]).astype(int)


def warp_channel(image, homography):
    """
    Warps a 2D image with a given homography.
    :param image: a 2D image.
    :param homography: homograhpy.
    :return: A 2d warped image.
    """
    #  back-warp the image to middle frame coords. full calculations in page 11
    h, w = image.shape
    min_coord, max_coord = compute_bounding_box(homography, w, h).astype(int)
    y = np.linspace(min_coord[1], max_coord[1], max_coord[1] - min_coord[1])
    x = np.linspace(min_coord[0], max_coord[0], max_coord[0] - min_coord[0])
    coords = np.meshgrid(x, y)
    np_coords = np.swapaxes(np.array(coords), 0, 2)
    warped_coords = apply_homography(np_coords.reshape((np_coords.shape[0] * np_coords.shape[1], 2)), np.linalg.inv(homography))
    warped_coords = warped_coords.reshape(np_coords.shape)
    warped_coords = np.swapaxes(warped_coords, 2, 0)

    new_image = map_coordinates(image, [warped_coords[1], warped_coords[0]], order=1, prefilter=False, mode='reflect')
    return new_image

def warp_image(image, homography):
    """
  Warps an RGB image with a given homography.
  :param image: an RGB image.
  :param homography: homograhpy.
  :return: A warped image.
  """
    return np.dstack([warp_channel(image[..., channel], homography) for channel in range(3)])


def filter_homographies_with_translation(homographies, minimum_right_translation):
    """
  Filters rigid transformations encoded as homographies by the amount of translation from left to right.
  :param homographies: homograhpies to filter.
  :param minimum_right_translation: amount of translation below which the transformation is discarded.
  :return: filtered homographies..
  """
    translation_over_thresh = [0]
    last = homographies[0][0, -1]
    for i in range(1, len(homographies)):
        if homographies[i][0, -1] - last > minimum_right_translation:
            translation_over_thresh.append(i)
            last = homographies[i][0, -1]
    return np.array(translation_over_thresh).astype(int)


def estimate_rigid_transform(points1, points2, translation_only=False):
    """
  Computes rigid transforming points1 towards points2, using least squares method.
  points1[i,:] corresponds to poins2[i,:]. In every point, the first coordinate is *x*.
  :param points1: array with shape (N,2). Holds coordinates of corresponding points from image 1.
  :param points2: array with shape (N,2). Holds coordinates of corresponding points from image 2.
  :param translation_only: whether to compute translation only. False (default) to compute rotation as well.
  :return: A 3x3 array with the computed homography.
  """
    centroid1 = points1.mean(axis=0)
    centroid2 = points2.mean(axis=0)

    if translation_only:
        rotation = np.eye(2)
        translation = centroid2 - centroid1

    else:
        centered_points1 = points1 - centroid1
        centered_points2 = points2 - centroid2

        sigma = centered_points2.T @ centered_points1
        U, _, Vt = np.linalg.svd(sigma)

        rotation = U @ Vt
        translation = -rotation @ centroid1 + centroid2

    H = np.eye(3)
    H[:2, :2] = rotation
    H[:2, 2] = translation
    return H


def non_maximum_suppression(image):
    """
  Finds local maximas of an image.
  :param image: A 2D array representing an image.
  :return: A boolean array with the same shape as the input image, where True indicates local maximum.
  """
    # Find local maximas.
    neighborhood = generate_binary_structure(2, 2)
    local_max = maximum_filter(image, footprint=neighborhood) == image
    local_max[image < (image.max() * 0.1)] = False

    # Erode areas to single points.
    lbs, num = label(local_max)
    centers = center_of_mass(local_max, lbs, np.arange(num) + 1)
    centers = np.stack(centers).round().astype(int)
    ret = np.zeros_like(image, dtype=bool)
    ret[centers[:, 0], centers[:, 1]] = True

    return ret


def spread_out_corners(im, m, n, radius):
    """
  Splits the image im to m by n rectangles and uses harris_corner_detector on each.
  :param im: A 2D array representing an image.
  :param m: Vertical number of rectangles.
  :param n: Horizontal number of rectangles.
  :param radius: Minimal distance of corner points from the boundary of the image.
  :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
  """
    corners = [np.empty((0, 2), dtype=int)]
    x_bound = np.linspace(0, im.shape[1], n + 1, dtype=int)
    y_bound = np.linspace(0, im.shape[0], m + 1, dtype=int)
    for i in range(n):
        for j in range(m):
            # Use Harris detector on every sub image.
            sub_im = im[y_bound[j]:y_bound[j + 1], x_bound[i]:x_bound[i + 1]]
            sub_corners = harris_corner_detector(sub_im)
            sub_corners += np.array([x_bound[i], y_bound[j]])[np.newaxis, :]
            corners.append(sub_corners)
    corners = np.vstack(corners)
    legit = ((corners[:, 0] > radius) & (corners[:, 0] < im.shape[1] - radius) &
             (corners[:, 1] > radius) & (corners[:, 1] < im.shape[0] - radius))
    ret = corners[legit, :]
    return ret
