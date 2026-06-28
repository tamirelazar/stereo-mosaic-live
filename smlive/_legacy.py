import numpy as np
import os
import matplotlib.pyplot as plt
import time


from scipy.ndimage import generate_binary_structure
from scipy.ndimage import maximum_filter, convolve
from scipy.ndimage import label, center_of_mass, map_coordinates
import shutil
from imageio import imwrite

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


class PanoramicVideoGenerator:
    """
  Generates panorama from a set of images.
  """

    def __init__(self, data_dir, file_prefix, num_images):
        """
    The naming convention for a sequence of images is file_prefixN.jpg,
    where N is a running number 001, 002, 003...
    :param data_dir: path to input images.
    :param file_prefix: see above.
    :param num_images: number of images to produce the panoramas with.
    """
        self.file_prefix = file_prefix
        self.files = [os.path.join(data_dir, '%s%03d.jpg' % (file_prefix, i + 1)) for i in range(num_images)]
        self.files = list(filter(os.path.exists, self.files))
        self.panoramas = None
        self.homographies = None
        print('found %d images' % len(self.files))

    def align_images(self, translation_only=False):
        """
    compute homographies between all images to a common coordinate system
    :param translation_only: see estimte_rigid_transform
    """
        # Extract feature point locations and descriptors.
        points_and_descriptors = []
        for file in self.files:
            image = sol4_utils.read_image(file, 1)
            self.h, self.w = image.shape
            pyramid, _ = sol4_utils.build_gaussian_pyramid(image, 3, 7)
            points_and_descriptors.append(find_features(pyramid))

        # Compute homographies between successive pairs of images.
        Hs = []
        for i in range(len(points_and_descriptors) - 1):
            points1, points2 = points_and_descriptors[i][0], points_and_descriptors[i + 1][0]
            desc1, desc2 = points_and_descriptors[i][1], points_and_descriptors[i + 1][1]

            # Find matching feature points.
            ind1, ind2 = match_features(desc1, desc2, .7)
            points1, points2 = points1[ind1, :], points2[ind2, :]

            # Compute homography using RANSAC.
            H12, inliers = ransac_homography(points1, points2, 100, 6, translation_only)

            # Uncomment for debugging: display inliers and outliers among matching points.
            # In the submitted code this function should be commented out! TODO
            # display_matches(self.images[i], self.images[i+1], points1 , points2, inliers)

            Hs.append(H12)

        # Compute composite homographies from the central coordinate system.
        accumulated_homographies = accumulate_homographies(Hs, (len(Hs) - 1) // 2)
        self.homographies = np.stack(accumulated_homographies)
        self.frames_for_panoramas = filter_homographies_with_translation(self.homographies, minimum_right_translation=5)
        self.homographies = self.homographies[self.frames_for_panoramas]

    def generate_panoramic_images(self, number_of_panoramas):
        """
    combine slices from input images to panoramas.
    :param number_of_panoramas: how many different slices to take from each input image
    """
        assert self.homographies is not None

        # compute bounding boxes of all warped input images in the coordinate system of the middle image (as given by the homographies)
        self.bounding_boxes = np.zeros((self.frames_for_panoramas.size, 2, 2))
        for i in range(self.frames_for_panoramas.size):
            self.bounding_boxes[i] = compute_bounding_box(self.homographies[i], self.w, self.h)

        # change our reference coordinate system to the panoramas
        # all panoramas share the same coordinate system
        global_offset = np.min(self.bounding_boxes, axis=(0, 1))
        self.bounding_boxes -= global_offset

        slice_centers = np.linspace(0, self.w, number_of_panoramas + 2, endpoint=True, dtype=int)[1:-1]
        warped_slice_centers = np.zeros((number_of_panoramas, self.frames_for_panoramas.size))
        # every slice is a different panorama, it indicates the slices of the input images from which the panorama
        # will be concatenated
        for i in range(slice_centers.size):
            slice_center_2d = np.array([slice_centers[i], self.h // 2])[None, :]
            # homography warps the slice center to the coordinate system of the middle image
            warped_centers = [apply_homography(slice_center_2d, h) for h in self.homographies]
            # we are actually only interested in the x coordinate of each slice center in the panoramas' coordinate system
            warped_slice_centers[i] = np.array(warped_centers)[:, :, 0].squeeze() - global_offset[0]

        panorama_size = np.max(self.bounding_boxes, axis=(0, 1)).astype(int) + 1

        # boundary between input images in the panorama
        x_strip_boundary = ((warped_slice_centers[:, :-1] + warped_slice_centers[:, 1:]) / 2)
        x_strip_boundary = np.hstack([np.zeros((number_of_panoramas, 1)),
                                      x_strip_boundary,
                                      np.ones((number_of_panoramas, 1)) * panorama_size[0]])
        x_strip_boundary = x_strip_boundary.round().astype(int)

        self.panoramas = np.zeros((number_of_panoramas, panorama_size[1], panorama_size[0], 3), dtype=np.float64)
        for i, frame_index in enumerate(self.frames_for_panoramas):
            # warp every input image once, and populate all panoramas
            image = sol4_utils.read_image(self.files[frame_index], 2)
            warped_image = warp_image(image, self.homographies[i])
            x_offset, y_offset = self.bounding_boxes[i][0].astype(int)
            y_bottom = y_offset + warped_image.shape[0]

            for panorama_index in range(number_of_panoramas):
                # take strip of warped image and paste to current panorama
                boundaries = x_strip_boundary[panorama_index, i:i + 2]
                image_strip = warped_image[:, boundaries[0] - x_offset: boundaries[1] - x_offset]
                x_end = boundaries[0] + image_strip.shape[1]
                self.panoramas[panorama_index, y_offset:y_bottom, boundaries[0]:x_end] = image_strip

        # crop out areas not recorded from enough angles
        # assert will fail if there is overlap in field of view between the left most image and the right most image
        crop_left = int(self.bounding_boxes[0][1, 0])
        crop_right = int(self.bounding_boxes[-1][0, 0])
        assert crop_left < crop_right, 'for testing your code with a few images do not crop.'
        print(crop_left, crop_right)
        self.panoramas = self.panoramas[:, :, crop_left:crop_right, :]

    def save_panoramas_to_video(self):
        assert self.panoramas is not None
        out_folder = 'tmp_folder_for_panoramic_frames/%s' % self.file_prefix
        try:
            shutil.rmtree(out_folder)
        except:
            print('could not remove folder')
            pass
        os.makedirs(out_folder)
        # save individual panorama images to 'tmp_folder_for_panoramic_frames'
        for i, panorama in enumerate(self.panoramas):
            imwrite('%s/panorama%02d.png' % (out_folder, i + 1), (panorama.clip(0, 1) * 255).astype('uint8'))
        if os.path.exists('%s.mp4' % self.file_prefix):
            os.remove('%s.mp4' % self.file_prefix)
        # write output video to current folder
        os.system('ffmpeg -framerate 3 -i %s/panorama%%02d.png %s.mp4' %
                  (out_folder, self.file_prefix))


    def show_panorama(self, panorama_index, figsize=(20, 20)):
        assert self.panoramas is not None
        plt.figure(figsize=figsize)
        plt.imshow(self.panoramas[panorama_index].clip(0, 1))
        plt.show()
