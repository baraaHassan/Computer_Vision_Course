import cv2
import numpy as np
import time

#////////////////////////////////////////////////////////////////////////////////////
#//																				   //
#//									Task1    									   //
#//																				   //
#////////////////////////////////////////////////////////////////////////////////////
def get_convolution_using_fourier_transform(image, kernel):
	rows, cols = image.shape
	kernel_padded = np.zeros((rows, cols), np.float32)
	
	center_y, center_x = rows//2, cols//2
	kernel_h, kernel_w = kernel.shape
	kernel_padded[center_y - kernel_h//2: center_y + kernel_h//2 + 1, center_x - kernel_w//2: center_x + kernel_w//2 + 1] = kernel

	img_freq = np.fft.fft2(image)
	img_freq = np.fft.fftshift(img_freq)

	kernel_padded = np.fft.ifftshift(kernel_padded)
	kernel_freq = np.fft.fft2(kernel_padded)
	kernel_freq = np.fft.fftshift(kernel_freq)

	result = img_freq * kernel_freq
	result = np.fft.ifftshift(result)
	result = np.fft.ifft2(result)
	result = np.abs(result)
	result /= result.max()

	return result

def task1():
	image = cv2.imread('../data/einstein.jpeg', 0)
	kernel = cv2.getGaussianKernel(7, 1)
	kernel = np.dot(kernel, kernel.T)


	conv_result = cv2.filter2D(image, cv2.CV_32F, kernel)
	conv_result /= conv_result.max()

	fft_result = get_convolution_using_fourier_transform(image, kernel)

	print('The mean absolute difference = ', np.abs(fft_result - conv_result).mean())
	cv2.imshow('Task1::Original', image)
	cv2.imshow('Task1::FFT Result', fft_result)
	cv2.imshow('Task1::Filter2d Result', conv_result)


#////////////////////////////////////////////////////////////////////////////////////
#//																				   //
#//									Task2    									   //
#//																				   //
#////////////////////////////////////////////////////////////////////////////////////
def sum_square_difference(image, template):
	image_h, image_w = image.shape
	template_h, template_w = template.shape

	result = np.zeros((image_h - template_h + 1, image_w - template_w + 1), 
					  dtype=np.float32)

	for i in range(result.shape[0]):
		for j in range(result.shape[1]):
			result[i, j] = np.sqrt(np.sum(np.power(template - image[i: i + template_h, j: j + template_w], 2)))

	result /= np.max(result)
	result = 1 - result
	return result

def normalized_cross_correlation(image, template):
	image_h, image_w = image.shape
	template_h, template_w = template.shape

	result = np.zeros((image_h - template_h + 1, image_w - template_w + 1), 
					  dtype=np.float32)

	template_mean = np.mean(template)
	for i in range(result.shape[0]):
		for j in range(result.shape[1]):
			patch = image[i: i + template_h, j: j + template_w]
			patch_mean = np.mean(patch)

			norm = np.sqrt(np.sum(np.power(template - template_mean, 2)) * np.sum(np.power(patch - patch_mean, 2)))
			result[i, j] = np.sum((template - template_mean) * (patch - patch_mean)) / norm

	#result /= np.max(result)
	return result

def draw_rectangle_at_matches(image, template_h, template_w, matches):
	matches = zip(matches[0], matches[1])
	for match in matches:
		cv2.rectangle(image, (match[1], match[0]), (match[1] + template_w, match[0] + template_h), 0)
	return image

def task2():
	image = cv2.imread('../data/lena.png', 0)
	template = cv2.imread('../data/eye.png', 0)
	template_h, template_w = template.shape

	image_float = np.float32(image) / 255.
	template_float = np.float32(template) / 255.

	result_ssd = sum_square_difference(image_float, template_float)
	result_ncc = normalized_cross_correlation(image_float, template_float)

	cv2.imshow('Task2::SSD', draw_rectangle_at_matches(image.copy(),
			   template_h, template_w,
			   np.where(result_ssd >= 0.7)))

	cv2.imshow('Task2::NCC', draw_rectangle_at_matches(image.copy(), 
			   template_h, template_w,
			   np.where(result_ncc >= 0.7)))

#////////////////////////////////////////////////////////////////////////////////////
#//																				   //
#//									Task3    									   //
#//																				   //
#////////////////////////////////////////////////////////////////////////////////////
def build_gaussian_pyramid_opencv(image, num_levels):
	result = []
	result.append(image)
	for i in range(num_levels):
		image = cv2.pyrDown(image)
		result.append(image)
	return result

def build_gaussian_pyramid(image, num_levels, sigma):
	result = []
	result.append(image)
	for i in range(num_levels):
		blured_image = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)
		height, width = blured_image.shape
		image = blured_image[[i for i in range(0, height, 2)],:][:, [i for i in range(0, width, 2)]]
		result.append(image)

	return result

def template_matching_multiple_scales(pyramid_image, pyramid_template, threshold, window_size):
	pyramid_image = pyramid_image[::-1]
	pyramid_template = pyramid_template[::-1]

	similarity_map = normalized_cross_correlation(pyramid_image[0], pyramid_template[0])
	
	for i in range(1, len(pyramid_image)):
		#similarity_map /= similarity_map.max()
		matches = np.where(similarity_map >= threshold)
		matches = zip(matches[0], matches[1])
		#cv2.imshow('Patch:{0}'.format(i), similarity_map)
		similarity_map = np.zeros_like(pyramid_image[i])

		image_h, image_w = pyramid_image[i].shape
		template_h, template_w = pyramid_template[i].shape

		for match in matches:
			match = [x * 2 for x in match]

			patch_top_left = (match[0] - window_size if match[0] - window_size >= 0 else 0,
							  match[1] - window_size if match[1] - window_size >= 0 else 0)
			match[0] += template_h
			match[1] += template_w

			patch_bot_right = (match[0] + window_size if match[0] + window_size <= image_h else image_h,
							   match[1] + window_size if match[1] + window_size <= image_w else image_w)
			patch = pyramid_image[i][patch_top_left[0]:patch_bot_right[0], patch_top_left[1]:patch_bot_right[1]]
			
			similarity = normalized_cross_correlation(patch, pyramid_template[i])
			similarity_final = np.zeros_like(patch)

			similarity_final[:similarity.shape[0], :similarity.shape[1]] = similarity

			similarity_map_patch = similarity_map[patch_top_left[0]: patch_bot_right[0], patch_top_left[1]: patch_bot_right[1]]
			mask = np.float32(similarity_final > similarity_map_patch)

			similarity_map[patch_top_left[0]: patch_bot_right[0], patch_top_left[1]: patch_bot_right[1]] =\
				(mask * similarity_final + (1 - mask) * similarity_map_patch)
			
			#print('{0}::{1}'.format(similarity_map_patch.max(), similarity_map.max()))
			#sys.exit()

	return similarity_map

def task3():
	image = cv2.imread('../data/traffic.jpg', 0)
	template = cv2.imread('../data/traffic-template.png', 0)

	cv_pyramid = build_gaussian_pyramid_opencv(np.float32(image)/255., 4)
	
	mine_pyramid = build_gaussian_pyramid(np.float32(image)/255., 4, 1.)
	mine_pyramid_template = build_gaussian_pyramid(np.float32(template)/255., 4, 1.)

	for idx, img in enumerate(mine_pyramid):
		print('TASK3::Pyramid_Dif_level_{0}:{1}'.format(idx, np.abs(img - cv_pyramid[idx]).mean()))

	start = time.time()
	similarity_map_wo_pyramid = normalized_cross_correlation(np.float32(image), np.float32(template))
	elapsed = time.time()
	print('Task3::Time_Matching_without_pyramid:{0}'.format(elapsed - start))

	start = time.time()
	similarity_map = template_matching_multiple_scales(mine_pyramid, mine_pyramid_template, 0.7, 5)
	elapsed = time.time()
	print('Task3::Time_Matching_without_pyramid:{0}'.format(elapsed - start))

	cv2.imshow('Task3::WithoutPyramid', 
		draw_rectangle_at_matches(np.float32(image)/255, template.shape[0], template.shape[1], np.where(similarity_map_wo_pyramid >= 0.7)))
	cv2.imshow('Task3::WithPyramid',
		draw_rectangle_at_matches(np.float32(image)/255, template.shape[0], template.shape[1], np.where(similarity_map >= 0.7)))
	

#////////////////////////////////////////////////////////////////////////////////////
#//																				   //
#//									Task4    									   //
#//																				   //
#////////////////////////////////////////////////////////////////////////////////////
def get_exponent(x, y, sigma):
	return -1 * (x*x + y*y) / (2 * sigma)

def get_derivative_of_gaussian_kernel(size, sigma):
	assert size > 0 and size % 2 == 1 and sigma > 0

	kernel_x = np.zeros((size, size))
	kernel_y = np.zeros((size, size))

	size_half = size // 2

	for i in range(size):
		y = i - size_half
		for j in range(size):
			x = j - size_half
			kernel_x[i, j] = -1 * (x / (2 * np.pi * sigma * sigma)) * np.exp(get_exponent(x, y, sigma))
			kernel_y[i, j] = -1 * (y / (2 * np.pi * sigma * sigma)) * np.exp(get_exponent(x, y, sigma))

	print('kernel_x', kernel_x)
	print('kernel_y', kernel_y)
	return kernel_x, kernel_y

def task4():
	image = cv2.imread('../data/einstein.jpeg', 0)

	kernel_x, kernel_y = get_derivative_of_gaussian_kernel(5, 0.6)

	edges_x = cv2.filter2D(image, -1, kernel_x)
	edges_y = cv2.filter2D(image, -1, kernel_y)

	magnitude = np.float32(np.sqrt( edges_x * edges_x + edges_y * edges_y ))
	magnitude /= magnitude.max()

	direction = np.float32(np.arctan2(edges_y, edges_x))

	cv2.imshow('Task4::Magnitude', magnitude)
	cv2.imshow('Task4::Direction', direction)


#////////////////////////////////////////////////////////////////////////////////////
#//																				   //
#//									Task5    									   //
#//																				   //
#////////////////////////////////////////////////////////////////////////////////////
def get_edges_using_canny(image):
	image_blurred = cv2.GaussianBlur(image, (5, 5), sigmaX=0.0)
	edges = cv2.Canny(image_blurred, 100, 250, 3)
	edges = np.float32(edges) * 1/255.

	edges[np.where(edges <= 0.7)] = 0.0
	edges[np.where(edges != 0.0)] = 1.0

	return 1 - edges

def find_horizontal_intersection(f, v, q, k):
	return ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k])

def l2_distance_transform_1D(f, pos_inf, neg_inf):
	d = np.zeros_like(f) 					# Distance transform values
	v = np.zeros_like(f).astype(np.int32)   # Locations of parabolas in lower envelope
	z = np.zeros_like(f)					# Locations of intersection of two parabolas

	z[0] = neg_inf
	z[1] = pos_inf

	k = 0  # num of parabolas 
	for q in range(1, len(f)):
		s = find_horizontal_intersection(f, v, q, k)
		while s <= z[k]:  # less than prev found intersection.
			k -= 1
			s = find_horizontal_intersection(f, v, q, k)
		k += 1
		v[k] = q
		z[k] = s
		z[k+1] = pos_inf

	k = 0
	for q in range(len(f)):
		while z[k+1] < q:
			k+=1
		d[q] = np.power(q - v[k], 2) + f[v[k]]
	return d

def l2_distance_transform_2D(edge_function, positive_inf, negative_inf):
	edge_function *= positive_inf
	d = np.zeros_like(edge_function)
	for i in range(edge_function.shape[1]):
		d[:, i] = l2_distance_transform_1D(edge_function[:, i], positive_inf, negative_inf)
	edge_function = d
	d = np.zeros_like(edge_function)
	for i in range(edge_function.shape[0]):
		d[i, :] = l2_distance_transform_1D(edge_function[i, :], positive_inf, negative_inf)
	d = np.sqrt(d)
	d /= np.max(d)
	return d

def task5():
	image = cv2.imread('../data/traffic.jpg', 0)

	edge_function = get_edges_using_canny(image)
	cv2.imshow('Task5::Canny_Edges', edge_function)
	cv2.waitKey(0)

	dist_transfom_mine = l2_distance_transform_2D(edge_function.copy(), 1024*1024*2, -1024*1024*2)
	
	dist_transfom_cv = cv2.distanceTransform(np.uint8(edge_function * 255.), distanceType=cv2.DIST_L2, maskSize=cv2.DIST_MASK_PRECISE)
	dist_transfom_cv /= dist_transfom_cv.max()

	cv2.imshow('Task5::Dist_Trans_Mine', dist_transfom_mine)
	cv2.imshow('Task5::Dist_Trans_CV', dist_transfom_cv)

	print(np.abs(dist_transfom_mine - dist_transfom_cv).mean())
	

task1()
cv2.waitKey(0)
cv2.destroyAllWindows()
task2()
cv2.waitKey(0)
cv2.destroyAllWindows()
task3()
cv2.waitKey(0)
cv2.destroyAllWindows()
task4()
cv2.waitKey(0)
cv2.destroyAllWindows()
task5()
cv2.waitKey(0)
cv2.destroyAllWindows()




