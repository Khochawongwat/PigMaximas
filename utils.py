import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

def pad(img, targets):
    height, width = img.shape
    target_height, target_width = targets

    if height < target_height or width < target_width:
        top_padding = (target_height - height) // 2
        bottom_padding = target_height - height - top_padding
        left_padding = (target_width - width) // 2
        right_padding = target_width - width - left_padding

        img = cv.copyMakeBorder(img, top_padding, bottom_padding, left_padding, right_padding,
                                        cv.BORDER_CONSTANT, value=0)
    else:
        img = cv.resize(img, (target_width, target_height))
    return img

def _calculate_centroid(contour):
    M = cv.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0
    return cX, cY

def _calculate_direction(covariance_matrix):
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    max_index = np.argmax(eigenvalues)
    return eigenvectors[:, max_index]

def _create_half_image(mask, y_line, condition):
    half_image = np.zeros_like(mask)
    for x in range(mask.shape[1]):
        y_boundary = y_line(x)
        for y in range(mask.shape[0]):
            if mask[y, x] == 255 and condition(y, y_boundary):
                half_image[y, x] = 255
    return half_image

def _calculate_orientation_and_rotate(img, direction, angle_degrees):
    image_center = (img.shape[1] // 2, img.shape[0] // 2)
    rotation_matrix = cv.getRotationMatrix2D(image_center, angle_degrees, 1.0)

    rotated_img = cv.warpAffine(img, rotation_matrix, img.shape[1::-1], flags=cv.INTER_LINEAR)
    return rotated_img

def _slice_half(img):
    contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    largest_contour = max(contours, key=cv.contourArea)
    
    cX, cY = _calculate_centroid(largest_contour)
    data_points = largest_contour[:, 0, :]
    mean = np.mean(data_points, axis=0)
    centered_data = data_points - mean
    covariance_matrix = np.cov(centered_data.T)

    direction = _calculate_direction(covariance_matrix)
    angle_degrees = np.degrees(np.arctan2(direction[1], direction[0]))

    mask = np.zeros_like(img)
    cv.drawContours(mask, [largest_contour], -1, (255), thickness=cv.FILLED)

    # Create upper and lower half images
    y_line = lambda x: (direction[1] / direction[0]) * (x - cX) + cY if direction[0] != 0 else float('inf')
    
    upper_half_image = _create_half_image(mask, y_line, lambda y, y_boundary: y < y_boundary)
    lower_half_image = _create_half_image(mask, y_line, lambda y, y_boundary: y > y_boundary)

    upper_half_image = cv.resize(upper_half_image, img.shape[1::-1], interpolation=cv.INTER_LINEAR)
    lower_half_image = cv.resize(lower_half_image, img.shape[1::-1], interpolation=cv.INTER_LINEAR)

    img = _calculate_orientation_and_rotate(img, direction, angle_degrees)
    upper_half_image = _calculate_orientation_and_rotate(upper_half_image, direction, angle_degrees)
    lower_half_image = _calculate_orientation_and_rotate(lower_half_image, direction, angle_degrees)
    
    lower_half_image = cv.flip(lower_half_image, 0)

    return img, upper_half_image, lower_half_image, (direction, angle_degrees)

def _fill_gaps(img, kernel_size = (5, 5)):
    kernel = np.ones(kernel_size, np.uint8)
    return cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)

def _get_valley_coords(img, n = 4, plot = True):
    contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv.contourArea)

    outline = np.zeros_like(img)
    cv.drawContours(outline, [largest_contour], -1, (255), thickness=1)

    center_line_width = 50
    center_row = outline.shape[0] // 2

    outline[center_row + center_line_width // 2 :, :] = 0
    outline[center_row - center_line_width // 2 :, :] = 0

    outline = outline[::-1]

    y_indices, x_indices = np.nonzero(outline)

    margin = 70

    valid_mask = (x_indices >= margin) & (x_indices < outline.shape[1] - margin)
    filtered_y_indices = y_indices[valid_mask]
    filtered_x_indices = x_indices[valid_mask]

    indices = np.array(list(zip(filtered_y_indices, filtered_x_indices)))

    sorted_indices = indices[np.argsort(indices[:, 1])]

    if sorted_indices.size > 0:
        interp_func = interp1d(
            sorted_indices[:, 1],
            sorted_indices[:, 0],
            kind="linear",
            fill_value="extrapolate",
        )
        x_new = np.linspace(sorted_indices[:, 1].min(), sorted_indices[:, 1].max(), num=500)
        y_new = interp_func(x_new)

        interpolated_coords = np.column_stack(
            (np.round(y_new).astype(int), x_new.astype(int))
        )

        inverted_y = -interpolated_coords[:, 0]

        valleys, _ = find_peaks(inverted_y, height=None, prominence=1, distance=30)
    else:
        valleys = np.array([])

    if len(valleys) > 0:
        coords = interpolated_coords[valleys]
        coords = coords[
            np.argsort(coords[:, 0])[::-1]
        ]

        if len(coords) > n:
            coords = coords[:n]

        coords = coords[
            np.argsort(coords[:, 1])
        ]
    else:
        coords = np.empty((0, 2))

    if len(coords) >= 3:
        if coords[2][0] > coords[0][0]:
            coords = np.delete(coords, 1, axis=0)

    if plot:
        plt.figure(figsize=(5, 4))
        plt.imshow(cv.cvtColor(outline, cv.COLOR_GRAY2BGR))

        for valley in coords:
            plt.axvline(x=valley[1], color="red", linestyle="--", linewidth=1)

        plt.scatter(
            coords[:, 1],
            coords[:, 0],
            color="green",
            s=25,
            label="Local Valleys",
        )
        plt.title(f"Local Valleys (Bottom {n})")
        plt.axis("off")
        plt.legend()
        plt.show()

    return coords

def _create_section(img, start_x, end_x):
    start_x = max(0, start_x)
    end_x = min(img.shape[1], end_x)
    mask = np.zeros(img.shape, dtype=np.uint8)
    cv.rectangle(mask, (start_x, 0), (end_x, img.shape[0] - 1), color=255, thickness=cv.FILLED)
    section = cv.bitwise_and(img, mask)
    #Some parts gets cut off from a different section
    largest_section = _get_largest_connected_component(section)
    return largest_section

def _get_largest_connected_component(img):
    _, labels, stats, _ = cv.connectedComponentsWithStats(img, connectivity=8)
    largest_component = 1 + np.argmax(stats[1:, cv.CC_STAT_AREA])
    largest_mask = (labels == largest_component).astype(np.uint8)
    largest_section = cv.bitwise_and(img, img, mask=largest_mask)
    return largest_section

def section(img, img_width = 512, sections = 4):

    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.GaussianBlur(img, (5, 5), 0)
    
    #We want horizontal images so if height is more than width then transpose.
    if img.shape[0] > img.shape[1]:
        img = img.T
    
    t_img = img.copy()

    #Threshold
    img = np.where(img > 1, 255, 0).astype(np.uint8)

    #Resize to match pigs anatomy. try getting the ratios of the pig it should be around 0.36-40.
    targets = (int(img_width * 0.36), img_width)

    #Pad for spaces around the pig
    img = pad(img, targets)

    t_img= pad(t_img, targets)

    #Slice the img in half based on the eigenvector. Basically find the center of the pig using eigen values.
    #Then find the direction with the highest relevancy/pixels using eigen vectors.
    #Then find the slope of that direction and cut in half.
    img, half_img, lower_half_img, (direction, degrees) = _slice_half(img)

    t_img = _calculate_orientation_and_rotate(t_img, direction, degrees)

    #Sanity check just in case some gaps were created during the slicing.
    half_img = _fill_gaps(half_img)

    #Get outline of the pig. Then we remove the gap in the middle of the pig.
    #(since we sliced in half, the bottom of the pig is now just a straight horizontal line so we remove that to make sure the algo doesnt think that they are valleys)
    #After getting outline, we can use linear interpolation betwen each pixels in the outline to make it continuous by smoothing the pixels.
    #The image is then flipped because we have 0th index. Basically the algorithm thinks y = 0 is the bottom so we have to flip it. Use distance > 10 to avoid the clustering of valleys.
    #The coordinates are then sorted based on the y-axis to find the deepest valleys. Then sorted by on the x coordiantes ascendingly. (we go from 0 to width in the x-axis)
    #The margin is to avoid edge valleys. I found 70 to be most optimal but will have to change if your image size changes. Can edit to be ratio like 70/512 instead.
    coords = _get_valley_coords(half_img, n = sections)

    #Some pigs from some angles have round side so we can use the otherside but flipped instead
    if len(coords) < sections - 1:
        lower_half_img = _fill_gaps(lower_half_img)
        coords = _get_valley_coords(lower_half_img, n = sections)

    sections = []
    
    if len(coords) > 0:
        
        # Section 1: Legs
        start_x1, end_x1 = 0, coords[0][1]
        section1 = _create_section(t_img, start_x1, end_x1)
        sections.append(section1)

        # Section 2: Body
        if len(coords) > 1:
            start_x2, end_x2 = coords[0][1], coords[1][1]
            section2 = _create_section(t_img, start_x2, end_x2)
            sections.append(section2)

        # Section 3: Shoulder
        if len(coords) > 2:
            start_x3, end_x3 = coords[1][1], coords[2][1]
            section3 = _create_section(t_img, start_x3, end_x3)
            sections.append(section3)

        # Section 4: Head
        start_x4, end_x4 = coords[2][1], t_img.shape[1]
        section4 = _create_section(t_img, start_x4, end_x4)
        sections.append(section4)

    return np.array(sections)