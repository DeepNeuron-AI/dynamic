from pathlib import Path
import sys
import math
from typing import List, Tuple

import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from dotenv import dotenv_values
import scipy
from scipy import stats

import echonet

# Just some types for us to use in type hints to make dev easier
Point = List[np.intp]
Box = Tuple[Point, Point, Point, Point]
Rectangle = Tuple[Point, Tuple[float, float], float] # [centre, (width, height), angle]


class RVDisappeared(Exception):
    """
    When the RV segmentation entirely vanishes (i.e. no True values in mask) in 
    at least one frame.
    """
    pass


def get_angle(rect: Rectangle) -> float:
    """
    Converts the angle returned by opencv for a rotated rectangle into a more
    reliable one that doesn't require you to know which sides correspond to 
    "height" and which correspond to "width".

    Returns
    -------
    angle: float
        Angle in degrees in the range of (-90, 90).
    """
    _, (width, height), angle = rect

    # print(f"Height = {height}, width = {width}, angle = {angle}")
    # if height == width:
    #     print(f"Height = width = {width}, angle = {angle}")
    if angle == -90:
        return angle
    if width < height:
        return 0 - angle
    elif height <= width:
        return 90 - angle


BOTTOM_RIGHT = 0
BOTTOM_LEFT = 1
TOP_RIGHT = 2
TOP_LEFT = 3

def find_corner(rect: Rectangle, which: int) -> Point:
    """
    Returns the coordinates of either the "bottom-right", "bottom-left", 
    "top-left", or "top-right" corner of the given (possible rotated) rectangle.
    The specific corner is chosen with the `which` parameter.

    Note these corner names are a little ambiguous, but were defined based on
    what was most applicable to our needs.
    """
    angle = get_angle(rect)
    box = np.intp(cv2.boxPoints(rect))

    # Critical angle regions are: [-90, -45), [-45, 0], (0, 45), [45, 90) ?
    if (-90 <= angle < -45) or (0 < angle < 45):
        if which == BOTTOM_LEFT:
            return box[3]
        elif which == TOP_LEFT:
            return box[0]
        elif which == TOP_RIGHT:
            return box[1]
        elif which == BOTTOM_RIGHT:
            return box[2]
    elif (-45 <= angle <= 0) or (45 <= angle <= 90):
        if which == BOTTOM_LEFT:
            return box[0]
        elif which == TOP_LEFT:
            return box[1]
        elif which == TOP_RIGHT:
            return box[2]
        elif which == BOTTOM_RIGHT:
            return box[3]
    else:
        print(f"??????{angle}")


def mask_to_image(mask: np.ndarray, max_val: int = 255) -> np.ndarray:
    """
    Converts a boolean mask array to a pure black and white image. Useful if
    you start with a mask but then want to find contours and perform other image
    analysis on that mask    
    """
    return mask.astype(np.uint8) * max_val


def image_to_mask(image: np.ndarray, threshold: int = 1) -> np.ndarray:
    """
    Converts a black and white image to a boolean mask, selecting every pixel 
    whose intensity is **greater than or equal to the threshold**
    """
    return image >= threshold


def extrapolate_line(p1: Point, p2: Point, frame_height: int, frame_width: int) -> List[Point]:
    """
    Extends a line between the given pair of points for all given x values and
    returns the list of points on this line.
    
    This clips the points for you in case any of the given x values correspond to
    y values outside of the image's borders.
    """
    # Just to be safe, make sure points are in ascending order of x (unless 
    # vertically alligned)
    if p2[0] < p1[0]:
        p1, p2 = p2, p1

    x1, y1 = p1
    x2, y2 = p2

    # Vertical line is special case
    if x1 == x2:
        # Vertical lines are a special case
        ys = np.arange(frame_height)
        xs = np.ones(ys.shape, dtype=np.intp) * x1
    else:
        gradient = (y2 - y1) / (x2 - x1)
        y_intercept = y1 - gradient * x1
        
        # Whether the straight line will be steep or not. If it *is* steep, then that
        # means that a single x value can correspond to multiply y values, due to
        # pixellated nature of the line, and vice versa if it is *not* steep. We
        # therefore need to treat the two steepness cases separately to ensure a
        # fully continuous line.
        is_steep = abs(y2 - y1) > abs(x2 - x1)
        if is_steep:
            # Start with all y values, and determine matching xs
            ys = np.arange(frame_height)
            #    y = mx + c
            # => x = (y - c) / m
            xs = (ys - y_intercept) / gradient
        else:
            # Start with all x values, and determine matching ys
            xs = np.arange(frame_width)
            ys = gradient * xs + y_intercept
    
    # Make sure coordinates are all corect type for opencv indexing
    xs = np.intp(xs)
    ys = np.intp(ys)
    points = list(zip(xs, ys))
    return points


def get_min_area_rect(image: np.ndarray) -> Rectangle:
    """
    Finds the largest contour in the given image, and returns the minimum area
    rectangle that bounds it.

    Returns
    -------
    ((centre_x, centre_y), (width, height), angle)
    """
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Assume we're only interested in max area contour
    areas = [cv2.contourArea(cnt) for cnt in contours]
    max_index = np.argmax(areas)
    biggest_contour = contours[max_index]

    min_area_rect = cv2.minAreaRect(biggest_contour)

    centre, (width, height), angle = min_area_rect
    if angle == 90:
        # Rotate counterclockwise by 90 degrees if angle is 90 degrees
        min_area_rect = (centre, (height, width), 0)

    return min_area_rect


def get_min_area_box(image: np.ndarray) -> Box:
    """
    Similar to get_min_area_rect(), but instead returns the coordinates of the
    box's four corners.
    """
    min_area_rect = get_min_area_rect(image)
    if min_area_rect is None:
        return None
    box = np.intp(cv2.boxPoints(min_area_rect))
    
    return box


def perpendicular_distance_to_line(line: List[List[int]], point: List[int]) -> float:
    """
    Returns the perpendicular distance from a straight line to a given point.

    Parameters
    ----------
    line: List[List[int]]
        `line` is a *pair* of points, where each point contains a single x and y
        value.
    point: List[int]
        A pair of x and y values.
    """
    (x1, y1), (x2, y2) = line
    x3, y3 = point

    if x2 == x1:
        return abs(x1 - x3)

    m = (y2 -y1) / (x2 - x1)
    a = -1
    b = 1 / m
    c = x1 - y1 / m

    d = abs(a * x3 + b * y3 + c) / math.sqrt(a**2 + b**2)
    return d


def crop_mask_with_line(mask: np.ndarray, line: List[Point], below=False, above=False, left=False, right=False):
    """
    Sets all pixels to some side(s) of the given line to False in the given 
    mask, effectively cropping the mask with an arbitrary line.

    Note that the given line can technically have points that are out of bounds
    of the frame. This is actually needed so we can crop based on particularly
    steep lines (e.g. the LV's septum border, which likely only takes up a few
    x values in the actual frame). 

    This modifies the given `mask` in-place.
    """
    mask_height, mask_width = mask.shape

    crop_count = 0
    for x, y in line:
        y_is_valid = (0 <= y < mask_height)
        x_is_valid = (0 <= x < mask_width)

        if below and x_is_valid:
            y = min(y, mask_height-1)
            crop_count += mask[y:, x].sum()
            mask[y:, x] = False
        if above and x_is_valid:
            y = max(y, 0)
            crop_count += mask[:y, x].sum()
            mask[:y, x] = False
        if left and y_is_valid:
            x = min(x, mask_width-1)
            crop_count += mask[y, :x].sum()
            mask[y, :x] = False
        if right and y_is_valid:
            x = max(x, 0)
            crop_count += mask[y, x:].sum()
            mask[y, x:] = False

    return crop_count


def crop_line_to_frame(line: List[Point], frame_height: int, frame_width: int) -> List[Point]:
    """
    Returns a subset of the given `line` that fits within the given frame 
    dimensions. This is useful if you have a line that potentially goes beyond 
    the frame, and you want to show that line using opencv (which will complain
    about index errors if any given points are out of bounds).
    """
    # We could probs use cv2.clipLine(), but that only returns the two endpoints,
    # so we'd still have to find all points in between anyway!
    cropped_line = []
    for point in line:
        x, y = point
        if (0 <= x < frame_width) and (0 <= y < frame_height):
            cropped_line.append(point)

    return cropped_line


def clip_and_draw_line(frame: np.ndarray, line: List[Point], colour: Tuple[int, int, int]):
    """
    Helper function to draw the given line on the given frame, making sure to clip
    the line to the frame to avoid any out-of-bounds errors.
    """
    frame_height, frame_width, _ = frame.shape
    img_rect = (0, 0, frame_width, frame_height)
    in_bounds, point1, point2 = cv2.clipLine(img_rect, line[0], line[-1])
    
    cv2.line(frame, point1, point2, colour)


def contains_RV(LV_masks: np.ndarray, RV_masks: np.ndarray) -> bool:
    """
    Returns whether we think RV even exists within the given video. This is useful
    for distinguishing between our normal A4C videos and those ones that are
    cropped to remove the RV.

    :return: True if we think RV is in video, False otherwise.
    """
    intersection = LV_masks & RV_masks
    intersection_counts = intersection.sum(axis=(1, 2))
    union = LV_masks | RV_masks
    union_counts = union.sum(axis=(1, 2))
    ious = intersection_counts / union_counts
    
    return np.mean(ious) < 0.5


def did_ventricle_disappear(RV_masks: np.ndarray) -> bool:
    """
    Returns True if the given segmentations contain at least one frame where
    the RV mask has **zero** pixels in it. If this happens, the RV segmentation
    should be considered practically useless, since the RV shouldn't collapse to
    a singularity in a normal human being.
    """
    return any(np.sum(RV_masks, axis=(1, 2)) == 0)

def cutoff_from_LV_box(LV_masks, RV_masks) -> np.ndarray:
    """
    Returns
    -------
    RV_masks
    """
    LV_segmentations = mask_to_image(LV_masks)
    num_frames, frame_height, frame_width = LV_masks.shape

    LV_rects = [get_min_area_rect(LV_segmentation) for LV_segmentation in LV_segmentations]

    valve_cutoff_points_list = []
    apex_cutoff_points_list = []
    LV_septum_border_cutoff_points_list = []

    original_pixel_count = RV_masks.sum()
    crop_count_total = 0
    for i, (LV_rect, RV_segmentation_mask) in enumerate(zip(LV_rects, RV_masks)):
        top_left_LV = find_corner(LV_rect, TOP_LEFT)
        top_right_LV = find_corner(LV_rect, TOP_RIGHT)
        bottom_left_LV = find_corner(LV_rect, BOTTOM_LEFT)
        bottom_right_LV = find_corner(LV_rect, BOTTOM_RIGHT)

        # Remove any points from RV segmentation that seem to be *below* tricuspid valve (e.g. in RA)
        valve_cutoff_points = extrapolate_line(bottom_left_LV, bottom_right_LV, frame_height=frame_height, frame_width=frame_width)
        valve_cutoff_points_list.append(valve_cutoff_points)
        crop_count_total += crop_mask_with_line(mask=RV_segmentation_mask, line=valve_cutoff_points, below=True)

        # Remove any points from RV segmentation that seem to be *above* heart's apex
        apex_cutoff_points = extrapolate_line(top_left_LV, top_right_LV, frame_height=frame_height, frame_width=frame_width)
        apex_cutoff_points_list.append(apex_cutoff_points)
        crop_count_total += crop_mask_with_line(mask=RV_segmentation_mask, line=apex_cutoff_points, above=True)

        # Remove any points from RV segmentation that seem to be *right* of the LV's inner edge/septum wall
        LV_septum_border_cutoff_points = extrapolate_line(bottom_left_LV, top_left_LV, frame_height=frame_height, frame_width=frame_width)
        LV_septum_border_cutoff_points_list.append(LV_septum_border_cutoff_points)
        crop_count_total += crop_mask_with_line(mask=RV_segmentation_mask, line=LV_septum_border_cutoff_points, right=True)

    print(f"Cropped out: {crop_count_total/original_pixel_count:.6f}")

    if did_ventricle_disappear(RV_masks):
        raise RVDisappeared("RV disappeared in at least one frame after `cutoff_from_LV_box()`")

    return RV_masks

def get_largest_contour(RV_masks: np.ndarray) -> np.ndarray:
    """
    Returns
    -------
    RV_masks
    """
    RV_segmentations = mask_to_image(RV_masks)

    # Only include largest contour of right segmentation for each frame. This removes
    # any smaller, disconnected blobs
    RV_segmentations_copy = RV_segmentations.copy()
    RV_segmentations = np.zeros(RV_segmentations.shape, dtype=RV_segmentations.dtype)
    for i, RV_segmentation in enumerate(RV_segmentations_copy):
        right_contours, hierarchy = cv2.findContours(RV_segmentation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(right_contours) == 0:
            raise RVDisappeared(f"RV segmentation completely vanished at frame {i}")
        areas = [cv2.contourArea(cnt) for cnt in right_contours]
        max_index = np.argmax(areas)
        biggest_contour = right_contours[max_index]

        biggest_RV_segmentation = np.zeros(RV_segmentation.shape)
        biggest_RV_segmentation = cv2.drawContours(biggest_RV_segmentation, [biggest_contour], -1, 255, -1)
        RV_segmentations[i] = biggest_RV_segmentation

    # Update masks too, would be nicer to have these masks and images automatically linked though!
    RV_masks = image_to_mask(RV_segmentations)
    return RV_masks


def crop_ultrasound_borders(RV_masks: np.ndarray) -> np.ndarray:
    num_frames, frame_height, frame_width = RV_masks.shape

    left_ultrasound_corner = (0, 67)
    top_ultrasound_corner = (61, 6)
    ultrasound_left_line = extrapolate_line(left_ultrasound_corner, top_ultrasound_corner, frame_height, frame_width)
    for s in RV_masks:
        crop_mask_with_line(s, ultrasound_left_line, above=True)

    # Repeat for right side of ultrasound
    right_ultrasound_corner = (111, 55)
    top_ultrasound_corner = (62, 6)
    ultrasound_right_line = extrapolate_line(right_ultrasound_corner, top_ultrasound_corner, frame_height, frame_width)
    for s in RV_masks:
        crop_mask_with_line(s, ultrasound_right_line, above=True)

    if did_ventricle_disappear(RV_masks):
        raise RVDisappeared("RV disappeared in at least one frame after `crop_ultrasound_borders()`")

    return RV_masks


def remove_septum(LV_masks, RV_masks, scaling_factor: float = 1.0) -> np.ndarray:
    LV_segmentations = mask_to_image(LV_masks)
    RV_segmentations = mask_to_image(RV_masks)
    num_frames, frame_height, frame_width = LV_masks.shape
    
    # Get end-point pairs for edge of LV near septum (i.e. the "left" edge)
    LV_rects = [get_min_area_rect(LV_segmentation) for LV_segmentation in LV_segmentations]
    LV_septum_borders = [[find_corner(rect, BOTTOM_LEFT), find_corner(rect, TOP_LEFT)] for rect in LV_rects]

    # Find septum widths
    septum_widths = []

    for (RV_segmentation, LV_line) in zip(RV_segmentations, LV_septum_borders):
        RV_rect = get_min_area_rect(RV_segmentation)
        RV_bottom_right = find_corner(RV_rect, BOTTOM_RIGHT)

        septum_width = perpendicular_distance_to_line(LV_line, RV_bottom_right)
        septum_widths.append(septum_width)

    RV_boxes = [get_min_area_box(RV_segmentation) for RV_segmentation in RV_segmentations]

    # Use average estimated septum width and translate the LV segmentation's inner
    # edge by that amount to guess the right edge of the RV.
    mean_septum_width = np.mean(septum_widths)

    # Potentially downscale septum width. Do this because we noticed some videos 
    # have highly over-estimated widths, which leads to bad cropping
    mean_septum_width *= scaling_factor

    RV_boxes = []
    RV_lines = []
    for LV_rect, LV_line, RV_segmentation_mask, RV_segmentation in zip(LV_rects, LV_septum_borders, RV_masks, RV_segmentations):
        LV_angle = get_angle(LV_rect)
        
        # Since we translate points *left*, and not necessarily perpendicular to LV
        # line, we include factor of sin(angle)
        translate_x = mean_septum_width * math.cos(math.radians(LV_angle))
        RV_line = [point.copy() for point in LV_line]
        for point in RV_line:
            point[0] -= translate_x

        RV_lines.append(RV_line)

        RV_box = get_min_area_box(RV_segmentation)
        # Just in case we need to access this later
        RV_boxes.append(RV_box)

        # Now remove any pixels in RV segmentation that go beyond its expected inner
        # edge
        # Note this is a greedier "right cutoff" than before, should probs just name this better!
        right_cutoff_points = extrapolate_line(RV_line[0], RV_line[1], frame_height=frame_height, frame_width=frame_width)
        crop_mask_with_line(RV_segmentation_mask, right_cutoff_points, right=True)

    if did_ventricle_disappear(RV_masks):
        raise RVDisappeared("RV disappeared in at least one frame after `remove_septum()`")

    return RV_masks


def replace_tiny_RV_frames(RV_masks) -> np.ndarray:
    RV_segmentations = mask_to_image(RV_masks)
    num_frames, frame_height, frame_width = RV_masks.shape

    RV_contours_list = [cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0][0] for frame in RV_segmentations]
    RV_areas = np.array([cv2.contourArea(contour) for contour in RV_contours_list])
    mean_RV_area = np.mean(RV_areas)
    std_RV_area = np.std(RV_areas)

    z = 1.5
    small_RV_indices = np.where(mean_RV_area - z * std_RV_area > RV_areas)[0]

    RV_areas_copy = RV_areas.copy()
    for i in small_RV_indices:
        offset = 1
        while True:
            left_i = i - offset
            if left_i >= 0 and left_i not in small_RV_indices:
                RV_masks[i] = RV_masks[left_i]
                RV_areas[i] = RV_areas_copy[left_i]
                break

            right_i = i + offset
            if right_i < num_frames and right_i not in small_RV_indices:
                RV_masks[i] = RV_masks[right_i]
                RV_areas[i] = RV_areas_copy[right_i]
                break

            offset += 1

    return RV_masks


def get_LV_RV_area_correlation(LV_masks, RV_masks) -> float:
    LV_areas = np.sum(LV_masks, axis=(1,2))
    RV_areas = np.sum(RV_masks, axis=(1,2))
    return stats.pearsonr(LV_areas, RV_areas).correlation


def get_average_eccentricity(masks: np.ndarray) -> float:
    """
    Returns the average eccentricity of the bounding boxes for the given video's
    masks. This may be a useful metric for determining how "reasonable" these
    segmentations are.

    Note this skips over frames that have a zero height/width bounding box.

    Returns
    -------
    eccentricity: float
        Should be in range of [0, 1].
    """
    images = mask_to_image(masks)
    eccentricities = np.zeros(len(images))
    
    for i, image in enumerate(images):
        ((centre_x, centre_y), (width, height), angle) = get_min_area_rect(image)
        
        # Ignore frames with zero length
        if height == 0 or width == 0:
            continue
        # Take min to ensure eccentricity is <= 1
        eccentricities[i] = min(width / height, height / width)

    eccentricities = np.sqrt(1 - eccentricities**2)
    return eccentricities.mean()