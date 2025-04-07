import cv2
import numpy as np
from skimage.filters import threshold_otsu

def compute_saliency_map(image):
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    (success, saliencyMap) = saliency.computeSaliency(image)
    saliencyMap = (saliencyMap * 255).astype("uint8")
    #save saliency
    cv2.imwrite('saliency_map.png', saliencyMap)  
    return saliencyMap

def compute_visual_weights(saliency_map):
    T = threshold_otsu(saliency_map)
    visual_weights = np.where(saliency_map > T, saliency_map, 0)
    cv2.imwrite('visual_weights.png', visual_weights)  # Save visual weights mask
    return visual_weights


def compute_center_of_mass(weight_map):
    h, w = weight_map.shape
    total_weight = np.sum(weight_map)
    #return image center if no weight
    if total_weight == 0:
        return w // 2, h // 2  

    # get cenre of mass
    x_coords = np.arange(w)
    y_coords = np.arange(h)
    x_center = np.sum(np.sum(weight_map, axis=0) * x_coords) / total_weight
    y_center = np.sum(np.sum(weight_map, axis=1) * y_coords) / total_weight
    print(f"center of mass: {x_center}, {y_center}")
    return int(x_center), int(y_center)

def determine_opposite_quadrant(center_of_mass, image_shape):
    h, w = image_shape[:2]
    cx, cy = w // 2, h // 2
    x, y = center_of_mass

    # find center of mass quadrant
    if x < cx and y < cy:
        return (cx, cy)  # bottom-right
    elif x >= cx and y < cy:
        return (0, cy)   # bottom-left
    elif x < cx and y >= cy:
        return (cx, 0)   # top-right
    else:
        return (0, 0)    # top-left

def grid_and_place_text(image, target_corner, text_size):
    #get img height and width
    h, w = image.shape[:2]
    # get text height and width
    tw, th = text_size
    best_pos = None
    best_distance = float('inf')

    #calc how many cells fit
    grid_w = w // tw
    grid_h = h // th

    for i in range(grid_w):
        for j in range(grid_h):
            x = i * tw
            y = j * th

            #skip cells outside of image
            if x + tw > w or y + th > h:
                continue
            # get cemter of cell
            center_of_block = (x + tw // 2, y + th // 2)
            # calc manhattan distance to corner
            dist = np.abs(center_of_block[0] - target_corner[0]) + np.abs(center_of_block[1] - target_corner[1])

            # if cell closer than current closes update
            if dist < best_distance:
                best_distance = dist
                best_pos = (x, y)

    return best_pos


def adjust_position_to_preferred(pos, text_size, image_shape, 
                                 diag_weight=0.7, buffer=20):
    """
    Adjust the given text block position so that its center is pulled toward both 
    the diagonal of the nearest corner and the image center, with a bit of a buffer from the boundaries.
    
    Parameters:
      pos (tuple): initial (x, y) position (top-left of text block)
      text_size (tuple): (text_width, text_height)
      image_shape (tuple): shape of the image (height, width, channels)
      diag_weight (float): weight for the diagonal projection (0.0 to 1.0). 
                           (1-diag_weight) is the weight for the image center.
      buffer (int): number of pixels to keep as a margin from the image boundaries.
      
    Returns:
      (new_x, new_y): the adjusted top-left coordinate of the text block.
    """
    h, w = image_shape[:2]
    x, y = pos
    tw, th = text_size

    # text block center
    cx, cy = x + tw // 2, y + th // 2

    # image corners
    corners = {
        "top-left": (0, 0),
        "top-right": (w, 0),
        "bottom-left": (0, h),
        "bottom-right": (w, h)
    }
    # find nearest corner
    corner_name, nearest_corner = min(corners.items(), 
                                      key=lambda kv: np.linalg.norm(np.array([cx, cy]) - np.array(kv[1])))
    
    # compute projection onto  bisecting diagonal 
    if corner_name == "top-left":
        #  y = x
        proj = ((cx + cy) // 2, (cx + cy) // 2)
    elif corner_name == "top-right":
        #y = -x + w
        # solve for projection point: (proj_x, proj_y) where proj_y = -proj_x + w and the perpendicular from (cx,cy) meets that line.
        proj_x = (cx - cy + w) / 2
        proj_y = (-cx + cy + w) / 2
        proj = (proj_x, proj_y)
    elif corner_name == "bottom-left":
        #y = -x + h
        proj_x = (cx + cy - h) / 2
        proj_y = (h - cx + cy) / 2
        proj = (proj_x, proj_y)
    elif corner_name == "bottom-right":
        #y = x - (w - h)  [if h < w] or similar
        proj = ((cx + cy + (h - w)) / 2, (cx + cy + (h - w)) / 2)
    
    # blend with image center
    center = (w / 2, h / 2)
    blended_x = diag_weight * proj[0] + (1 - diag_weight) * center[0]
    blended_y = diag_weight * proj[1] + (1 - diag_weight) * center[1]
    
    # calculate new top left coordinates
    new_x = int(blended_x - tw // 2)
    new_y = int(blended_y - th // 2)
    
    # apply buffer from boundries
    new_x = max(buffer, min(w - tw - buffer, new_x))
    new_y = max(buffer, min(h - th - buffer, new_y))
    
    return (new_x, new_y)

def place_text_using_visual_balance(image, text_size=(100, 50), use_diagonal=False):
    # get saliency map
    saliency_map = compute_saliency_map(image)
    # compute weight map
    weight_map = compute_visual_weights(saliency_map)
    # find center of visual mass
    center_mass = compute_center_of_mass(weight_map)
    # find opposite quadrant
    target_corner = determine_opposite_quadrant(center_mass, image.shape)
    # get text position
    text_position = grid_and_place_text(image, target_corner, text_size)

    if use_diagonal and text_position:
        # refinw with diagonal method
        text_position = adjust_position_to_preferred(text_position, text_size, image.shape)

    return text_position