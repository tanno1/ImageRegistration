import numpy as np
import cv2
from scipy.optimize import minimize

def test_2_full(file, blend_file, k, block_size, aperture_size): 
    # load the unregistered and reference images
    unreg = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    reference = cv2.imread(blend_file, cv2.IMREAD_UNCHANGED)

    # convert images to float32 for processing
    unreg = unreg.astype(np.float32)
    reference = reference.astype(np.float32)

    # normalize the values to the range [0, 1]
    unreg_norm = cv2.normalize(unreg, None, 0, 1, cv2.NORM_MINMAX)
    reference_norm = cv2.normalize(reference, None, 0, 1, cv2.NORM_MINMAX)

    threshold = 0.8

    # Run Harris corner detection
    dst_unreg = cv2.cornerHarris(unreg_norm, block_size, aperture_size, k)
    dst_reference = cv2.cornerHarris(reference_norm, block_size, aperture_size, k)

    # Normalize the corner response maps
    dst_unreg = cv2.normalize(dst_unreg, None, 0, 1, cv2.NORM_MINMAX)
    dst_reference = cv2.normalize(dst_reference, None, 0, 1, cv2.NORM_MINMAX)

    # threshold the corner response maps to find strong corners
    unreg_strong_corners = dst_unreg > threshold
    ref_strong_corners = dst_reference > threshold

    unreg_strong_coords = np.argwhere(unreg_strong_corners)
    ref_strong_coords = np.argwhere(ref_strong_corners)


    # Get the coordinates of strong corners
    corner_y, corner_x = np.where(unreg_strong_corners)

    # extract small areas around corners
    def extract_patch(img, center, size=11):
        half = size //2
        x, y = center
        if x - half < 0 or y - half < 0 or x + half >= img.shape[1] or y + half >= img.shape[0]:
            return None # skip edge
        return img[y - half:y + half + 1, x - half:x + half + 1]

    # extract patches around strong corners with NCC
    def NCC(patch1, patch2):
        if patch1 is None or patch2 is None:
            return 0
        patch1 = patch1 - np.mean(patch1)
        patch2 = patch2 - np.mean(patch2)
        denom = np.std(patch1) * np.std(patch2)
        if denom == 0 :
            return -1
        return np.mean(patch1 * patch2) / denom

    matches = []

    for pt1 in ref_strong_coords:
        #print(f"Processing point {pt1}")
        best_score = -1
        best_match = None
        patch = extract_patch(reference_norm, pt1[::-1])
        #print(patch)
        for pt2 in unreg_strong_coords:
            #print(f"Comparing with point {pt2}")
            patch2 = extract_patch(unreg_norm, pt2[::-1])
            #print(patch2)
            score = NCC(patch, patch2)
            if score > best_score:
                best_score = score 
                best_match = pt2
        if best_score > .5:  # threshold for a good match
            matches.append((tuple(pt1), tuple(best_match)))

    # print(f"Number of matches found: {len(matches)}")
    if len(matches) < 3:
        print(f"Warning: not enough matches, registration may be poor for image {file}")
        return None, None, None, None

    # plot matches on both images
    # Extract matched points in unregistered image
    unreg_pts = [pt[1] for pt in matches]
    x_unreg = [p[1] for p in unreg_pts]
    y_unreg = [p[0] for p in unreg_pts]


    # Extract matched points in reference image
    ref_pts = [pt[0] for pt in matches]
    x_ref = [p[1] for p in ref_pts]
    y_ref = [p[0] for p in ref_pts]

    # apply affine transformation to each image
    pts_ref = np.float32([pt[0] for pt in matches])
    pts_unreg = np.float32([pt[1] for pt in matches])
    M, inliers = cv2.estimateAffinePartial2D(pts_unreg, pts_ref, method=cv2.RANSAC)

    h, w = reference.shape[:2]
    registered = cv2.warpAffine(unreg, M, (w, h), flags=cv2.INTER_CUBIC)        # bicubic interpolation for better quality
    registered_norm = (registered - registered.min()) / (registered.max() - registered.min())
    registered_uint16 = (registered_norm * 65535).astype(np.uint16)

    # Make sure both are float32 and same shape
    abs_diff = np.abs(reference.astype(np.float32) - registered.astype(np.float32))
    print(f"Percentage calculation: {np.mean(abs_diff) / (reference.max() - reference.min()) * 100:.2f}%")
    percentage_diff = np.mean(abs_diff) / (reference.max() - reference.min()) * 100

    return registered_uint16, abs_diff, percentage_diff, matches

def objective(params, file, blend_file):
    k, block_size, aperture_size = params
    
    # Aperture size must be odd and >= 3
    if aperture_size % 2 == 0 or aperture_size < 3:
        return np.inf  # Return a high value for invalid parameters
    
    print(f"Testing parameters: k={k}, block_size={block_size}, aperture_size={aperture_size}")
    
    # call test fn
    registered, abs_diff, percentage_diff, matches = test_2_full(file, blend_file, k, int(block_size), int(aperture_size))

    if registered is None:
        return np.inf  # Return a large value if registration fails
    
    return percentage_diff

# registered_uint16, abs_diff, percent_diff, matches = test_2_full(tif_path, ref, K_var.get(), BlockSize_var.get(), Aperature_var.get())

def iterator(tif_path, ref, K_var, BlockSize_var, Aperature_var):
    initial_params = [K_var, BlockSize_var, Aperature_var]  # Initial guess for k, block_size, aperture_size

    bounds = [
        (0.01, 0.1),  # k must be between 0.01 and 0.1
        (2, 10),      # block_size must be between 2 and 10
        (3, 7)        # aperture_size must be between 3 and 7 (odd values only)
    ]

    result = minimize(
        objective,
        initial_params,
        args=(tif_path, ref),
        bounds=bounds,
        method='L-BFGS-B'
    )

    # Print the results
    print("Optimization Results:")
    print(f"Optimal k: {result.x[0]}")
    print(f"Optimal block_size: {int(round(result.x[1]))}")
    print(f"Optimal aperture_size: {int(round(result.x[2]))}")
    print(f"Minimum percentage difference: {result.fun}")

    return result.x[0], result.x[1], result.x[2]