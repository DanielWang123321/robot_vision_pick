"""
CV-based detection: find objects using depth segmentation.
The object is elevated above the table, so its depth is less than the table surface.
"""
import cv2
import numpy as np

from coord_transform import DEPTH_MIN_M, DEPTH_MAX_M


def depth_detect_object(depth_img, search_cx, search_cy, search_radius=500, thresh_mm=10):
    """Find elevated object nearest to search center via depth segmentation.

    Returns dict with cx, cy, width, height, angle_deg, aspect_ratio, area,
    obj_depth, table_depth — or None if detection fails.
    """
    seg = _depth_segmentation(depth_img, search_cx, search_cy, search_radius, thresh_mm)
    if seg is None:
        return None
    contours, obj_mask, table_d, x1, y1 = seg

    # Select contour closest to search center
    local_cx = search_cx - x1
    local_cy = search_cy - y1
    best = None
    best_dist = float('inf')
    for cnt in contours:
        if cv2.contourArea(cnt) < 200:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        dist = np.sqrt((cx - local_cx)**2 + (cy - local_cy)**2)
        if dist < best_dist:
            best_dist = dist
            best = cnt

    if best is None:
        return None

    return _contour_to_result(best, depth_img, obj_mask, table_d, x1, y1)



def extract_color_ref(color_img, cx, cy, radius=60):
    """Extract HSV histogram from a patch around (cx, cy) as color reference."""
    h, w = color_img.shape[:2]
    x1, y1 = max(0, cx - radius), max(0, cy - radius)
    x2, y2 = min(w, cx + radius), min(h, cy + radius)
    patch = color_img[y1:y2, x1:x2]
    if patch.size == 0:
        return None
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def extract_color_ref_bbox(color_img, x1, y1, x2, y2, margin=10):
    """Extract HSV histogram from bbox region with inward margin to avoid background."""
    h, w = color_img.shape[:2]
    x1 = max(0, min(w, x1 + margin))
    y1 = max(0, min(h, y1 + margin))
    x2 = max(0, min(w, x2 - margin))
    y2 = max(0, min(h, y2 - margin))
    if x2 <= x1 or y2 <= y1:
        return None
    patch = color_img[y1:y2, x1:x2]
    if patch.size == 0:
        return None
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def _depth_segmentation(depth_img, search_cx, search_cy, search_radius, thresh_mm):
    """Common depth segmentation: returns (contours, obj_mask, table_depth, x1, y1) or None."""
    h, w = depth_img.shape
    x1, y1 = max(0, search_cx - search_radius), max(0, search_cy - search_radius)
    x2, y2 = min(w, search_cx + search_radius), min(h, search_cy + search_radius)
    crop = depth_img[y1:y2, x1:x2].copy()

    valid = ~np.isnan(crop) & (crop > DEPTH_MIN_M) & (crop < DEPTH_MAX_M)
    if np.sum(valid) < 100:
        return None

    bw = min(30, search_radius // 4)
    border = np.zeros_like(valid)
    border[:bw, :] = border[-bw:, :] = border[:, :bw] = border[:, -bw:] = True
    border_vals = crop[border & valid]
    if len(border_vals) < 20:
        return None

    table_d = np.median(border_vals)
    obj = valid & (crop < table_d - thresh_mm / 1000.0)
    obj = obj.astype(np.uint8) * 255
    if np.sum(obj > 0) < 100:
        return None

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    obj = cv2.morphologyEx(obj, cv2.MORPH_CLOSE, kernel, iterations=2)
    obj = cv2.morphologyEx(obj, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    return contours, obj, table_d, x1, y1


def _contour_to_result(cnt, depth_img, obj_mask, table_d, x1, y1):
    """Convert a selected contour to the standard detection result dict."""
    rect = cv2.minAreaRect(cnt)
    (rcx, rcy), (rw, rh), angle = rect
    if rw < rh:
        rw, rh = rh, rw
        angle += 90

    M = cv2.moments(cnt)
    if M["m00"] > 0:
        mcx = M["m10"] / M["m00"] + x1
        mcy = M["m01"] / M["m00"] + y1
    else:
        mcx, mcy = rcx + x1, rcy + y1

    best_mask = np.zeros_like(obj_mask)
    cv2.drawContours(best_mask, [cnt], -1, 255, -1)
    obj_region = depth_img[y1:y1 + obj_mask.shape[0], x1:x1 + obj_mask.shape[1]].copy()
    obj_pixels = obj_region[best_mask > 0]
    obj_depth = float(np.median(obj_pixels[~np.isnan(obj_pixels)])) if np.sum(~np.isnan(obj_pixels)) > 0 else None

    return {
        'cx': int(mcx), 'cy': int(mcy),
        'width': rw, 'height': rh,
        'angle_deg': angle,
        'aspect_ratio': rw / max(rh, 1),
        'area': cv2.contourArea(cnt),
        'obj_depth': obj_depth,
        'table_depth': table_d,
    }


def depth_detect_with_color(depth_img, color_img, search_cx, search_cy,
                            ref_hist, search_radius=500, thresh_mm=10):
    """Find elevated object that best matches reference color histogram.

    Uses depth segmentation to find all candidate objects, then selects
    the one whose color best matches ref_hist (HSV histogram from initial detection).
    """
    seg = _depth_segmentation(depth_img, search_cx, search_cy, search_radius, thresh_mm)
    if seg is None:
        return None
    contours, obj_mask, table_d, x1, y1 = seg

    x2 = x1 + obj_mask.shape[1]
    y2 = y1 + obj_mask.shape[0]
    crop_color = color_img[y1:y2, x1:x2]
    hsv_crop = cv2.cvtColor(crop_color, cv2.COLOR_BGR2HSV)

    best = None
    best_score = -1
    for cnt in contours:
        if cv2.contourArea(cnt) < 200:
            continue
        mask = np.zeros(obj_mask.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        hist = cv2.calcHist([hsv_crop], [0, 1], mask, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        score = cv2.compareHist(ref_hist, hist, cv2.HISTCMP_CORREL)
        if score > best_score:
            best_score = score
            best = cnt

    if best is None:
        return None

    result = _contour_to_result(best, depth_img, obj_mask, table_d, x1, y1)
    result['color_score'] = best_score
    return result


def compute_gripper_yaw(obj_angle_deg):
    """Convert object long-axis angle in image to gripper yaw in robot base frame.

    At detect position (roll=180, pitch=0, yaw=0) with camera yaw ~91°:
        Image +X ≈ Base -Y,  Image +Y ≈ Base -X
        Image angle α → base long-axis angle = -(α + 90°)

    At yaw=0, gripper fingers open along Base Y (90° from X).
    We want fingers PERPENDICULAR to long axis (= along short axis) to grip the narrow side.
    finger_direction = 90° + yaw = long_axis + 90°  →  yaw = long_axis.
    Normalized to [-90, 90] using gripper's 180° symmetry.
    """
    base_long_axis = -(obj_angle_deg + 90.0)
    gripper_yaw = base_long_axis

    while gripper_yaw > 90:
        gripper_yaw -= 180
    while gripper_yaw < -90:
        gripper_yaw += 180

    return gripper_yaw


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 4:
        print("Usage: python cv_refine.py <depth_npy> <search_cx> <search_cy>")
        sys.exit(1)

    depth = np.load(sys.argv[1])
    cpx, cpy = int(sys.argv[2]), int(sys.argv[3])

    print(f"Depth: {depth.shape[1]}x{depth.shape[0]}")
    print(f"Search center: ({cpx}, {cpy})")

    result = depth_detect_object(depth, cpx, cpy, search_radius=300)
    if result:
        print(f"Detected: center=({result['cx']}, {result['cy']}), "
              f"size={result['width']:.0f}x{result['height']:.0f}, "
              f"aspect={result['aspect_ratio']:.2f}, angle={result['angle_deg']:.1f}")
    else:
        print("Detection failed")
