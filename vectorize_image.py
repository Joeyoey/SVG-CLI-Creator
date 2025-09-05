import cv2
import numpy as np
from svgwrite import Drawing
import argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def str2bool(v):
    if isinstance(v, bool):
        return v
    val = str(v).strip().lower()
    if val in ("yes", "true", "t", "1", "y"): 
        return True
    if val in ("no", "false", "f", "0", "n"): 
        return False
    raise argparse.ArgumentTypeError("Boolean value expected for --use_gradients")

def color_distance(c1, c2):
    return np.sqrt(np.sum((c1 - c2) ** 2))  # Euclidean in BGR

def find(parent, i):
    if parent[i] != i:
        parent[i] = find(parent, parent[i])
    return parent[i]

def union(parent, rank, sums, counts, x, y):
    xroot = find(parent, x)
    yroot = find(parent, y)
    if xroot != yroot:
        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
            sums[yroot] += sums[xroot]
            counts[yroot] += counts[xroot]
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot
            sums[xroot] += sums[yroot]
            counts[xroot] += counts[yroot]
        else:
            parent[yroot] = xroot
            sums[xroot] += sums[yroot]
            counts[xroot] += counts[yroot]
            rank[xroot] += 1

def get_region_avg(sums, counts, root):
    return sums[root] / counts[root] if counts[root] > 0 else np.array([0, 0, 0])

def smooth_polygon(points, iterations=1):
    """Chaikin corner-cutting to smooth a closed polygon.
    points: Nx2 array-like of (x,y). Returns Nx2 (grows with iterations).
    """
    if iterations <= 0 or points is None or len(points) < 3:
        return points
    pts = np.asarray(points, dtype=np.float64)
    # ensure closed by ignoring duplicate at end; we will close after smoothing
    for _ in range(int(iterations)):
        n = len(pts)
        q = []
        for i in range(n):
            p0 = pts[i]
            p1 = pts[(i + 1) % n]
            q.append(0.75 * p0 + 0.25 * p1)
            q.append(0.25 * p0 + 0.75 * p1)
        pts = np.vstack(q)
    return pts

def compute_gradients_chunk(args):
    labels_chunk, img, all_labels, height, width = args
    chunk_data = {}
    for label in labels_chunk:
        indices = np.where(all_labels == label)
        positions = np.column_stack((indices[1], indices[0]))  # x,y
        if len(positions) < 3:  # Skip tiny regions
            continue
        colors = img[indices[0], indices[1]].astype(np.float64)  # BGR
        chunk_data[label] = (positions, colors)
    return chunk_data

def compute_contours_chunk(args):
    labels_chunk, region_colors, region_data, use_gradients, img, all_labels, height, width, epsilon_ratio, min_area, hole_min_area, dilate_px, grow_px, close_iterations, smooth_iter, gradient_r2, min_gradient_area = args
    chunk_paths = []
    for label in labels_chunk:
        mask = (all_labels == label).astype(np.uint8) * 255
        if dilate_px > 0:
            k = 2 * int(dilate_px) + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=max(1, int(close_iterations)))
        # Optional post-close growth to eliminate hairline gaps between adjacent regions
        if isinstance(grow_px, int) and grow_px > 0:
            gk = 2 * int(grow_px) + 1
            gkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gk, gk))
            mask = cv2.dilate(mask, gkernel, iterations=1)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            continue
        hierarchy = hierarchy[0]
        for i, contour in enumerate(contours):
            # Only process outer contours (parent == -1)
            if hierarchy[i][3] != -1:
                continue
            if len(contour) < 3:
                continue
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            peri = cv2.arcLength(contour, True)
            epsilon = epsilon_ratio * peri
            approx = cv2.approxPolyDP(contour, epsilon, closed=True)
            if smooth_iter and smooth_iter > 0:
                base_pts = approx.reshape(-1, 2)
                sm = smooth_polygon(base_pts, iterations=smooth_iter)
                sm_int = np.round(sm).astype(int)
                path_parts = ['M ' + ' '.join(f'{p[0]},{p[1]}' for p in sm_int) + ' Z']
            else:
                path_parts = ['M ' + ' '.join(f'{p[0][0]},{p[0][1]}' for p in approx) + ' Z']
            # Append holes (children) as additional subpaths
            child = hierarchy[i][2]
            while child != -1:
                hole = contours[child]
                # Only subtract sufficiently large holes; small ones are ignored to avoid alpha pinholes
                if len(hole) >= 3 and cv2.contourArea(hole) >= hole_min_area:
                    hole_peri = cv2.arcLength(hole, True)
                    hole_eps = epsilon_ratio * hole_peri
                    hole_ap = cv2.approxPolyDP(hole, hole_eps, closed=True)
                    if smooth_iter and smooth_iter > 0:
                        hpts = hole_ap.reshape(-1, 2)
                        hsm = smooth_polygon(hpts, iterations=smooth_iter)
                        hsm_int = np.round(hsm).astype(int)
                        path_parts.append('M ' + ' '.join(f'{p[0]},{p[1]}' for p in hsm_int) + ' Z')
                    else:
                        path_parts.append('M ' + ' '.join(f'{p[0][0]},{p[0][1]}' for p in hole_ap) + ' Z')
                child = hierarchy[child][0]  # next sibling
            path_data = ' '.join(path_parts)

            fill = region_colors.get(label, 'black')
            if use_gradients and label in region_data:
                positions, colors = region_data[label]
                centroid = np.mean(positions, axis=0)
                cov = np.cov(positions.T)
                eigenvalues, eigenvectors = np.linalg.eig(cov)
                principal = eigenvectors[:, np.argmax(eigenvalues)]
                angle = np.degrees(np.arctan2(principal[1], principal[0]))
                projections = np.dot(positions - centroid, principal)
                sorted_idx = np.argsort(projections)
                proj_dist = projections[sorted_idx]
                sorted_colors = colors[sorted_idx]
                start_colors = []
                end_colors = []
                r2_vals = []
                for ch in range(3):
                    coeffs, residuals, _, _, _ = np.polyfit(proj_dist, sorted_colors[:, ch], 1, full=True)
                    slope, intercept = coeffs[0], coeffs[1]
                    y = sorted_colors[:, ch]
                    y_pred = slope * proj_dist + intercept
                    ss_res = residuals[0] if residuals.size > 0 else np.sum((y - y_pred) ** 2)
                    ss_tot = np.sum((y - y.mean()) ** 2) + 1e-8
                    r2_vals.append(1 - ss_res / ss_tot)
                    start_val = slope * proj_dist.min() + intercept
                    end_val = slope * proj_dist.max() + intercept
                    start_colors.append(int(np.clip(start_val, 0, 255)))
                    end_colors.append(int(np.clip(end_val, 0, 255)))
                r2_mean = float(np.mean(r2_vals))
                if (r2_mean >= gradient_r2) and (positions.shape[0] >= min_gradient_area):
                    grad_name = f'grad_{int(label)}'
                    start_rgb = f'rgb({start_colors[2]},{start_colors[1]},{start_colors[0]})'
                    end_rgb = f'rgb({end_colors[2]},{end_colors[1]},{end_colors[0]})'
                    dx, dy = np.cos(np.radians(angle)), np.sin(np.radians(angle))
                    fill = f'url(#{grad_name})'
                    chunk_paths.append((path_data, fill, grad_name, start_rgb, end_rgb, dx, dy))
                else:
                    chunk_paths.append((path_data, fill))
            else:
                chunk_paths.append((path_data, fill))
    return chunk_paths

def region_growing_vectorize(input_path, output_path, tolerance=10.0, blur_sigma=0.0, use_gradients=False,
                             connectivity=8, epsilon_ratio=0.003, min_area=9, dilate_px=1,
                             gradient_r2=0.55, min_gradient_area=400, hole_min_area=64, add_background=True, median_ksize=0,
                             grow_px=0, close_iterations=1, smooth_iter=0, stroke_px=0.0, stroke_round=True):
    # Load image
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError("Image not loaded")
    height, width, channels = img.shape
    total_pixels = height * width
    
    # Optional pre-smoothing
    # Median blur suppresses speckles while roughly preserving edges
    if isinstance(median_ksize, int) and median_ksize >= 3 and (median_ksize % 2 == 1):
        img = cv2.medianBlur(img, median_ksize)
    # Gaussian blur for further smoothing of color fields
    if blur_sigma > 0:
        kernel_size = 2 * int(3 * blur_sigma) + 1
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), blur_sigma)
    
    # Union-find setup
    parent = list(range(total_pixels))
    rank = [0] * total_pixels
    sums = np.zeros((total_pixels, channels), dtype=np.float64)  # Per-region color sums
    counts = np.ones(total_pixels, dtype=int)  # Per-region pixel counts
    
    # Initialize sums with each pixel's color
    flat_img = img.reshape(total_pixels, channels).astype(np.float64)
    sums[:] = flat_img
    
    # Scanline merging with running avg comparison and progress bar
    for y in tqdm(range(height), desc="Processing pixels", total=height):
        for x in range(width):
            idx = y * width + x
            curr_color = flat_img[idx]
            
            # Check left
            if x > 0:
                left_idx = y * width + (x - 1)
                left_root = find(parent, left_idx)
                left_avg = get_region_avg(sums, counts, left_root)
                if color_distance(curr_color, left_avg) < tolerance:
                    union(parent, rank, sums, counts, idx, left_idx)
            
            # Check top
            if y > 0:
                top_idx = (y - 1) * width + x
                top_root = find(parent, top_idx)
                top_avg = get_region_avg(sums, counts, top_root)
                if color_distance(curr_color, top_avg) < tolerance:
                    union(parent, rank, sums, counts, idx, top_idx)
            
            # Optional 8-connectivity diagonals
            if connectivity == 8 and y > 0:
                if x > 0:
                    tl_idx = (y - 1) * width + (x - 1)
                    tl_root = find(parent, tl_idx)
                    tl_avg = get_region_avg(sums, counts, tl_root)
                    if color_distance(curr_color, tl_avg) < tolerance:
                        union(parent, rank, sums, counts, idx, tl_idx)
                if x < width - 1:
                    tr_idx = (y - 1) * width + (x + 1)
                    tr_root = find(parent, tr_idx)
                    tr_avg = get_region_avg(sums, counts, tr_root)
                    if color_distance(curr_color, tr_avg) < tolerance:
                        union(parent, rank, sums, counts, idx, tr_idx)
    
    # Resolve parents and create label map
    labels = np.array([find(parent, i) for i in range(total_pixels)]).reshape(height, width)
    unique_labels = np.unique(labels)
    
    # Compute final avg colors (BGR to RGB)
    region_colors = {}
    for label in tqdm(unique_labels, desc="Determining colors", total=len(unique_labels)):
        avg_color = get_region_avg(sums, counts, label).astype(int)
        region_colors[label] = f'rgb({avg_color[2]},{avg_color[1]},{avg_color[0]})'
    
    # Draw larger regions first to reduce visible seams
    unique_labels = np.array(sorted(unique_labels, key=lambda l: counts[l], reverse=True))
    
    # Parallel gradient calculation with chunks
    region_data = {}
    if use_gradients:
        num_processes = min(cpu_count(), len(unique_labels) // 100)
        if num_processes > 1:
            with Pool(processes=num_processes) as pool:
                chunk_size = max(100, len(unique_labels) // (num_processes * 4))
                labels_chunks = [unique_labels[i:i + chunk_size] for i in range(0, len(unique_labels), chunk_size)]
                args = [(chunk, img, labels, height, width) for chunk in labels_chunks]
                results = list(tqdm(pool.imap(compute_gradients_chunk, args), desc="Calculating gradients", total=len(labels_chunks)))
                for result in results:
                    region_data.update(result)
        else:
            for label in tqdm(unique_labels, desc="Calculating gradients", total=len(unique_labels)):
                indices = np.where(labels == label)
                positions = np.column_stack((indices[1], indices[0]))  # x,y
                if len(positions) < 3:
                    continue
                colors = img[indices[0], indices[1]].astype(np.float64)  # BGR
                region_data[label] = (positions, colors)
    
    # Parallel contour and SVG path generation
    all_paths = []
    if len(unique_labels) > 1:  # Parallelize only if worth it
        num_processes = min(cpu_count(), len(unique_labels) // 100)
        if num_processes > 1:
            with Pool(processes=num_processes) as pool:
                chunk_size = max(100, len(unique_labels) // (num_processes * 4))
                labels_chunks = [unique_labels[i:i + chunk_size] for i in range(0, len(unique_labels), chunk_size)]
                args = [
                    (
                        chunk,
                        region_colors,
                        region_data,
                        use_gradients,
                        img,
                        labels,
                        height,
                        width,
                        epsilon_ratio,
                        min_area,
                        hole_min_area,
                        dilate_px,
                        grow_px,
                        close_iterations,
                        smooth_iter,
                        gradient_r2,
                        min_gradient_area,
                    )
                    for chunk in labels_chunks
                ]
                results = list(tqdm(pool.imap(compute_contours_chunk, args), desc="Generating contours", total=len(labels_chunks)))
                for result in results:
                    all_paths.extend(result)
        else:
            for label in tqdm(unique_labels, desc="Generating contours", total=len(unique_labels)):
                mask = (labels == label).astype(np.uint8) * 255
                if dilate_px > 0:
                    k = 2 * int(dilate_px) + 1
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=max(1, int(close_iterations)))
                if isinstance(grow_px, int) and grow_px > 0:
                    gk = 2 * int(grow_px) + 1
                    gkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gk, gk))
                    mask = cv2.dilate(mask, gkernel, iterations=1)
                contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                if hierarchy is None:
                    continue
                hierarchy = hierarchy[0]
                for i, contour in enumerate(contours):
                    if hierarchy[i][3] != -1:
                        continue
                    if len(contour) < 3:
                        continue
                    area = cv2.contourArea(contour)
                    if area < min_area:
                        continue
                    peri = cv2.arcLength(contour, True)
                    epsilon = epsilon_ratio * peri
                    approx = cv2.approxPolyDP(contour, epsilon, closed=True)
                    if smooth_iter and smooth_iter > 0:
                        base_pts = approx.reshape(-1, 2)
                        sm = smooth_polygon(base_pts, iterations=smooth_iter)
                        sm_int = np.round(sm).astype(int)
                        path_parts = ['M ' + ' '.join(f'{p[0]},{p[1]}' for p in sm_int) + ' Z']
                    else:
                        path_parts = ['M ' + ' '.join(f'{p[0][0]},{p[0][1]}' for p in approx) + ' Z']
                    child = hierarchy[i][2]
                    while child != -1:
                        hole = contours[child]
                        # Only subtract sufficiently large holes; small ones are ignored to avoid alpha pinholes
                        if len(hole) >= 3 and cv2.contourArea(hole) >= hole_min_area:
                            hole_peri = cv2.arcLength(hole, True)
                            hole_eps = epsilon_ratio * hole_peri
                            hole_ap = cv2.approxPolyDP(hole, hole_eps, closed=True)
                            path_parts.append('M ' + ' '.join(f'{p[0][0]},{p[0][1]}' for p in hole_ap) + ' Z')
                        child = hierarchy[child][0]
                    path_data = ' '.join(path_parts)
                    fill = region_colors.get(label, 'black')
                    if use_gradients and label in region_data:
                        positions, colors = region_data[label]
                        centroid = np.mean(positions, axis=0)
                        cov = np.cov(positions.T)
                        eigenvalues, eigenvectors = np.linalg.eig(cov)
                        principal = eigenvectors[:, np.argmax(eigenvalues)]
                        angle = np.degrees(np.arctan2(principal[1], principal[0]))
                        projections = np.dot(positions - centroid, principal)
                        sorted_idx = np.argsort(projections)
                        proj_dist = projections[sorted_idx]
                        sorted_colors = colors[sorted_idx]
                        start_colors = []
                        end_colors = []
                        r2_vals = []
                        for ch in range(3):  # Assuming 3 channels (BGR)
                            coeffs, residuals, _, _, _ = np.polyfit(proj_dist, sorted_colors[:, ch], 1, full=True)
                            slope, intercept = coeffs[0], coeffs[1]
                            y = sorted_colors[:, ch]
                            y_pred = slope * proj_dist + intercept
                            ss_res = residuals[0] if residuals.size > 0 else np.sum((y - y_pred) ** 2)
                            ss_tot = np.sum((y - y.mean()) ** 2) + 1e-8
                            r2_vals.append(1 - ss_res / ss_tot)
                            start_val = slope * proj_dist.min() + intercept
                            end_val = slope * proj_dist.max() + intercept
                            start_colors.append(int(np.clip(start_val, 0, 255)))
                            end_colors.append(int(np.clip(end_val, 0, 255)))
                        r2_mean = float(np.mean(r2_vals))
                        if (r2_mean >= gradient_r2) and (positions.shape[0] >= min_gradient_area):
                            grad_name = f'grad_{int(label)}'
                            start_rgb = f'rgb({start_colors[2]},{start_colors[1]},{start_colors[0]})'
                            end_rgb = f'rgb({end_colors[2]},{end_colors[1]},{end_colors[0]})'
                            dx, dy = np.cos(np.radians(angle)), np.sin(np.radians(angle))
                            all_paths.append((path_data, f'url(#{grad_name})', grad_name, start_rgb, end_rgb, dx, dy))
                        else:
                            all_paths.append((path_data, fill))
    else:
        # Single label case
        label = unique_labels[0]
        mask = (labels == label).astype(np.uint8) * 255
        if dilate_px > 0:
            k = 2 * int(dilate_px) + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=max(1, int(close_iterations)))
        if isinstance(grow_px, int) and grow_px > 0:
            gk = 2 * int(grow_px) + 1
            gkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gk, gk))
            mask = cv2.dilate(mask, gkernel, iterations=1)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            pass
        else:
            hierarchy = hierarchy[0]
            for i, contour in enumerate(contours):
                if hierarchy[i][3] != -1:
                    continue
                if len(contour) < 3:
                    continue
                area = cv2.contourArea(contour)
                if area < min_area:
                    continue
                peri = cv2.arcLength(contour, True)
                epsilon = epsilon_ratio * peri
                approx = cv2.approxPolyDP(contour, epsilon, closed=True)
                if smooth_iter and smooth_iter > 0:
                    base_pts = approx.reshape(-1, 2)
                    sm = smooth_polygon(base_pts, iterations=smooth_iter)
                    sm_int = np.round(sm).astype(int)
                    path_parts = ['M ' + ' '.join(f'{p[0]},{p[1]}' for p in sm_int) + ' Z']
                else:
                    path_parts = ['M ' + ' '.join(f'{p[0][0]},{p[0][1]}' for p in approx) + ' Z']
                child = hierarchy[i][2]
                while child != -1:
                    hole = contours[child]
                    # Only subtract sufficiently large holes; small ones are ignored to avoid alpha pinholes
                    if len(hole) >= 3 and cv2.contourArea(hole) >= hole_min_area:
                        hole_peri = cv2.arcLength(hole, True)
                        hole_eps = epsilon_ratio * hole_peri
                        hole_ap = cv2.approxPolyDP(hole, hole_eps, closed=True)
                        if smooth_iter and smooth_iter > 0:
                            hpts = hole_ap.reshape(-1, 2)
                            hsm = smooth_polygon(hpts, iterations=smooth_iter)
                            hsm_int = np.round(hsm).astype(int)
                            path_parts.append('M ' + ' '.join(f'{p[0]},{p[1]}' for p in hsm_int) + ' Z')
                        else:
                            path_parts.append('M ' + ' '.join(f'{p[0][0]},{p[0][1]}' for p in hole_ap) + ' Z')
                    child = hierarchy[child][0]
                path_data = ' '.join(path_parts)
                fill = region_colors.get(label, 'black')
                if use_gradients and label in region_data:
                    positions, colors = region_data[label]
                    centroid = np.mean(positions, axis=0)
                    cov = np.cov(positions.T)
                    eigenvalues, eigenvectors = np.linalg.eig(cov)
                    principal = eigenvectors[:, np.argmax(eigenvalues)]
                    angle = np.degrees(np.arctan2(principal[1], principal[0]))
                    projections = np.dot(positions - centroid, principal)
                    sorted_idx = np.argsort(projections)
                    proj_dist = projections[sorted_idx]
                    sorted_colors = colors[sorted_idx]
                    start_colors = []
                    end_colors = []
                    r2_vals = []
                    for ch in range(3):  # Assuming 3 channels (BGR)
                        coeffs, residuals, _, _, _ = np.polyfit(proj_dist, sorted_colors[:, ch], 1, full=True)
                        slope, intercept = coeffs[0], coeffs[1]
                        y = sorted_colors[:, ch]
                        y_pred = slope * proj_dist + intercept
                        ss_res = residuals[0] if residuals.size > 0 else np.sum((y - y_pred) ** 2)
                        ss_tot = np.sum((y - y.mean()) ** 2) + 1e-8
                        r2_vals.append(1 - ss_res / ss_tot)
                        start_val = slope * proj_dist.min() + intercept
                        end_val = slope * proj_dist.max() + intercept
                        start_colors.append(int(np.clip(start_val, 0, 255)))
                        end_colors.append(int(np.clip(end_val, 0, 255)))
                    r2_mean = float(np.mean(r2_vals))
                    if (r2_mean >= gradient_r2) and (positions.shape[0] >= min_gradient_area):
                        grad_name = f'grad_{int(label)}'
                        start_rgb = f'rgb({start_colors[2]},{start_colors[1]},{start_colors[0]})'
                        end_rgb = f'rgb({end_colors[2]},{end_colors[1]},{end_colors[0]})'
                        dx, dy = np.cos(np.radians(angle)), np.sin(np.radians(angle))
                        all_paths.append((path_data, f'url(#{grad_name})', grad_name, start_rgb, end_rgb, dx, dy))
                    else:
                        all_paths.append((path_data, fill))

    # Final SVG assembly
    dwg = Drawing(output_path, size=(width, height))
    dwg.attribs['shape-rendering'] = 'geometricPrecision'
    # Add a solid background to avoid any alpha holes from subtraction or skipped tiny regions
    if add_background:
        try:
            bg_label = int(unique_labels[0])
            bg_color = region_colors.get(bg_label, 'white')
        except Exception:
            bg_color = 'white'
        dwg.add(dwg.rect(insert=(0, 0), size=(width, height), fill=bg_color))
    grad_id = 0
    for path_data, fill, *gradient_args in all_paths:
        if len(gradient_args) == 5:  # Gradient case
            grad_name, start_rgb, end_rgb, dx, dy = gradient_args
            grad_id += 1
            # Map direction to [0,1] objectBoundingBox coordinates centered at (0.5, 0.5)
            sx, sy = 0.5 - 0.5*dx, 0.5 - 0.5*dy
            ex, ey = 0.5 + 0.5*dx, 0.5 + 0.5*dy
            grad = dwg.linearGradient((sx, sy), (ex, ey), id=grad_name)
            grad.add_stop_color(0, start_rgb)
            grad.add_stop_color(1, end_rgb)
            dwg.defs.add(grad)
            fill = f'url(#{grad_name})'
        # Create path with optional edge stroke to hide hairlines
        p = dwg.path(d=path_data, fill=fill, stroke='none', fill_rule='evenodd')
        if isinstance(stroke_px, (int, float)) and stroke_px > 0:
            p.update({
                'stroke': fill,
                'stroke-width': stroke_px,
                'stroke-linejoin': 'round' if stroke_round else 'miter',
                'stroke-linecap': 'round' if stroke_round else 'butt',
            })
        dwg.add(p)
    
    dwg.save()

def main():
    parser = argparse.ArgumentParser(description="Convert raster image to SVG with region growing.")
    parser.add_argument("input_path", type=str, help="Input image file path (e.g., input.png)")
    parser.add_argument("output_path", type=str, help="Output SVG file path (e.g., output.svg)")
    parser.add_argument("--tolerance", type=float, default=10.0, help="Color distance tolerance (default: 10.0)")
    parser.add_argument("--blur_sigma", type=float, default=0.0, help="Gaussian blur sigma (default: 0.0)")
    parser.add_argument("--use_gradients", type=str2bool, default=False, help="Enable gradients True/False (default: False)")
    parser.add_argument("--connectivity", type=int, choices=[4,8], default=8, help="Neighbor connectivity for region growing (4 or 8, default: 8)")
    parser.add_argument("--epsilon_ratio", type=float, default=0.003, help="Polygon simplification ratio of perimeter (default: 0.003)")
    parser.add_argument("--min_area", type=int, default=9, help="Skip contours smaller than this area in pixels (default: 9)")
    parser.add_argument("--dilate_px", type=int, default=1, help="Dilate/close masks by this many pixels before contouring to avoid seams (default: 1)")
    parser.add_argument("--gradient_r2", type=float, default=0.55, help="Min average R^2 across B,G,R channels to enable a linear gradient fill for a region (default: 0.55)")
    parser.add_argument("--min_gradient_area", type=int, default=400, help="Minimum number of pixels in a region to consider applying a gradient (default: 400)")
    parser.add_argument("--hole_min_area", type=int, default=64, help="Minimum hole area (px) to subtract from a region. Smaller interior holes will be ignored to avoid alpha pinholes (default: 64)")
    parser.add_argument("--add_background", type=str2bool, default=True, help="Add a solid background rectangle using the largest region color (default: True)")
    parser.add_argument("--median_ksize", type=int, default=0, help="Median blur kernel size (odd >=3). 0 disables. Helps reduce speckling (default: 0)")
    parser.add_argument("--grow_px", type=int, default=0, help="Extra dilation (px) applied to each region mask after closing to eliminate hairline gaps (default: 0)")
    parser.add_argument("--close_iterations", type=int, default=1, help="Iterations for morphological closing on each mask (default: 1)")
    parser.add_argument("--stroke_px", type=float, default=0.0, help="Optional edge stroke width in px (SVG units) using same fill to hide seams (default: 0.0)")
    parser.add_argument("--stroke_round", type=str2bool, default=True, help="Use round joins/caps for edge stroke (default: True)")
    parser.add_argument("--smooth_iter", type=int, default=0, help="Number of Chaikin smoothing iterations for contours/holes (default: 0)")
    args = parser.parse_args()

    region_growing_vectorize(
        args.input_path,
        args.output_path,
        args.tolerance,
        args.blur_sigma,
        args.use_gradients,
        args.connectivity,
        args.epsilon_ratio,
        args.min_area,
        args.dilate_px,
        args.gradient_r2,
        args.min_gradient_area,
        args.hole_min_area,
        args.add_background,
        args.median_ksize,
        args.grow_px,
        args.close_iterations,
        args.smooth_iter,
        args.stroke_px,
        args.stroke_round,
    )

if __name__ == "__main__":
    main()