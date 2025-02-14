import laspy
import numpy as np
import os
import json
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from scipy.spatial import distance
from tqdm import tqdm
from scipy.spatial import cKDTree
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from scipy.stats import kstest
from scipy.spatial import KDTree
import math
import csv
import shutil
import matplotlib.pyplot as plt


def nearest_points(p1, p2, point_cloud, num_points=300, gutter=False):
    if gutter:
        num_points = 300
    line_direction = p2 - p1
    line_length = np.linalg.norm(line_direction)
    unit_line_direction = line_direction / line_length
    vectors_to_points = point_cloud - p1
    dot_products = np.dot(vectors_to_points, unit_line_direction)
    projected_points = p1 + np.clip(dot_products[:, np.newaxis], 0, line_length) * unit_line_direction
    distances = np.linalg.norm(projected_points - point_cloud, axis=1)
    nearest_point_indices = np.argsort(distances)[:num_points]

    return point_cloud[nearest_point_indices]

def divide_line_into_segments(p_start, p_end, num_segments=10):

    return np.array([p_start + (p_end - p_start) * t for t in np.linspace(0, 1, num_segments + 1)])

def divide_line_into_segments_100(p_start, p_end, num_segments=100):

    return np.array([p_start + (p_end - p_start) * t for t in np.linspace(0, 1, num_segments + 1)])

def calculate_slope_3d(point_1, point_2):
    delta_x = point_2[0] - point_1[0]
    delta_y = point_2[1] - point_1[1]
    delta_z = point_2[2] - point_1[2]

    if delta_x == 0 and delta_y == 0:
        return float('inf')
    else:
        return round(delta_z / np.sqrt(delta_x**2 + delta_y**2)*100, 1)
    
def calculate_slope_pca(pca_model):
    direction_vector = pca_model.components_[0]
    dx, dy, dz = direction_vector
    if dx == 0 and dy == 0:
        slope = float('inf')
    else:
        slope = dz / np.sqrt(dx**2 + dy**2) 

    slope_percentage = abs(round(slope * 100, 1))

    return slope_percentage

def update_mask(original_mask, new_mask):
    indices = np.where(original_mask)[0]
    updated_mask = np.copy(original_mask)
    updated_mask[indices] = updated_mask[indices] & new_mask
    return updated_mask

def fit_line_with_pca_threshold(X, y, threshold=62.5, max_iterations=10):
    data = np.hstack((X, y[:, np.newaxis]))
    cul_mask = np.ones(len(data), dtype=bool)
    
    iteration = 0
    while iteration < max_iterations:
        pca = PCA(n_components=1)
        pca.fit(data)
        projected = pca.transform(data)
        reconstructed = pca.inverse_transform(projected)
        residuals_xyz = np.sqrt(np.sum((data - reconstructed)**2, axis=1))

        mask = residuals_xyz <= threshold
        refined_data = data[mask]

        # print(f"Iteration {iteration+1}: Original dataset size: {len(data)}, Refined dataset size: {len(refined_data)}")
        
        if len(refined_data) < len(data):
            data = refined_data
        else:
            break
        
        iteration += 1

        cul_mask = update_mask(cul_mask, mask)
    
    return pca, cul_mask

def fit_line_with_pca_threshold_z(X, y, cul_mask, z_threshold=62.5, max_iterations=10):
    data = np.hstack((X, y[:, np.newaxis]))
    data = data[cul_mask]

    iteration = 0
    while iteration < max_iterations:
        pca = PCA(n_components=1)
        pca.fit(data)
        projected = pca.transform(data)
        reconstructed = pca.inverse_transform(projected)
        residuals_z = np.abs(data[:, 2] - reconstructed[:, 2])

        mask = residuals_z <= z_threshold
        refined_data = data[mask]

        # print(f"z Iteration {iteration+1}: Original dataset size: {len(data)}, Refined dataset size: {len(refined_data)}")
        
        if len(refined_data) < len(data):
            data = refined_data
        else:
            break
        
        iteration += 1

        cul_mask = update_mask(cul_mask, mask)
    
    return pca, cul_mask

def project_points_to_line(points, line_point, line_direction):
    line_direction_normalized = line_direction / np.linalg.norm(line_direction)
    projections = line_point + np.dot(points - line_point, line_direction_normalized[:, np.newaxis]) * line_direction_normalized

    return projections

def find_gaps_in_projections(projections, gap_threshold=0.25):
    holes = []
    distances = np.linalg.norm(np.diff(projections, axis=0), axis=1)
    hole_indices = np.where(distances > gap_threshold)[0]
    
    # print(sorted(distances)[::-1])
    
    for idx in hole_indices:
        holes.append((idx, idx + 1))

    return holes

def find_intersection(p1, d1, p2, d2):
    t = np.linalg.solve(np.array([d1, -d2]).T, p2 - p1)
    intersection = p1 + d1 * t[0]
    
    return intersection

def find_z_for_xy_on_line(x, y, line_point, direction_vector):
    px, py, pz = line_point
    dx, dy, dz = direction_vector

    if dx != 0:
        t = (x - px) / dx
    elif dy != 0:
        t = (y - py) / dy
    else:
        raise ValueError("Line is vertical")

    z = pz + t * dz

    return z

def calculate_3d_distance(point1, point2):
    x1, y1, z1 = point1
    x2, y2, z2 = point2

    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)*12

    return round(distance, 1)

def yes_no(tf):
    if tf:
        return 'Yes'
    else:
        return 'No'
    
def normalize_points(corner_points, point_cloud_mean):
    return corner_points - point_cloud_mean

def normalize_point_cloud(point_cloud, point_cloud_mean):
    return point_cloud - point_cloud_mean

def crop_point_cloud(corner_points, point_cloud, padding):
    p1, p2, p3, p4 = corner_points
    min_x = min(p1[0], p2[0], p3[0], p4[0])
    max_x = max(p1[0], p2[0], p3[0], p4[0])
    min_y = min(p1[1], p2[1], p3[1], p4[1])
    max_y = max(p1[1], p2[1], p3[1], p4[1])

    filtered_points = point_cloud[(point_cloud[:, 0] >= min_x - padding) & (point_cloud[:, 0] <= max_x + padding) & 
                                (point_cloud[:, 1] >= min_y - padding) & (point_cloud[:, 1] <= max_y + padding)]

    sorted_points = filtered_points[np.argsort(filtered_points[:, 2])]

    return sorted_points

def solve_for_t(p1, p2, A, B, C, D):
    t = (D - A*p1[0] - B*p1[1] - C*p1[2]) / (A*(p2[0]-p1[0]) + B*(p2[1]-p1[1]) + C*(p2[2]-p1[2]))
    return t

def get_perpendicular(p34_interest, p1, p2, p3, p4):
    normal_vector = p4 - p3
    A, B, C = normal_vector
    D = np.dot(normal_vector, p34_interest)

    t = solve_for_t(p1, p2, A, B, C, D)
    intersection_point = p1 + t * (p2 - p1)
    direction_vector = intersection_point - p34_interest
    return intersection_point, direction_vector


def adjusted_points(p34_5, plane, p2close):
    p1, p2, p3, p4 = plane
    p34_5_intersection_point, p34_5_direction_vector = get_perpendicular(p34_5, p1, p2, p3, p4)
    ## p2 is closer
    if p2close:
        new_intersection_point, new_direction_vector = get_perpendicular(p2, p4, p1, p34_5_intersection_point, p34_5)
        return [new_intersection_point, p2, p3, p4]

    else:
        new_intersection_point, new_direction_vector = get_perpendicular(p1, p3, p2, p34_5_intersection_point, p34_5)
        return [p1, new_intersection_point, p3, p4]
    

def refine_corner_points(p1, p2, p3, p4):
    p34 = p3-p4

    p34_5 = (p3 + p4) / 2
    p34_1 = (1 * p3 + 9 * p4) / 10
    p34_9 = (9 * p3 + 1 * p4) / 10

    ramp_points = [p1, p2, p3, p4]

    p1_proj = project_points_to_line(p1, p34_5, p34)
    p2_proj = project_points_to_line(p2, p34_5, p34)
    p2close = calculate_3d_distance(p1, p1_proj) > calculate_3d_distance(p2, p2_proj)
    # if p2close:
    #     print('p2 is closer')
    # else:
    #     print('p1 is closer')
    adj_ramp_points = adjusted_points(p34_5, ramp_points, p2close)
    return adj_ramp_points


def find_perpendicular_top_line(p1, p2, p3, p4, point_cloud, search_radius=0.5):
    direction_vector = (p2[:2] - p1[:2]) / np.linalg.norm(p2[:2] - p1[:2])
    normal_vector = np.array([-direction_vector[1], direction_vector[0]])  # Perpendicular vector

    bottom_direction = (p4[:2] - p3[:2]) / np.linalg.norm(p4[:2] - p3[:2])

    if np.dot(normal_vector, bottom_direction) > 0:
        normal_vector = -normal_vector

    candidate_top_p1 = point_cloud[np.linalg.norm(point_cloud[:, :2] - (p1[:2] + search_radius * normal_vector), axis=1) <= search_radius]
    candidate_top_p2 = point_cloud[np.linalg.norm(point_cloud[:, :2] - (p2[:2] + search_radius * normal_vector), axis=1) <= search_radius]

    if len(candidate_top_p1) == 0 or len(candidate_top_p2) == 0:
        raise ValueError("No points found near the top search area for p1 and p2.")

    top_p1 = project_point_to_perpendicular(candidate_top_p1, p1, normal_vector)
    top_p2 = project_point_to_perpendicular(candidate_top_p2, p2, normal_vector)

    direction_p1_p4 = (p4[:2] - p1[:2]) / np.linalg.norm(p4[:2] - p1[:2])
    direction_p1_new = (top_p1[:2] - p1[:2]) / np.linalg.norm(top_p1[:2] - p1[:2])

    if np.dot(direction_p1_p4, direction_p1_new) > 0:
        displacement = top_p1[:2] - p1[:2]
        top_p1[:2] = p1[:2] - displacement

    direction_p2_p3 = (p3[:2] - p2[:2]) / np.linalg.norm(p3[:2] - p2[:2])
    direction_p2_new = (top_p2[:2] - p2[:2]) / np.linalg.norm(top_p2[:2] - p2[:2])

    if np.dot(direction_p2_p3, direction_p2_new) > 0:
        displacement = top_p2[:2] - p2[:2]
        top_p2[:2] = p2[:2] - displacement

    fitted_line = np.array([top_p1, top_p2])
    return fitted_line

def find_perpendicular_top_line_gutter(p1, p2, p3, p4, point_cloud, search_radius):
    direction_vector = (p2[:2] - p1[:2]) / np.linalg.norm(p2[:2] - p1[:2])
    normal_vector = np.array([-direction_vector[1], direction_vector[0]])  # Perpendicular vector
    bottom_direction = (p4[:2] - p3[:2]) / np.linalg.norm(p4[:2] - p3[:2])

    if np.dot(normal_vector, bottom_direction) < 0:
        normal_vector = -normal_vector

    candidate_top_p1 = point_cloud[np.linalg.norm(point_cloud[:, :2] - (p1[:2] + search_radius * normal_vector), axis=1) <= search_radius]
    candidate_top_p2 = point_cloud[np.linalg.norm(point_cloud[:, :2] - (p2[:2] + search_radius * normal_vector), axis=1) <= search_radius]

    if len(candidate_top_p1) == 0 or len(candidate_top_p2) == 0:
        raise ValueError("No points found near the top search area for p1 and p2.")

    top_p1 = candidate_top_p1.mean(axis=0)
    top_p2 = candidate_top_p2.mean(axis=0)

    direction_p1_p4 = (p4[:2] - p1[:2]) / np.linalg.norm(p4[:2] - p1[:2])
    direction_p1_new = (top_p1[:2] - p1[:2]) / np.linalg.norm(top_p1[:2] - p1[:2])

    if np.dot(direction_p1_p4, direction_p1_new) < 0:
        displacement = top_p1[:2] - p1[:2]
        top_p1[:2] = p1[:2] - displacement

    direction_p2_p3 = (p3[:2] - p2[:2]) / np.linalg.norm(p3[:2] - p2[:2])
    direction_p2_new = (top_p2[:2] - p2[:2]) / np.linalg.norm(top_p2[:2] - p2[:2])

    if np.dot(direction_p2_p3, direction_p2_new) < 0:
        displacement = top_p2[:2] - p2[:2]
        top_p2[:2] = p2[:2] - displacement

    fitted_line = np.array([top_p1, top_p2])
    return fitted_line


def project_point_to_perpendicular(candidate_points, base_point, normal_vector):
    relative_points = candidate_points[:, :2] - base_point[:2]
    projections = (relative_points @ normal_vector)[:, None] * normal_vector
    projected_points = base_point[:2] + projections
    return np.column_stack((projected_points, candidate_points[:, 2])).mean(axis=0) 

def project_fitted_line_to_z(fitted_line, point_cloud, z_direction='-z', search_radius=0.2):
    top_p1, top_p2 = fitted_line

    z_offset = 1.0 if z_direction == '+z' else -1.0

    line_p1_projected = np.array([top_p1[0], top_p1[1], top_p1[2] + z_offset * 100])  
    line_p2_projected = np.array([top_p2[0], top_p2[1], top_p2[2] + z_offset * 100])

    points_near_line_p1 = point_cloud[np.abs(point_cloud[:, 0] - top_p1[0]) <= search_radius]
    points_near_line_p1 = points_near_line_p1[np.abs(points_near_line_p1[:, 1] - top_p1[1]) <= search_radius]

    points_near_line_p2 = point_cloud[np.abs(point_cloud[:, 0] - top_p2[0]) <= search_radius]
    points_near_line_p2 = points_near_line_p2[np.abs(points_near_line_p2[:, 1] - top_p2[1]) <= search_radius]

    if len(points_near_line_p1) == 0:
        distances = np.linalg.norm(point_cloud[:, :2] - top_p1[:2], axis=1)
        intersection_p1 = point_cloud[np.argmin(distances)]
    else:
        intersection_p1 = points_near_line_p1[np.argmin(np.abs(points_near_line_p1[:, 2] - top_p1[2]))]

    if len(points_near_line_p2) == 0:
        distances = np.linalg.norm(point_cloud[:, :2] - top_p2[:2], axis=1)
        intersection_p2 = point_cloud[np.argmin(distances)]
    else:
        intersection_p2 = points_near_line_p2[np.argmin(np.abs(points_near_line_p2[:, 2] - top_p2[2]))]

    intersection_points = np.array([intersection_p1, intersection_p2])

    return intersection_points


def create_new_top_landing_line(intersection_points):
    new_top_p1, new_top_p2 = intersection_points
    new_top_landing_line = np.array([new_top_p1, new_top_p2])
    return new_top_landing_line

def create_colored_area(p1, p2, new_top_p1, new_top_p2, color):
    vertices = np.array([
        p1,        # Bottom left
        p2,        # Bottom right
        new_top_p2,  # Top right
        new_top_p1   # Top left
    ])
    i = [0, 1, 2, 3] 

    return go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        color=color, 
        opacity=0.5,
        alphahull=0,
        name='Landing Area'
    )

def create_colored_area_flare(p1, p2, p3, color):
    vertices = np.array([
        p1,  # First point
        p2,  # Second point
        p3   # Third point
    ])

    return go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=[0],
        j=[1],
        k=[2],  
        color=color,
        opacity=0.5,
        name='Triangle Area'
    )


def find_downward_intersection(p1, p2, p3, p4, point_cloud, max_extension=1000, xy_tolerance=0.2):
    direction_p1_p4 = (p4 - p1) / np.linalg.norm(p4 - p1)  # Normalize
    direction_p2_p3 = (p3 - p2) / np.linalg.norm(p3 - p2)  # Normalize
    extended_p4 = p4 + np.array([0, 0, -max_extension])
    extended_p3 = p3 + np.array([0, 0, -max_extension])
    candidates_p1_p4 = []
    candidates_p2_p3 = []

    for point in point_cloud:
        t1 = np.dot(point[:2] - p1[:2], direction_p1_p4[:2])
        projected_point1 = p1[:2] + t1 * direction_p1_p4[:2]

        t2 = np.dot(point[:2] - p2[:2], direction_p2_p3[:2])
        projected_point2 = p2[:2] + t2 * direction_p2_p3[:2]

        if np.linalg.norm(projected_point1 - point[:2]) <= xy_tolerance:
            candidates_p1_p4.append(point)

        if np.linalg.norm(projected_point2 - point[:2]) <= xy_tolerance:
            candidates_p2_p3.append(point)

    if len(candidates_p1_p4) == 0 or len(candidates_p2_p3) == 0:
        raise ValueError("No intersecting points found near the extended lines.")

    candidates_p1_p4 = np.array(candidates_p1_p4)
    candidates_p2_p3 = np.array(candidates_p2_p3)

    intersect_point1 = candidates_p1_p4[np.argmin(candidates_p1_p4[:, 2])]
    intersect_point2 = candidates_p2_p3[np.argmin(candidates_p2_p3[:, 2])]

    return intersect_point1, intersect_point2


def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):  # Convert numpy integers
        return int(obj)
    elif isinstance(obj, np.floating):  # Convert numpy floats
        return float(obj)
    elif isinstance(obj, np.ndarray):  # Convert numpy arrays
        return obj.tolist()
    else:
        return obj

def normalize_points(points, offsets, scales, scales_old, offsets_old):
    return ((points - offsets) / scales) * scales_old + offsets_old


def read_point_cloud(las_file_path):
    las_file_name = las_file_path.split('/')[-1].replace('.las', '')
    in_file = laspy.read(las_file_path)
    points = np.vstack((in_file.x, in_file.y, in_file.z)).transpose()

    offsets_n_scales = np.load(f"pointclouds/OffsetsScales/{'_'.join(las_file_name.split('_')[:-1])}.npy")
    offsets_old, scales_old = offsets_n_scales[:3], offsets_n_scales[3:]

    points = normalize_points(points, in_file.header.offsets, in_file.header.scales, scales_old, offsets_old)
    classification = in_file.classification

    return points, classification


def project_to_image(points, classifications, corner_points, img_size=256, corner_size=3):
    """Project both point cloud and corner points to a top-down image."""
    x = points[:, 0]
    y = points[:, 1]

    x_min, x_max = min(x.min(), corner_points[:, 0].min()), max(x.max(), corner_points[:, 0].max())
    y_min, y_max = min(y.min(), corner_points[:, 1].min()), max(y.max(), corner_points[:, 1].max())

    x_normalized = ((x - x_min) / (x_max - x_min)) * (img_size - 1)
    y_normalized = ((y - y_min) / (y_max - y_min)) * (img_size - 1)

    image = np.zeros((img_size, img_size, 3), dtype=np.uint8)

    colors = {
        1: [0, 255, 0],  # Green for main ramp
        2: [0, 0, 255],  # Blue for left flare
        3: [255, 0, 0],  # Red for right flare
    }

    for i, classification in enumerate(classifications):
        color = colors.get(classification, [128, 128, 128])  
        x_idx = int(y_normalized[i])  
        y_idx = int(x_normalized[i])
        image[x_idx, y_idx] = color

    # Project corner points in orange and make them larger
    corners_x = corner_points[:, 0]
    corners_y = corner_points[:, 1]
    corners_x_normalized = ((corners_x - x_min) / (x_max - x_min)) * (img_size - 1)
    corners_y_normalized = ((corners_y - y_min) / (y_max - y_min)) * (img_size - 1)

    for i in range(len(corners_x_normalized)):
        x_idx = int(corners_y_normalized[i])  # Swap axes for image coordinates
        y_idx = int(corners_x_normalized[i])
        # Draw a larger orange point (as a square around the center point)
        for dx in range(-corner_size, corner_size + 1):
            for dy in range(-corner_size, corner_size + 1):
                nx, ny = x_idx + dx, y_idx + dy
                if 0 <= nx < img_size and 0 <= ny < img_size:  # Ensure within bounds
                    image[nx, ny] = [255, 165, 0]  # Orange for corner points

    return image


def save_image(image, output_path):
    """Saves the generated image."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()



def calculate_class_counts_near_corners(points, classifications, corner_points, radius=1.0):
    kd_tree = KDTree(points[:, :2])  # Use only x, y for top-down projection
    corner_class_counts = {}

    for i, corner in enumerate(corner_points):
        # Find neighbors within the radius
        neighbor_idxs = kd_tree.query_ball_point(corner[:2], r=radius)
        
        # Get classifications of neighbors
        neighbor_classes = classifications[neighbor_idxs]
        count_class1 = np.sum(neighbor_classes == 1)
        count_class2 = np.sum(neighbor_classes == 2)
        count_class3 = np.sum(neighbor_classes == 3)

        # Store counts for this corner
        corner_class_counts[f"corner_{i+1}"] = {
            "class1_neighbors": count_class1,
            "class2_neighbors": count_class2,
            "class3_neighbors": count_class3
        }

    return corner_class_counts

def calculate_angle(p3, fl, p4, fr):
    vector1 = fl - p3
    vector2 = fr - p4

    # Compute dot product and norms
    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)

    # Avoid division by zero
    if norm1 == 0 or norm2 == 0:
        return None

    # Compute angle in radians and convert to degrees
    angle_radians = math.acos(dot_product / (norm1 * norm2))
    angle_degrees = math.degrees(angle_radians)
    return angle_degrees


def analyze_corners(base_path, json_file_path, output_dir, img_size=1024, radius=1.0):
    os.makedirs(f'{target_dir}/top_down', exist_ok=True)
    # Load corner data
    with open(json_file_path, 'r') as json_file:
        corners_data = json.load(json_file)

    output_data = {}

    # Process each .las file
    for las_file in tqdm(os.listdir(base_path)):
        if not las_file.endswith('.las'):
            continue

        las_file_path = os.path.join(base_path, las_file)
        try:
            points, classifications = read_point_cloud(las_file_path)

            if las_file in corners_data:
                corner_points = np.array(corners_data[las_file]['corners'])

                corner_class_counts = calculate_class_counts_near_corners(
                    points, classifications, corner_points, radius
                )

                # Extract individual corners
                p1, p2, p3, p4, fr, fl = corner_points

                angle_p3_fl_p4_fr = calculate_angle(corner_points[2], corner_points[4], corner_points[3], corner_points[5])



                angle_c2c3_c3c5 = calculate_angle(corner_points[1], corner_points[2], corner_points[2], corner_points[4])
                angle_c1c4_c4c6 = calculate_angle(corner_points[0], corner_points[3], corner_points[3], corner_points[5])
                angle_c2c3_c3c4 = calculate_angle(corner_points[1], corner_points[2], corner_points[2], corner_points[3])
                angle_c1c4_c4c3 = calculate_angle(corner_points[0], corner_points[3], corner_points[3], corner_points[2])
                angle_c1c2_c3c4 = calculate_angle(corner_points[0], corner_points[1], corner_points[2], corner_points[3])

                # Save data for this file
                output_data[las_file] = {
                    "corner_class_counts": corner_class_counts,
                    "angles": {
                        "p3_fl_p4_fr": angle_p3_fl_p4_fr,
                        "c2c3_c3c5": angle_c2c3_c3c5,
                        "c1c4_c4c6": angle_c1c4_c4c6,
                        "c2c3_c3c4": angle_c2c3_c3c4,
                        "c1c4_c4c3": angle_c1c4_c4c3,
                        "c1c2_c3c4": angle_c1c2_c3c4
                    }
                }

            else:
                print(f"No corner points found for {las_file}. Skipping.")
                continue

            # Save the image (optional, already in original script)
            image = project_to_image(points, classifications, corner_points, img_size)
            output_path = os.path.join(f'{output_dir}/top_down', las_file.replace('.las', '.png'))
            save_image(image, output_path)

        except Exception as e:
            print(f"Error processing {las_file}: {e}")

    # Save the output data to a JSON file
    with open(os.path.join(output_dir, "quality_analysis.json"), 'w') as json_file:
        json.dump(convert_numpy_types(output_data), json_file, indent=4)


def find_std_deviation(file_names, file_name, data):
    for i, fname in enumerate(file_names):
        if fname == file_name:
            
            std_away = (data[i] - np.mean(data)) / np.std(data)
            print(f"File: {file_name}, Value: {data[i]}, Mean: {np.mean(data)}, Std Deviation: {np.std(data)}, Std Deviations Away: {std_away}")
            return std_away
        
    return None

    

def remove_outliers(title, data, original_data, file_names, threshold=3):
    """Remove data points that are more than 'threshold' standard deviations from the mean. Print filenames of outliers."""
    if len(data) == 0:
        print("No data to process.")
        return data, []

    data = np.array(data)
    mean = np.mean(data)
    std = np.std(data)

    if title == "Angle Between Corner 1->2 and Corner 3->4":
        outliers = [(value, file_names[i]) for i, value in enumerate(original_data) if value <= 170]
        filtered_data = data[(data > 170)]
        for value, fname in outliers:
            print(f"Outlier Value: {value}, File: {fname}")

    else:
        # Define bounds
        lower_bound = mean - threshold * std
        upper_bound = mean + threshold * std

        # Identify outliers
        outliers = [(value, file_names[i]) for i, value in enumerate(original_data) if value < lower_bound or value > upper_bound]
        outlier_values = [value for value, _ in outliers]

        # Filter data
        filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
        print(f"Removed {len(data) - len(filtered_data)} outliers.")
        for value, fname in outliers:
            print(f"Outlier Value: {value}, File: {fname}")

    return filtered_data, outliers

def load_json(file_path):
    """Load the JSON data from the file."""
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def extract_distributions(json_data):
    """Extract distributions for angles, class 3 at corner 5, class 2 at corner 6, and new angles."""
    angles = []
    class3_corner5 = []
    class2_corner6 = []

    # New angles
    angle_c2c3_c3c5 = []
    angle_c1c4_c4c6 = []
    angle_c2c3_c3c4 = []
    angle_c1c4_c4c3 = []
    angle_c1c2_c3c4 = []

    file_names = []

    for fname, file_data in json_data.items():
        file_names.append(fname)
        # Extract the main angle
        if 'angles' in file_data:
            if 'p3_fl_p4_fr' in file_data['angles']:
                angles.append(file_data['angles']['p3_fl_p4_fr'])

            # Extract new angles
            if 'c2c3_c3c5' in file_data['angles']:
                angle_c2c3_c3c5.append(file_data['angles']['c2c3_c3c5'])
            if 'c1c4_c4c6' in file_data['angles']:
                angle_c1c4_c4c6.append(file_data['angles']['c1c4_c4c6'])
            if 'c2c3_c3c4' in file_data['angles']:
                angle_c2c3_c3c4.append(file_data['angles']['c2c3_c3c4'])
            if 'c1c4_c4c3' in file_data['angles']:
                angle_c1c4_c4c3.append(file_data['angles']['c1c4_c4c3'])
            if 'c1c2_c3c4' in file_data['angles']:
                angle_c1c2_c3c4.append(file_data['angles']['c1c2_c3c4'])
                print(f"File: {fname}, Angle: {file_data['angles']['c1c2_c3c4']}")

        # Extract class 3 for corner 5 and class 2 for corner 6
        if 'corner_class_counts' in file_data:
            corner_counts = file_data['corner_class_counts']

            if 'corner_5' in corner_counts:
                class3_corner5.append(corner_counts['corner_5'].get('class3_neighbors', 0))
            
            if 'corner_6' in corner_counts:
                class2_corner6.append(corner_counts['corner_6'].get('class2_neighbors', 0))

    return (
        angles,
        class3_corner5,
        class2_corner6,
        angle_c2c3_c3c5,
        angle_c1c4_c4c6,
        angle_c2c3_c3c4,
        angle_c1c4_c4c3,
        angle_c1c2_c3c4,
        file_names
    )

def plot_distribution(data, title, xlabel, ylabel, bins=15):
    """Plot a histogram for the given data, ignoring None values."""
    # Filter out None values
    data = [d for d in data if d is not None]

    if not data:  # If the filtered data is empty, skip plotting
        print(f"No valid data to plot for {title}.")
        return

    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=bins, alpha=0.7, color='blue', edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()

def filter_outliers(target_dir, base_path):
    # Path to the JSON file
    json_file_path = f"{target_dir}/quality_analysis.json"
    # Load JSON data
    json_data = load_json(json_file_path)

    # Extract distributions
    (
        angles,
        class3_corner5,
        class2_corner6,
        angle_c2c3_c3c5,
        angle_c1c4_c4c6,
        angle_c2c3_c3c4,
        angle_c1c4_c4c3,
        angle_c1c2_c3c4,
        file_names
    ) = extract_distributions(json_data)

    datasets = {
        "Main Angle (p3->fl and p4->fr)": angles,
        "Angle Between Corner 2->3 and Corner 3->5": angle_c2c3_c3c5,
        "Angle Between Corner 1->4 and Corner 4->6": angle_c1c4_c4c6,
        "Angle Between Corner 2->3 and Corner 3->4": angle_c2c3_c3c4,
        "Angle Between Corner 1->4 and Corner 4->3": angle_c1c4_c4c3,
        "Angle Between Corner 1->2 and Corner 3->4": angle_c1c2_c3c4
    }

    all_outliers = []

    for title, data in datasets.items():
        print(f"\nProcessing {title}:")
        
        # Remove outliers
        filtered_data, outliers = remove_outliers(title, data, data, file_names)

        for value, fname in outliers:
            all_outliers.append(fname)

        # Test for normality
        if len(filtered_data) > 0:
            stat, p_value = kstest(filtered_data, 'norm', args=(np.mean(filtered_data), np.std(filtered_data)))
            print(f"Kolmogorov-Smirnov Test: Statistic={stat}, p-value={p_value}")
            if p_value < 0.05:
                print(f"The data for {title} is likely not normally distributed (p < 0.05).")
            else:
                print(f"The data for {title} is likely normally distributed (p >= 0.05).")

        # Plot distribution
        # plot_distribution(
        #     filtered_data,
        #     title=title,
        #     xlabel="Angle (degrees)",
        #     ylabel="Frequency",
        #     bins=20
        # )

        # sort fils with number of las points
    file_list = os.listdir(base_path)
    data = {}
    for file in file_list:
        las_file = laspy.read(f'{base_path}/{file}')
        data[file] = las_file.header.point_records_count


    sorted_data = dict(sorted(data.items(), key=lambda item: item[1], reverse=True))
    # if value is less than 10000, add to outliers
    for key, value in sorted_data.items():
        if value < 10000:
            all_outliers.append(key)
            print(f"Outlier Value: {value}, File: {key}")
    
    # Save the list of outliers to a file
    with open("outliers.txt", "w") as file:
        for fname in set(all_outliers):
            file.write(f"{fname}\n")
    print(f"Outliers saved to 'outliers.txt'.")
    print(f"Total outliers: {len(set(all_outliers))}")

    image_files = []
    image_path = f"{target_dir}/top_down"
    for fname in set(all_outliers):
        image_files.append(fname.replace(".las", ".png"))

    # Copy images to a new directory
    outlier_path = f"{target_dir}/images_outliers"
    os.makedirs(outlier_path, exist_ok=True)
    for image_file in image_files:
        source_path = os.path.join(image_path, image_file)
        target_path = os.path.join(outlier_path, image_file)
        if os.path.exists(source_path):
            shutil.copy(source_path, target_path)
            print(f"Copied: {image_file}")
        else:
            print(f"File not found: {image_file}")

    # copy remaining images to the new directory
    remaining_images_dir = f"{target_dir}/images_remaining"
    os.makedirs(remaining_images_dir, exist_ok=True)
    all_images = set(os.listdir(image_path))
    outlier_images = set(os.listdir(outlier_path))
    remaining_images = all_images - outlier_images
    for file_name in remaining_images:
        source_path = os.path.join(image_path, file_name)
        target_path = os.path.join(remaining_images_dir, file_name)
        if os.path.exists(source_path):
            shutil.copy(source_path, target_path)
            print(f"Copied: {file_name}")
        else:
            print(f"File not found: {file_name}")

def compare_measurements(target_dir):
         
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt


    # file1 = 'output/results/csv/ada_ramp_measurements_no_outliers.csv'
    # file1 = 'output/results/csv_shankar/ada_ramp_measurements.csv'
    file1 = f'{target_dir}/csv/ada_ramp_measurements.csv'
    file2 = 'manual_measurements_spreadsheet.csv'


    # Load CSV files into DataFrames
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Rename 'File Name' in df2 to match df1
    df2["File Name"] = df2["File Name"].str.replace(" - Cloud", "", regex=False)

    # Match files in df2 with df1 and create combined DataFrames
    df1.set_index("File Name", inplace=True)
    df2.set_index("File Name", inplace=True)
    matching_files = df2.index.intersection(df1.index)
    combined_df1 = df1.loc[matching_files].reset_index()
    combined_df2 = df2.loc[matching_files].reset_index()

    # print(combined_df1)

    # Define categories and compliance thresholds
    categories = {
        "A": range(1, 4), "B": range(4, 7), "C": range(7, 10), "D": [10], "E": [11],     
        "F": range(12, 14), "G": range(14, 17), "H": range(17, 20), "I": range(20, 23), "J": range(23, 26), "K": range(26, 29), "L": range(29, 32)
        
    }

    compliance_thresholds = {
        "A": 7.7, "B": 1.7, "C": 49.75, "D": 9.2, "E": 9.2,
        "F": 1.7, "G": 5.2, "H": 5.2, "I": 1.7, "J": 1.7, "K": 49.75, "L": 49.75
    }

    # Define flagging margins
    # 10percent of compliance threshold is the flagging margin
    flagging_margins = {}        
    for category, threshold in compliance_thresholds.items():
        if category in ["C", "K", "L"]:
            flagging_margins[category] = threshold * 0.20
        else:
            flagging_margins[category] = threshold * 0.20
    # flagging_margins = {
    #     "A": 0.77, "B": 0.17, "C": 4.975, "D": 0.92, "E": 0.92,
    #     "F": 0.17, "G": 0.52, "H": 0.52, "I": 0.17, "J": 0.17, "K": 4.975, "L": 4.975
    # }

    # flagging_margins = {
    #     "A": 0.1, "B": 0.1, "C": 2, "D": 0.1, "E": 0.1,
    #     "F": 0.1, "G": 0.1, "H": 0.1, "I": 0.1, "J": 0.1, "K": 2, "L": 2
    # }

    # Lower and upper bounds for flagging
    compliance_thresholds_lower = {k: v - flagging_margins[k] for k, v in compliance_thresholds.items()}
    compliance_thresholds_upper = {k: v + flagging_margins[k] for k, v in compliance_thresholds.items()}

    # Compute per-row values based on compliance rules
    processed_df1 = pd.DataFrame()
    processed_df2 = pd.DataFrame()

    for category, indices in categories.items():
        if category in ["C", "K", "L"]:
            processed_df1[category] = combined_df1.iloc[:, indices].apply(lambda row: min(row) if min(row) < compliance_thresholds[category] else compliance_thresholds[category], axis=1)
            processed_df2[category] = combined_df2.iloc[:, indices].apply(lambda row: min(row) if min(row) < compliance_thresholds[category] else compliance_thresholds[category], axis=1)
        else:
            processed_df1[category] = combined_df1.iloc[:, indices].apply(lambda row: max(row) if max(row) > compliance_thresholds[category] else compliance_thresholds[category], axis=1)
            processed_df2[category] = combined_df2.iloc[:, indices].apply(lambda row: max(row) if max(row) > compliance_thresholds[category] else compliance_thresholds[category], axis=1)

    raw_df1 = pd.DataFrame()
    raw_df2 = pd.DataFrame()

    for category, indices in categories.items():
        if category in ["C", "K", "L"]:
            raw_df1[category] = combined_df1.iloc[:, indices].apply(lambda row: min(row), axis=1)
            raw_df2[category] = combined_df2.iloc[:, indices].apply(lambda row: min(row),  axis=1)
        else:
            raw_df1[category] = combined_df1.iloc[:, indices].apply(lambda row: max(row), axis=1)
            raw_df2[category] = combined_df2.iloc[:, indices].apply(lambda row: max(row), axis=1)

    # Compute compliance dataframes
    compliance_df1 = pd.DataFrame()
    compliance_df2 = pd.DataFrame()
    flagged_df = pd.DataFrame()

    for category in categories.keys():
        within_margin = (processed_df1[category] >= compliance_thresholds_lower[category]) & (processed_df1[category] <= compliance_thresholds_upper[category])

        if category in ["C", "K", "L"]:
            compliance_df1[category] = processed_df1[category] >= compliance_thresholds[category]
            compliance_df2[category] = processed_df2[category] >= compliance_thresholds[category]
        else:
            compliance_df1[category] = processed_df1[category] <= compliance_thresholds[category]
            compliance_df2[category] = processed_df2[category] <= compliance_thresholds[category]

        flagged_df[category] = within_margin

    # print(flagged_df)

    # Compute flipped cases, excluding flagged ones
    flipped_mask = (compliance_df1 != compliance_df2) & ~flagged_df
    opposed_compliance_diff_df = raw_df1.where(flipped_mask) - raw_df2.where(flipped_mask)

    # Convert flipped differences to long format for box plot
    opposed_diff_melted = opposed_compliance_diff_df.melt(var_name="Category", value_name="Difference").dropna()

    # Compute flipped ratio
    flipped_ratio_df = flipped_mask.mean().reset_index()
    flipped_ratio_df.columns = ["Category", "Flipped Ratio"]
    flipped_ratio_df["Flipped Ratio"] *= 100

    # Compute flagged ratio
    flagged_ratio_df = flagged_df.mean().reset_index()
    flagged_ratio_df.columns = ["Category", "Flagged Ratio"]
    flagged_ratio_df["Flagged Ratio"] *= 100

    ramp_count = len(combined_df1)  

    # Create subplots
    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1], width_ratios=[1, 1])
    fig.suptitle(f"Comparison of Our Measurements vs. Manual Field Measurements ({ramp_count}ramps)", fontsize=16, fontweight="bold")

    palette = sns.color_palette("Spectral", len(categories))
    category_colors = {category: palette[i] for i, category in enumerate(categories.keys())}

    # Top plot: Flagged Ratio
    # ax0 = fig.add_subplot(gs[0, :])
    # sns.barplot(
    #     ax=ax0,
    #     data=flagged_ratio_df,
    #     palette=category_colors,
    #     x="Category",
    #     y="Flagged Ratio"
    # )
    # ax0.set_title("Percentage of Flagged Cases for Manual Review per Label", pad=10)
    # ax0.set_xlabel("Label")
    # ax0.set_ylabel("Percentage")
    # ax0.set_ylim(0, 100)
    # for i, value in enumerate(flagged_ratio_df["Flagged Ratio"]):
    #     ax0.text(i, value, f"{value:.2f}", ha='center', va='bottom', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

    # # Middle plot: Flipped Ratio
    ax1 = fig.add_subplot(gs[0, :])
    sns.barplot(
        ax=ax1,
        data=flipped_ratio_df,
        palette=category_colors,
        x="Category",
        y="Flipped Ratio"
    )
    ax1.set_title("Percentage of Confident Conflicting ADA-Compliance Assessment Exceeding Margin per Label", pad=10)
    ax1.set_xlabel("Label")
    ax1.set_ylabel("Percentage")
    ax1.set_ylim(0, 100)
    for i, value in enumerate(flipped_ratio_df["Flipped Ratio"]):
        ax1.text(i, value, f"{value:.2f}", ha='center', va='bottom', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

    # Bottom plot: Measurement Differences
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_xlabel("Label (Slope)")
    ax2.set_ylabel("Measurement Error")
    ax2.set_ylim(-5, 5)
    ax2.set_title('Measurement Errors for Confident Compliance Flips Exceeding Margin per Label', pad=10, loc='right')

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_xlabel("Label (Length)")
    ax3.set_ylabel("Measurement Error")
    ax3.set_ylim(-20, 20)

    left_categories = [c for c in categories.keys() if c not in ["C", "K", "L"]]
    right_categories = ["C", "K", "L"]

    # Scatter plot for slope-related categories
    sns.scatterplot(
        ax=ax2,
        data=opposed_diff_melted[opposed_diff_melted["Category"].isin(left_categories)],
        x="Category",
        y="Difference",
        hue="Category",  # Keeps colors consistent
        palette=category_colors, # {c: category_colors[c] for c in left_categories},
        s=100,  # Size of the points
        edgecolor="black"
    )

    # Scatter plot for length-related categories
    sns.scatterplot(
        ax=ax3,
        data=opposed_diff_melted[opposed_diff_melted["Category"].isin(right_categories)],
        x="Category",
        y="Difference",
        hue="Category",
        palette={c: category_colors[c] for c in right_categories},
        s=100,
        edgecolor="black"
    )

    # Remove legend if not needed
    ax2.legend_.remove()
    ax3.legend_.remove()

    plt.tight_layout()
    # save plt
    plt.savefig(f'{target_dir}/ours_vs_manual.png')
    # plt.show()

def main(las_file_path, corner_points, target_dir):

    selected_segments_ramp = [9, 5, 1]
    selected_segments_road = [9, 5, 1]
    selected_segments_landing = [9, 6, 3]
    selected_segments_gutter = [8, 5]
    selected_segments_flare = [10]
    
    las_file_name = las_file_path.split('/')[-1].replace('.las', '')
    # in_file = laspy.read(las_file_path)


    in_file = laspy.read(las_file_path)
    temp_infile = laspy.read(las_file_path.replace('all_pointcloud', 'data/pointclouds'))
    points = np.vstack((in_file.x, in_file.y, in_file.z)).transpose()
    offsetsNscales = np.load(f"OffsetsScales/{'_'.join(las_file_name.split('_')[:-1])}.npy")
    offsetsOld, scalesOld = offsetsNscales[:3], offsetsNscales[3:]
    # points = ((points - in_file.header.offsets) / in_file.header.scales) * scalesOld + offsetsOld
    classification = in_file.classification
    classification_1 = np.where(classification == 1) # main ramp
    classification_2 = np.where(classification == 2) # left flare
    classification_3 = np.where(classification == 3) # right flare
    # point_cloud = np.vstack((in_file.x, in_file.y, in_file.z)).T
    # point_cloud_mean = point_cloud.mean(axis=0)

    # corner_points = normalize_points(corner_points, point_cloud_mean)
    point_cloud = points
    # point_cloud = normalize_point_cloud(point_cloud, point_cloud_mean)
    point_cloud_1 = point_cloud[classification_1]
    point_cloud_2 = point_cloud[classification_2]
    point_cloud_3 = point_cloud[classification_3]

    # point_cloud = crop_point_cloud(corner_points[:4], point_cloud, padding=300)

    # print(len(point_cloud))
    # print(len(point_cloud_1))
    # print(len(point_cloud_2))
    # print(len(point_cloud_3))

    p1, p2, p3, p4, fr, fl = corner_points
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    p4 = np.array(p4)
    fr = np.array(fr)
    fl = np.array(fl)

    # opposite of ((points - in_file.header.offsets) / in_file.header.scales) * scalesOld + offsetsOld
    p1 = ((p1 - offsetsOld) / scalesOld) * temp_infile.header.scales + temp_infile.header.offsets
    p2 = ((p2 - offsetsOld) / scalesOld) * temp_infile.header.scales + temp_infile.header.offsets
    p3 = ((p3 - offsetsOld) / scalesOld) * temp_infile.header.scales + temp_infile.header.offsets
    p4 = ((p4 - offsetsOld) / scalesOld) * temp_infile.header.scales + temp_infile.header.offsets
    fr = ((fr - offsetsOld) / scalesOld) * temp_infile.header.scales + temp_infile.header.offsets
    fl = ((fl - offsetsOld) / scalesOld) * temp_infile.header.scales + temp_infile.header.offsets



    # p1, p2, p3, p4 = refine_corner_points(p1, p2, p3, p4)


    # Step 1: Calculate the direction vector
    direction_vector = (p2[:2] - p1[:2]) / np.linalg.norm(p2[:2] - p1[:2])
    normal_vector = np.array([-direction_vector[1], direction_vector[0]])  # Perpendicular vector

    offset = 1  # Convert 20 inches to meters (adjust if needed)
    p1_top = p1.copy()
    p2_top = p2.copy()
    p1_top[:2] += normal_vector * offset
    p2_top[:2] += normal_vector * offset

    # Step 3: Fit the top line to the point cloud
    diffs = []
    fitted_line = find_perpendicular_top_line(p1, p2, p3, p4, point_cloud, search_radius=5000)
    for i in range(3100, 100000, 100):
        # print(fitted_line)
        new_fitted_line = find_perpendicular_top_line(p1, p2, p3, p4, point_cloud, search_radius=i)
        # if distance between the two lines in z direction is more than 1000, break
        diffs.append(abs(0.5 * (fitted_line[0][2] + fitted_line[1][2]) - 0.5 * (new_fitted_line[0][2] + new_fitted_line[1][2])))
        fitted_line = new_fitted_line
        
    # what index is the largest difference in diffs
    # small_index = diffs.index(min(diffs))
    # new_diffs = diffs[small_index:]
    # large_index = new_diffs.index(max(new_diffs))
    small_index = diffs.index(min(diffs))
    large_index = diffs.index(max(diffs))
    index = (small_index + large_index) // 2
    search_index = 1100 + 100 * (index)
    # print(diffs)
    # plot diffs

    # if las_file_path == 'all_pointcloud/woodlandADA_r8_Left_GND_16.las':
    #     plt.plot(diffs)
    #     plt.show()    

    
    
    fitted_line = find_perpendicular_top_line(p1, p2, p3, p4, point_cloud, search_radius=search_index)



    intersection_points = project_fitted_line_to_z(fitted_line, point_cloud, z_direction='-z')
    new_top_landing_line = create_new_top_landing_line(intersection_points)
    new_top_p1, new_top_p2 = new_top_landing_line

    colored_area_ramp = create_colored_area(p1, p2, p3, p4, 'rgba(255, 0, 0, 0.5)')

    colored_area_flare_l = create_colored_area_flare(fl, p1, p4, 'rgba(0, 0, 255, 0.5)')
    colored_area_flare_r = create_colored_area_flare(fr, p2, p3, 'rgba(0, 0, 255, 0.5)')

    colored_area_landing = create_colored_area(p1, p2, new_top_p1, new_top_p2, 'rgba(0, 255, 0, 0.5)')



    

    # fitted_gutter_bottom_line = find_perpendicular_top_line(p3, p4, p1, p2, point_cloud, search_radius=2)
    # intersection_points_gutter_bottom = project_fitted_line_to_z(fitted_gutter_bottom_line, point_cloud, z_direction='-z')
    # new_bottom_gutter_line = create_new_top_landing_line(intersection_points_gutter_bottom)
    # new_bottom_g1, new_bottom_g2 = new_bottom_gutter_line
    # colored_area_2 = create_colored_area(p4, p3, new_bottom_g1, new_bottom_g2)

    # fitted_gutter_middle_line = find_perpendicular_top_line(p3, p4, p1, p2, point_cloud, search_radius=4)
    # intersection_points_gutter_middle = project_fitted_line_to_z(fitted_gutter_middle_line, point_cloud, z_direction='-z')
    # new_middle_gutter_line = create_new_top_landing_line(intersection_points_gutter_middle)
    # new_middle_g1, new_middle_g2 = new_middle_gutter_line
    # colored_area_3 = create_colored_area(p4, p3, new_middle_g1, new_middle_g2)

    # Step 4: Find the furthest points on the extended lines
    # g1, g2 = find_downward_intersection(p1, p2, p3, p4, point_cloud, max_extension=10, xy_tolerance=0.2)

    g1 = p4
    g2 = p3

    fitted_gutter_line = find_perpendicular_top_line(g2, g1, p1, p2, point_cloud, search_radius=20000)
    intersection_points_gutter = project_fitted_line_to_z(fitted_gutter_line, point_cloud, z_direction='-z')
    new_top_gutter_line = create_new_top_landing_line(intersection_points_gutter)
    g3, g4 = new_top_gutter_line
    colored_area_gutter = create_colored_area(g1, g2, g3, g4, 'rgba(255, 165, 0, 0.5)')


    # fitted_gutter_middle_line = find_perpendicular_top_line(g2, g1, p1, p2, point_cloud, search_radius=8)
    # intersection_points_gutter_middle = project_fitted_line_to_z(fitted_gutter_middle_line, point_cloud, z_direction='-z')
    # new_middle_gutter_line = create_new_top_landing_line(intersection_points_gutter_middle)
    # new_middle_g1, new_middle_g2 = new_middle_gutter_line
    # colored_area_2 = create_colored_area(g1, g2, new_middle_g1, new_middle_g2)



    segments_p1_p4 = divide_line_into_segments(p1, p4)
    segments_p2_p3 = divide_line_into_segments(p2, p3)
    segments_p1_p2 = divide_line_into_segments(p1, p2)
    segments_p4_p3 = divide_line_into_segments(p4, p3)

    
    l3 = g3 
    l4 = g4 
    segments_g1_g2 = divide_line_into_segments(g1, g2)
    segments_g1_g4 = divide_line_into_segments_100(g1, g4)
    
    l1 = g4 =  segments_g1_g4[85]
    segments_g2_g3 = divide_line_into_segments_100(g2, g3)
    l2 = g3 = segments_g2_g3[85]

    # l3 = segments_g2_g3[8]
    # l4 = segments_g1_g4[8]
    segments_g4_g3 = divide_line_into_segments(g4, g3)
    segments_g1_g4 = divide_line_into_segments(g1, g4)
    segments_g2_g3 = divide_line_into_segments(g2, g3)

    segments_l1_l2 = divide_line_into_segments(l1, l2)
    segments_l2_l3 = divide_line_into_segments(l2, l3)
    segments_l4_l3 = divide_line_into_segments(l4, l3)
    segments_l1_l4 = divide_line_into_segments(l1, l4)  



    segments_p1_fl = divide_line_into_segments(p1, fl)
    segments_p2_fr = divide_line_into_segments(p2, fr)

    nearest_points_a = np.empty((0, 3))
    nearest_points_a_masked = np.empty((0, 3))
    nearest_points_selected_segments_b_hole = np.empty((0, 3))
    nearest_points_selected_segments_b_masked_hole = np.empty((0, 3))

    nearest_points_b = np.empty((0, 3))
    nearest_points_b_masked = np.empty((0, 3))

    nearest_points_d = np.empty((0, 3))
    nearest_points_d_masked = np.empty((0, 3))

    nearest_points_e = np.empty((0, 3))
    nearest_points_e_masked = np.empty((0, 3))

    nearest_points_f = np.empty((0, 3))
    nearest_points_f_masked = np.empty((0, 3))

    nearest_points_g = np.empty((0, 3))
    nearest_points_g_masked = np.empty((0, 3))

    a_calc, b_calc, c_calc, d_calc, e_calc, f_calc, g_calc = [[] for _ in range(7)]
    aline_points, bline_points, dline_points, eline_points, fline_points, gline_points = [[] for _ in range(6)]
    trace_a_lines, trace_b_lines, trace_d_lines, trace_e_lines, trace_f_lines, trace_g_lines = [[] for _ in range(6)]

    # FOR LANDING AREA
    segments_landing_top_p1_top_p2 = divide_line_into_segments(new_top_p1, new_top_p2)
    segments_landing_top_p2_p2 = divide_line_into_segments(new_top_p2, p2)
    segments_landing_p1_p2 = divide_line_into_segments(p1, p2)
    segments_landing_top_p1_p1 = divide_line_into_segments(new_top_p1, p1)
    new_top_p1 = segments_landing_top_p1_p1[1]
    new_top_p2 = segments_landing_top_p2_p2[1]
    segments_landing_top_p1_p1 = divide_line_into_segments(new_top_p1, p1)
    segments_landing_top_p2_p2 = divide_line_into_segments(new_top_p2, p2)


    neatest_points_i = np.empty((0, 3))
    neatest_points_i_masked = np.empty((0, 3))
    i_calc = []
    iline_points = []
    trace_i_lines = []
    k_calc = []

    nearest_points_j = np.empty((0, 3))
    nearest_points_j_masked = np.empty((0, 3))
    j_calc = []
    jline_points = []
    trace_j_lines = []
    l_calc = []


    nearest_points_h = np.empty((0, 3))
    nearest_points_h_masked = np.empty((0, 3))
    h_calc = []
    hline_points = []
    trace_h_lines = []
    h_calc = []


    # Ramp Slope
    for k, i in enumerate(selected_segments_ramp):
        temp_nearest_points_a = nearest_points(segments_p1_p2[i], segments_p4_p3[i], point_cloud, gutter=False)
        nearest_points_a = np.vstack((nearest_points_a, temp_nearest_points_a))

        X = temp_nearest_points_a[:, :2]
        y = temp_nearest_points_a[:, 2]

        pca, cul_mask = fit_line_with_pca_threshold(X, y)
        pca, cul_mask = fit_line_with_pca_threshold_z(X, y, cul_mask)
        
        temp_nearest_points_a_masked = temp_nearest_points_a[cul_mask]
        nearest_points_a_masked = np.vstack((nearest_points_a_masked, temp_nearest_points_a_masked))

        slope = calculate_slope_pca(pca)
        a_calc.append(slope)

        mean = pca.mean_
        pca_direction = pca.components_[0]
        pca_direction_xy = pca_direction[:2]


        d_p1_p2 = p2[:2] - p1[:2]
        d_p3_p4 = p4[:2] - p3[:2]
        intersection1_xy = find_intersection(p1[:2], d_p1_p2, mean[:2], pca_direction_xy)
        intersection2_xy = find_intersection(p3[:2], d_p3_p4, mean[:2], pca_direction_xy)

        line_point = np.array(mean) 
        direction_vector = np.array(pca_direction)

        x1, y1 = intersection1_xy
        x2, y2 = intersection2_xy
        z1 = find_z_for_xy_on_line(x1, y1, line_point, direction_vector)
        z2 = find_z_for_xy_on_line(x2, y2, line_point, direction_vector)

        intersection_point_1 = np.array([x1, y1, z1])
        intersection_point_2 = np.array([x2, y2, z2])

        aline_points.append([intersection_point_1, intersection_point_2])

    # Ramp Cross Slope
    for k, i in enumerate(selected_segments_ramp):
        temp_nearest_points_b = nearest_points(segments_p1_p4[i], segments_p2_p3[i], point_cloud, gutter=False)
        nearest_points_b = np.vstack((nearest_points_b, temp_nearest_points_b))

        X = temp_nearest_points_b[:, :2]
        y = temp_nearest_points_b[:, 2]

        pca, cul_mask = fit_line_with_pca_threshold(X, y)
        pca, cul_mask = fit_line_with_pca_threshold_z(X, y, cul_mask)
        
        temp_nearest_points_b_masked = temp_nearest_points_b[cul_mask]
        nearest_points_b_masked = np.vstack((nearest_points_b_masked, temp_nearest_points_b_masked))

        slope = calculate_slope_pca(pca)
        b_calc.append(slope)

        pca_mean = pca.mean_
        pca_direction = pca.components_[0]
        pca_direction_xy = pca_direction[:2]


        d_p4_p1 = p1[:2] - p4[:2]
        d_p2_p3 = p3[:2] - p2[:2]

        intersection1_xy = find_intersection(p4[:2], d_p4_p1, pca_mean[:2], pca_direction_xy)
        intersection2_xy = find_intersection(p2[:2], d_p2_p3, pca_mean[:2], pca_direction_xy)

        line_point = np.array(pca_mean)
        direction_vector = np.array(pca_direction)

        x1, y1 = intersection1_xy
        x2, y2 = intersection2_xy
        z1 = find_z_for_xy_on_line(x1, y1, line_point, direction_vector)
        z2 = find_z_for_xy_on_line(x2, y2, line_point, direction_vector)

        intersection_point_1 = np.array([x1, y1, z1])
        intersection_point_2 = np.array([x2, y2, z2])

        sorted_indices = np.argsort(np.array([distance.euclidean(intersection_point_1, point) for point in temp_nearest_points_b_masked]))
        nearest_points_sorted = temp_nearest_points_b_masked[sorted_indices]
        projections = project_points_to_line(nearest_points_sorted, line_point, pca.components_[0])
        projections = np.vstack((intersection_point_1, projections))
        projections = np.vstack((projections, intersection_point_2))

        gaps = find_gaps_in_projections(projections)

        if len(gaps) > 0: # gap exist
            end_to_hole = calculate_3d_distance(intersection_point_1, projections[gaps[0][0]])
            hole_to_end = calculate_3d_distance(projections[gaps[0][1]], intersection_point_2)
            bline_points.append([intersection_point_1, projections[gaps[0][0]], intersection_point_2, projections[gaps[0][1]]])
            c_calc.append(max(end_to_hole, hole_to_end))
        else:
            bline_points.append([intersection_point_1, intersection_point_2])
            c_calc.append(calculate_3d_distance(intersection_point_1, intersection_point_2))

    # Left Flare Slope
    for k, i in enumerate(selected_segments_flare):
        temp_nearest_points_d = nearest_points(segments_p1_fl[i], segments_p1_p4[i], point_cloud, gutter=False)
        nearest_points_d = np.vstack((nearest_points_d, temp_nearest_points_d))

        X = temp_nearest_points_d[:, :2]
        y = temp_nearest_points_d[:, 2]

        pca, cul_mask = fit_line_with_pca_threshold(X, y)
        pca, cul_mask = fit_line_with_pca_threshold_z(X, y, cul_mask)
        
        temp_nearest_points_d_masked = temp_nearest_points_d[cul_mask]
        nearest_points_d_masked = np.vstack((nearest_points_d_masked, temp_nearest_points_d_masked))

        slope = calculate_slope_pca(pca)
        d_calc.append(slope)

        mean = pca.mean_
        pca_direction = pca.components_[0]
        pca_direction_xy = pca_direction[:2]


        d_p1_p4 = p4[:2] - p1[:2]
        d_fl_p1 = p1[:2] - fl[:2]
        intersection1_xy = find_intersection(p1[:2], d_p1_p4, mean[:2], pca_direction_xy)
        intersection2_xy = find_intersection(fl[:2], d_fl_p1, mean[:2], pca_direction_xy)

        line_point = np.array(mean) 
        direction_vector = np.array(pca_direction)

        x1, y1 = intersection1_xy
        x2, y2 = intersection2_xy
        z1 = find_z_for_xy_on_line(x1, y1, line_point, direction_vector)
        z2 = find_z_for_xy_on_line(x2, y2, line_point, direction_vector)

        intersection_point_1 = np.array([x1, y1, z1])
        intersection_point_2 = np.array([x2, y2, z2])

        dline_points.append([intersection_point_1, intersection_point_2])

    # Right Flare Slope
    for k, i in enumerate(selected_segments_flare):
        temp_nearest_points_e = nearest_points(segments_p2_fr[i], segments_p2_p3[i], point_cloud, gutter=False)
        nearest_points_e = np.vstack((nearest_points_e, temp_nearest_points_e))

        X = temp_nearest_points_e[:, :2]
        y = temp_nearest_points_e[:, 2]

        pca, cul_mask = fit_line_with_pca_threshold(X, y)
        pca, cul_mask = fit_line_with_pca_threshold_z(X, y, cul_mask)
        
        temp_nearest_points_e_masked = temp_nearest_points_e[cul_mask]
        nearest_points_e_masked = np.vstack((nearest_points_e_masked, temp_nearest_points_e_masked))

        slope = calculate_slope_pca(pca)
        e_calc.append(slope)

        mean = pca.mean_
        pca_direction = pca.components_[0]
        pca_direction_xy = pca_direction[:2]


        d_p2_fr = fr[:2] - p2[:2]
        d_p3_p2 = p2[:2] - p3[:2]
        intersection1_xy = find_intersection(p2[:2], d_p2_fr, mean[:2], pca_direction_xy)
        intersection2_xy = find_intersection(p3[:2], d_p3_p2, mean[:2], pca_direction_xy)

        line_point = np.array(mean) 
        direction_vector = np.array(pca_direction)

        x1, y1 = intersection1_xy
        x2, y2 = intersection2_xy
        z1 = find_z_for_xy_on_line(x1, y1, line_point, direction_vector)
        z2 = find_z_for_xy_on_line(x2, y2, line_point, direction_vector)

        intersection_point_1 = np.array([x1, y1, z1])
        intersection_point_2 = np.array([x2, y2, z2])

        eline_points.append([intersection_point_1, intersection_point_2])

    # Landing Slope (l)
    for k, i in enumerate(selected_segments_ramp):
        temp_nearest_points_j = nearest_points(segments_landing_top_p1_top_p2[i], segments_landing_p1_p2[i], point_cloud, gutter=False)
        nearest_points_j = np.vstack((nearest_points_j, temp_nearest_points_j))

        X = temp_nearest_points_j[:, :2]
        y = temp_nearest_points_j[:, 2]

        pca, cul_mask = fit_line_with_pca_threshold(X, y)
        pca, cul_mask = fit_line_with_pca_threshold_z(X, y, cul_mask)

        temp_nearest_points_j_masked = temp_nearest_points_j[cul_mask]
        nearest_points_j_masked = np.vstack((nearest_points_j_masked, temp_nearest_points_j_masked))

        slope = calculate_slope_pca(pca)
        j_calc.append(slope)

        mean = pca.mean_
        pca_direction = pca.components_[0]
        pca_direction_xy = pca_direction[:2]


        d_p1_p2 = new_top_p2[:2] - new_top_p1[:2]
        d_p3_p4 = p2[:2] - p1[:2]
        intersection1_xy = find_intersection(new_top_p1[:2], d_p1_p2, mean[:2], pca_direction_xy)
        intersection2_xy = find_intersection(p1[:2], d_p3_p4, mean[:2], pca_direction_xy)

        line_point = np.array(mean)
        direction_vector = np.array(pca_direction)

        x1, y1 = intersection1_xy
        x2, y2 = intersection2_xy
        z1 = find_z_for_xy_on_line(x1, y1, line_point, direction_vector)
        z2 = find_z_for_xy_on_line(x2, y2, line_point, direction_vector)

        intersection_point_1 = np.array([x1, y1, z1])
        intersection_point_2 = np.array([x2, y2, z2])

        jline_points.append([intersection_point_1, intersection_point_2])
        l_calc.append(calculate_3d_distance(intersection_point_1, intersection_point_2))

    # Landing Cross Slope (K)
    for k, i in enumerate(selected_segments_landing):
        temp_nearest_points_k = nearest_points(segments_landing_top_p1_p1[i], segments_landing_top_p2_p2[i], point_cloud, gutter=False)
        neatest_points_i = np.vstack((neatest_points_i, temp_nearest_points_k))

        X = temp_nearest_points_k[:, :2]
        y = temp_nearest_points_k[:, 2]

        pca, cul_mask = fit_line_with_pca_threshold(X, y)
        pca, cul_mask = fit_line_with_pca_threshold_z(X, y, cul_mask)

        temp_nearest_points_k_masked = temp_nearest_points_k[cul_mask]
        neatest_points_i_masked = np.vstack((neatest_points_i_masked, temp_nearest_points_k_masked))

        slope = calculate_slope_pca(pca)
        i_calc.append(slope)

        mean = pca.mean_
        pca_direction = pca.components_[0]
        pca_direction_xy = pca_direction[:2]

        d_p1_p1 = p1[:2] - new_top_p1[:2]
        d_p2_p2 = p2[:2] - new_top_p2[:2]
        intersection1_xy = find_intersection(new_top_p1[:2], d_p1_p1, mean[:2], pca_direction_xy)
        intersection2_xy = find_intersection(new_top_p2[:2], d_p2_p2, mean[:2], pca_direction_xy)

        line_point = np.array(mean)
        direction_vector = np.array(pca_direction)

        x1, y1 = intersection1_xy
        x2, y2 = intersection2_xy
        z1 = find_z_for_xy_on_line(x1, y1, line_point, direction_vector)
        z2 = find_z_for_xy_on_line(x2, y2, line_point, direction_vector)

        intersection_point_1 = np.array([x1, y1, z1])
        intersection_point_2 = np.array([x2, y2, z2])

        iline_points.append([intersection_point_1, intersection_point_2])
        k_calc.append(calculate_3d_distance(intersection_point_1, intersection_point_2))

    # Gutter Slope
    for k, i in enumerate(selected_segments_ramp):
        temp_nearest_points_g = nearest_points(segments_g1_g2[i], segments_g4_g3[i], point_cloud, gutter=False)
        nearest_points_g = np.vstack((nearest_points_g, temp_nearest_points_g))

        X = temp_nearest_points_g[:, :2]
        y = temp_nearest_points_g[:, 2]

        pca, cul_mask = fit_line_with_pca_threshold(X, y)
        pca, cul_mask = fit_line_with_pca_threshold_z(X, y, cul_mask)
        
        temp_nearest_points_g_masked = temp_nearest_points_g[cul_mask]
        nearest_points_g_masked = np.vstack((nearest_points_g_masked, temp_nearest_points_g_masked))

        slope = calculate_slope_pca(pca)
        g_calc.append(slope)

        mean = pca.mean_
        pca_direction = pca.components_[0]
        pca_direction_xy = pca_direction[:2]


        d_g1_g2 = g2[:2] - g1[:2]
        d_g3_g4 = g4[:2] - g3[:2]
        intersection1_xy = find_intersection(g1[:2], d_g1_g2, mean[:2], pca_direction_xy)
        intersection2_xy = find_intersection(g3[:2], d_g3_g4, mean[:2], pca_direction_xy)

        line_point = np.array(mean) 
        direction_vector = np.array(pca_direction)

        x1, y1 = intersection1_xy
        x2, y2 = intersection2_xy
        z1 = find_z_for_xy_on_line(x1, y1, line_point, direction_vector)
        z2 = find_z_for_xy_on_line(x2, y2, line_point, direction_vector)

        intersection_point_1 = np.array([x1, y1, z1])
        intersection_point_2 = np.array([x2, y2, z2])

        gline_points.append([intersection_point_1, intersection_point_2])

    # for gutter cross slope
    for k, i in enumerate(selected_segments_gutter):
        temp_nearest_points_f = nearest_points(segments_g1_g4[i], segments_g2_g3[i], point_cloud, gutter=False)
        nearest_points_f = np.vstack((nearest_points_f, temp_nearest_points_f))

        X = temp_nearest_points_f[:, :2]
        y = temp_nearest_points_f[:, 2]

        pca, cul_mask = fit_line_with_pca_threshold(X, y)
        pca, cul_mask = fit_line_with_pca_threshold_z(X, y, cul_mask)
        
        temp_nearest_points_f_masked = temp_nearest_points_f[cul_mask]
        nearest_points_f_masked = np.vstack((nearest_points_f_masked, temp_nearest_points_f_masked))

        slope = calculate_slope_pca(pca)
        f_calc.append(slope)

        pca_mean = pca.mean_
        pca_direction = pca.components_[0]
        pca_direction_xy = pca_direction[:2]


        d_g4_g1 = g1[:2] - g4[:2]
        d_g2_g3 = g3[:2] - g2[:2]

        intersection1_xy = find_intersection(g4[:2], d_g4_g1, pca_mean[:2], pca_direction_xy)
        intersection2_xy = find_intersection(g2[:2], d_g2_g3, pca_mean[:2], pca_direction_xy)

        line_point = np.array(pca_mean)
        direction_vector = np.array(pca_direction)

        x1, y1 = intersection1_xy
        x2, y2 = intersection2_xy
        z1 = find_z_for_xy_on_line(x1, y1, line_point, direction_vector)
        z2 = find_z_for_xy_on_line(x2, y2, line_point, direction_vector)

        intersection_point_1 = np.array([x1, y1, z1])
        intersection_point_2 = np.array([x2, y2, z2])

        fline_points.append([intersection_point_1, intersection_point_2])

    # for road surface slope
    for k, i in enumerate(selected_segments_road):
        temp_nearest_points_h = nearest_points(segments_l1_l2[i], segments_l4_l3[i], point_cloud, gutter=True)
        nearest_points_h = np.vstack((nearest_points_h, temp_nearest_points_h))

        X = temp_nearest_points_h[:, :2]
        y = temp_nearest_points_h[:, 2]

        pca, cul_mask = fit_line_with_pca_threshold(X, y)
        pca, cul_mask = fit_line_with_pca_threshold_z(X, y, cul_mask)
        
        temp_nearest_points_h_masked = temp_nearest_points_h[cul_mask]
        nearest_points_h_masked = np.vstack((nearest_points_h_masked, temp_nearest_points_h_masked))

        slope = calculate_slope_pca(pca)
        h_calc.append(slope)

        mean = pca.mean_
        pca_direction = pca.components_[0]
        pca_direction_xy = pca_direction[:2]


        d_l1_l2 = l2[:2] - l1[:2]
        d_l3_l4 = l4[:2] - l3[:2]
        intersection1_xy = find_intersection(l1[:2], d_l1_l2, mean[:2], pca_direction_xy)
        intersection2_xy = find_intersection(l3[:2], d_l3_l4, mean[:2], pca_direction_xy)

        line_point = np.array(mean) 
        direction_vector = np.array(pca_direction)

        x1, y1 = intersection1_xy
        x2, y2 = intersection2_xy
        z1 = find_z_for_xy_on_line(x1, y1, line_point, direction_vector)
        z2 = find_z_for_xy_on_line(x2, y2, line_point, direction_vector)

        intersection_point_1 = np.array([x1, y1, z1])
        intersection_point_2 = np.array([x2, y2, z2])

        hline_points.append([intersection_point_1, intersection_point_2])


    c_calc = [round(c/1000, 2) for c in c_calc]
    k_calc = [round(k/1000, 2) for k in k_calc]
    l_calc = [round(l/1000, 2) for l in l_calc]

    ###############################################
    ################ VISUALIZATION ################
    ###############################################
            
    trace_point_cloud = go.Scatter3d(
        x=point_cloud[:, 0],
        y=point_cloud[:, 1],
        z=point_cloud[:, 2],
        mode='markers',
        marker=dict(size=1, color='black', opacity=0.3),
        name='Point Cloud'
    )
    trace_point_cloud_1 = go.Scatter3d(
        x=point_cloud_1[:, 0],
        y=point_cloud_1[:, 1],
        z=point_cloud_1[:, 2],
        mode='markers',
        marker=dict(size=1, color='green', opacity=0.3),
        name='Point Cloud Class 1'
    )
    trace_point_cloud_2 = go.Scatter3d(
        x=point_cloud_2[:, 0],
        y=point_cloud_2[:, 1],
        z=point_cloud_2[:, 2],
        mode='markers',
        marker=dict(size=1, color='blue', opacity=0.3),
        name='Point Cloud Class 2'
    )
    trace_point_cloud_3 = go.Scatter3d(
        x=point_cloud_3[:, 0],
        y=point_cloud_3[:, 1],
        z=point_cloud_3[:, 2],
        mode='markers',
        marker=dict(size=1, color='blue', opacity=0.3),
        name='Point Cloud Class 3'
    )

    ramp_x = [p1[0], p2[0], None, p2[0], p3[0], None, p3[0], p4[0], None, p4[0], p1[0]]
    ramp_y = [p1[1], p2[1], None, p2[1], p3[1], None, p3[1], p4[1], None, p4[1], p1[1]]
    ramp_z = [p1[2], p2[2], None, p2[2], p3[2], None, p3[2], p4[2], None, p4[2], p1[2]]

    # Creating a single trace for all segments
    trace_ramp_outline = go.Scatter3d(x=ramp_x, y=ramp_y, z=ramp_z, mode='lines', line=dict(width=3, color='black'), name='Ramp Outline')

    flare_left_x = [p1[0], fl[0], None, p4[0], fl[0]]
    flare_left_y = [p1[1], fl[1], None, p4[1], fl[1]]
    flare_left_z = [p1[2], fl[2], None, p4[2], fl[2]]

    trace_flare_left_outline = go.Scatter3d(x=flare_left_x, y=flare_left_y, z=flare_left_z, mode='lines', line=dict(width=3, color='black'), name='Left Flare Outline')

    flare_right_x = [p2[0], fr[0], None, p3[0], fr[0]]
    flare_right_y = [p2[1], fr[1], None, p3[1], fr[1]]
    flare_right_z = [p2[2], fr[2], None, p3[2], fr[2]]

    trace_flare_right_outline = go.Scatter3d(x=flare_right_x, y=flare_right_y, z=flare_right_z, mode='lines', line=dict(width=3, color='black'), name='Right Flare Outline')

    trace_a_points = go.Scatter3d(
        x=nearest_points_a_masked[:, 0],
        y=nearest_points_a_masked[:, 1],
        z=nearest_points_a_masked[:, 2],
        mode='markers',
        marker=dict(size=2.5, color='green', opacity=0.4),
        name='Ramp slope points (A)'
    )

    for i, pts in enumerate(aline_points):
        trace_a_line = go.Scatter3d(
            x=[pts[0][0], pts[1][0]],
            y=[pts[0][1], pts[1][1]],
            z=[pts[0][2], pts[1][2]],
            mode='lines+text',
            line=dict(color='green', width=4),
            text=f'A{i+1}',
            name=f'Ramp slope (A{i+1})'
        )
        trace_a_lines.append(trace_a_line)

    trace_a = [trace_a_points] + trace_a_lines

    trace_b_points = go.Scatter3d(
        x=nearest_points_b_masked[:, 0],
        y=nearest_points_b_masked[:, 1],
        z=nearest_points_b_masked[:, 2],
        mode='markers',
        marker=dict(size=2.5, color='red', opacity=0.5),
        name='Ramp cross slope points (B)'
    )

    trace_k_points = go.Scatter3d(
        x=neatest_points_i_masked[:, 0],
        y=neatest_points_i_masked[:, 1],
        z=neatest_points_i_masked[:, 2],
        mode='markers',
        marker=dict(size=2.5, color='red', opacity=0.5),
        name='Landing cross slope points (I)'
    )

    trace_l_points = go.Scatter3d(
        x=nearest_points_j_masked[:, 0],
        y=nearest_points_j_masked[:, 1],
        z=nearest_points_j_masked[:, 2],
        mode='markers',
        marker=dict(size=2.5, color='green', opacity=0.5),
        name='Landing slope points (J)'
    )

    trace_h_points = go.Scatter3d(
        x=nearest_points_h_masked[:, 0],
        y=nearest_points_h_masked[:, 1],
        z=nearest_points_h_masked[:, 2],
        mode='markers',
        marker=dict(size=2.5, color='red', opacity=0.5),
        name='Road surface points (H)'
    )

    trace_furthest_points = go.Scatter3d(
        x=[g1[0], g2[0]],
        y=[g1[1], g2[1]],
        z=[g1[2], g2[2]],
        mode='lines',
        line=dict(color='green', width=3),
        name='Furthest points'
    )

    for i, pts in enumerate(bline_points):
        if len(pts) == 2:
            trace_b_line = go.Scatter3d(
                x=[pts[0][0], pts[1][0]],
                y=[pts[0][1], pts[1][1]],
                z=[pts[0][2], pts[1][2]],
                mode='lines+text',
                line=dict(color='red', width=4),
                text=f'B{i+1}',
                name=f'Ramp cross slope (B{i+1})'
            )
            trace_b_lines.append(trace_b_line)
        else:
            trace_b_line = go.Scatter3d(
                x=[pts[0][0], pts[1][0], None, pts[2][0], pts[3][0]],
                y=[pts[0][1], pts[1][1], None, pts[2][1], pts[3][1]],
                z=[pts[0][2], pts[1][2], None, pts[2][2], pts[3][2]],
                mode='lines+text',
                line=dict(color='red', width=4),
                text=f'B{i+1}',
                name=f'Ramp cross slope (B{i+1})'
            )
            trace_b_lines.append(trace_b_line)

    trace_b = [trace_b_points] + trace_b_lines

    trace_d_points = go.Scatter3d(
        x=nearest_points_d_masked[:, 0],
        y=nearest_points_d_masked[:, 1],
        z=nearest_points_d_masked[:, 2],
        mode='markers',
        marker=dict(size=2.5, color='brown', opacity=0.5),
        name='Left flare slope points (D)'
    )

    for i, pts in enumerate(dline_points):
        trace_d_line = go.Scatter3d(
            x=[pts[0][0], pts[1][0]],
            y=[pts[0][1], pts[1][1]],
            z=[pts[0][2], pts[1][2]],
            mode='lines+text',
            line=dict(color='brown', width=4),
            text=f'D{i+1}',
            name=f'Left flare slope (D{i+1})'
        )
        trace_d_lines.append(trace_d_line)

    trace_d = [trace_d_points] + trace_d_lines

    trace_e_points = go.Scatter3d(
        x=nearest_points_e_masked[:, 0],
        y=nearest_points_e_masked[:, 1],
        z=nearest_points_e_masked[:, 2],
        mode='markers',
        marker=dict(size=2.5, color='brown', opacity=0.5),
        name='Right flare slope points (E)'
    )

    for i, pts in enumerate(eline_points):
        trace_e_line = go.Scatter3d(
            x=[pts[0][0], pts[1][0]],
            y=[pts[0][1], pts[1][1]],
            z=[pts[0][2], pts[1][2]],
            mode='lines+text',
            line=dict(color='brown', width=4),
            text=f'E{i+1}',
            name=f'Right flare slope (E{i+1})'
        )
        trace_e_lines.append(trace_e_line)

    trace_e = [trace_e_points] + trace_e_lines

    trace_f_points = go.Scatter3d(
        x=nearest_points_f_masked[:, 0],
        y=nearest_points_f_masked[:, 1],
        z=nearest_points_f_masked[:, 2],
        mode='markers',
        marker=dict(size=2.5, color='red', opacity=0.5),
        name='Gutter slope points (F)'
    )

    for i, pts in enumerate(fline_points):
        trace_f_line = go.Scatter3d(
            x=[pts[0][0], pts[1][0]],
            y=[pts[0][1], pts[1][1]],
            z=[pts[0][2], pts[1][2]],
            mode='lines+text',
            line=dict(color='red', width=4),
            text=f'F{i+1}',
            name=f'Gutter slope (F{i+1})'
        )
        trace_f_lines.append(trace_f_line)

    trace_f = [trace_f_points] + trace_f_lines

    trace_g_points = go.Scatter3d(
        x=nearest_points_g_masked[:, 0],
        y=nearest_points_g_masked[:, 1],
        z=nearest_points_g_masked[:, 2],
        mode='markers',
        marker=dict(size=2.5, color='green', opacity=0.5),
        name='Gutter slope points (G)'
    )

    for i, pts in enumerate(gline_points):
        trace_g_line = go.Scatter3d(
            x=[pts[0][0], pts[1][0]],
            y=[pts[0][1], pts[1][1]],
            z=[pts[0][2], pts[1][2]],
            mode='lines+text',
            line=dict(color='green', width=4),
            text=f'G{i+1}',
            name=f'Gutter slope (G{i+1})'
        )
        trace_g_lines.append(trace_g_line)

    trace_g = [trace_g_points] + trace_g_lines

    for i, pts in enumerate(iline_points):
        trace_k_line = go.Scatter3d(
            x=[pts[0][0], pts[1][0]],
            y=[pts[0][1], pts[1][1]],
            z=[pts[0][2], pts[1][2]],
            mode='lines+text',
            line=dict(color='red', width=4),
            text=f'I{i+1}',
            name=f'Landing cross slope (I{i+1})'
        )
        trace_i_lines.append(trace_k_line)

    trace_k = [trace_k_points] + trace_i_lines

    for i, pts in enumerate(jline_points):
        trace_l_line = go.Scatter3d(
            x=[pts[0][0], pts[1][0]],
            y=[pts[0][1], pts[1][1]],
            z=[pts[0][2], pts[1][2]],
            mode='lines+text',
            line=dict(color='green', width=4),
            text=f'J{i+1}',
            name=f'Landing slope (J{i+1})'
        )
        trace_j_lines.append(trace_l_line)

    trace_l = [trace_l_points] + trace_j_lines

    for i, pts in enumerate(hline_points):
        trace_h_line = go.Scatter3d(
            x=[pts[0][0], pts[1][0]],
            y=[pts[0][1], pts[1][1]],
            z=[pts[0][2], pts[1][2]],
            mode='lines+text',
            line=dict(color='red', width=4),
            text=f'H{i+1}',
            name=f'Road Surface slope (H{i+1})'
        )
        trace_h_lines.append(trace_h_line)

    trace_h = [trace_h_points] + trace_h_lines

    # trace_gutter_line_outline = go.Scatter3d(
    #     x=[g1[0], g2[0]],
    #     y=[g1[1], g2[1]],
    #     z=[g1[2], g2[2]],
    #     mode='lines',
    #     line=dict(width=3, color='orange'),
    #     name='Gutter Line'
    # )

    # trace_gutter_top_line_outline = go.Scatter3d(
    #     x=[top_g1[0], top_g2[0]],
    #     y=[top_g1[1], top_g2[1]],
    #     z=[top_g1[2], top_g2[2]],
    #     mode='lines',
    #     line=dict(width=3, color='orange'),
    #     name='Top Gutter Line'
    # )

    trace_landing_top_line = go.Scatter3d(
        x=[fitted_line[0][0], fitted_line[1][0]],
        y=[fitted_line[0][1], fitted_line[1][1]],
        z=[fitted_line[0][2], fitted_line[1][2]],
        mode='lines',
        line=dict(color='purple', width=4),
        name='Top Line of Landing Area'
    )

    trace_gutter_fitted_line = go.Scatter3d(
        x=[fitted_gutter_line[0][0], fitted_gutter_line[1][0]],
        y=[fitted_gutter_line[0][1], fitted_gutter_line[1][1]],
        z=[fitted_gutter_line[0][2], fitted_gutter_line[1][2]],
        mode='lines',
        line=dict(color='purple', width=4),
        name='Fitted Gutter Line'
    )

    # trace_gutter_bottom_line = go.Scatter3d(
    #     x=[fitted_gutter_bottom_line[0][0], fitted_gutter_bottom_line[1][0]],
    #     y=[fitted_gutter_bottom_line[0][1], fitted_gutter_bottom_line[1][1]],
    #     z=[fitted_gutter_bottom_line[0][2], fitted_gutter_bottom_line[1][2]],
    #     mode='lines',
    #     line=dict(color='purple', width=4),
    #     name='Bottom Line of Gutter Area'
    # )

    # trace_gutter_middle_line = go.Scatter3d(
    #     x=[fitted_gutter_middle_line[0][0], fitted_gutter_middle_line[1][0]],
    #     y=[fitted_gutter_middle_line[0][1], fitted_gutter_middle_line[1][1]],
    #     z=[fitted_gutter_middle_line[0][2], fitted_gutter_middle_line[1][2]],
    #     mode='lines',
    #     line=dict(color='purple', width=4),
    #     name='Middle Line of Gutter Area'
    # )

    trace_intersection_points = go.Scatter3d(
        x=intersection_points[:, 0],
        y=intersection_points[:, 1],
        z=intersection_points[:, 2],
        mode='markers',
        marker=dict(size=5, color='red'),
        name='Intersection Points'
    )

    trace_new_top_line = go.Scatter3d(
        x=[new_top_landing_line[0][0], new_top_landing_line[1][0]],
        y=[new_top_landing_line[0][1], new_top_landing_line[1][1]],
        z=[new_top_landing_line[0][2], new_top_landing_line[1][2]],
        mode='lines',
        line=dict(color='green', width=4),
        name='New Top Landing Line'
    )
    trace_new_top_gutter_line = go.Scatter3d(
        x=[new_top_gutter_line[0][0], new_top_gutter_line[1][0]],
        y=[new_top_gutter_line[0][1], new_top_gutter_line[1][1]],
        z=[new_top_gutter_line[0][2], new_top_gutter_line[1][2]],
        mode='lines',
        line=dict(color='green', width=4),
        name='New Top Gutter Line'
    )

    # trace_new_middle_gutter_line = go.Scatter3d(
    #     x=[new_middle_gutter_line[0][0], new_middle_gutter_line[1][0]],
    #     y=[new_middle_gutter_line[0][1], new_middle_gutter_line[1][1]],
    #     z=[new_middle_gutter_line[0][2], new_middle_gutter_line[1][2]],
    #     mode='lines',
    #     line=dict(color='green', width=4),
    #     name='New Middle Gutter Line'
    # )


    # trace_new_bottom_gutter_line = go.Scatter3d(
    #     x=[new_bottom_gutter_line[0][0], new_bottom_gutter_line[1][0]],
    #     y=[new_bottom_gutter_line[0][1], new_bottom_gutter_line[1][1]],
    #     z=[new_bottom_gutter_line[0][2], new_bottom_gutter_line[1][2]],
    #     mode='lines',
    #     line=dict(color='green', width=4),
    #     name='New Bottom Gutter Line'
    # )

    # trace_new_middle_gutter_line = go.Scatter3d(
    #     x=[new_middle_gutter_line[0][0], new_middle_gutter_line[1][0]],
    #     y=[new_middle_gutter_line[0][1], new_middle_gutter_line[1][1]],
    #     z=[new_middle_gutter_line[0][2], new_middle_gutter_line[1][2]],
    #     mode='lines',
    #     line=dict(color='green', width=4),
    #     name='New Middle Gutter Line'
    # )




    # trace_landing_top_line_outline = go.Scatter3d(
    #     x=[landing_g1[0], top_g2[0]],
    #     y=[landing_g1[1], top_g2[1]],
    #     z=[landing_g1[2], top_g2[2]],
    #     mode='lines',
    #     line=dict(width=3, color='orange'),
    #     name='Top Gutter Line'
    # )

    layout = go.Layout(
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.1),
            camera=dict(
                eye=dict(x=0.3, y=0.6, z=0.3),
            )
        ),
        title='3D Visualization of Point Cloud and Selected Points'
    )

    traces = [
        trace_point_cloud, 
        # trace_point_cloud_1,
        # trace_point_cloud_2,
        # trace_point_cloud_3,

        #trace_ramp_outline,
        #trace_flare_left_outline,
        #trace_flare_right_outline,
        # trace_gutter_line_outline,
        # trace_gutter_top_line_outline,
        # trace_landing_top_line,
        # trace_intersection_points,
        trace_new_top_line,
        # trace_gutter_fitted_line,
        trace_new_top_gutter_line,
        # trace_new_middle_gutter_line,
        # trace_gutter_bottom_line,
        # trace_new_bottom_gutter_line,
        # trace_gutter_middle_line,
        # trace_new_middle_gutter_line,
        # colored_area_ramp,
        # colored_area_flare_l,
        # colored_area_flare_r,
        # colored_area_landing,
        # colored_area_gutter,
        # colored_area_2,
        # colored_area_3,

        trace_furthest_points
    ]

    traces += trace_a # a slope
    traces += trace_b # b, c cross slope
    traces += trace_d # d left flare
    traces += trace_e # e right flare
    traces += trace_f # f gutter slope
    traces += trace_g # g gutter cross slope
    traces += trace_k # I k landing cross slope
    traces += trace_l # l J landing slope
    traces += trace_h # h road surface slope
    


#     layout = go.Layout(
#     scene=dict(
#         xaxis=dict(title='X'),
#         yaxis=dict(title='Y'),
#         zaxis=dict(title='Z'),
#         aspectmode='manual',
#         aspectratio=dict(x=1, y=1, z=0.1),
#         camera=dict(
#             eye=dict(x=0, y=0, z=1), 
#         )
#     ),
#     showlegend=False,
#     title=f'MEASUREMENT LINES ({las_file_path.split("/")[-1]})'
# )
#     fig = go.Figure(data=traces, layout=layout)

    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.4, 0.3, 0.3],
        specs=[[{'type': 'scatter3d', 'rowspan': 1}], [{'type': 'table'}], [{'type': 'table'}]]
    )

    fig.update_layout(
        scene=dict(
            aspectratio=dict(x=1, y=1, z=0.2),  # Customize the ratios as needed
            xaxis=dict(title='X-axis'),
            yaxis=dict(title='Y-axis'),
            zaxis=dict(title='Z-axis'),
        ),
        height=1200,
        title_text="3D Scatter Plot with Aspect Ratio"
    )


    for trace in traces:
        fig.add_trace(trace, row=1, col=1)


    headers_top = [
        'Ramp Slope (x.x%)', 
        'Ramp Cross Slope (x.x%)',
        'Ramp Width (inches)', 
        'Left Flare Slope (x.x%)', 
        'Right Flare Slope (x.x%)', #  
        'Gutter Slope (x.x%)', 
        'Gutter Cross Slope (x.x%)',
        'Road Surface Cross Slope (x.x%)',
        ]
    rows_top = [
        [f'A1: {a_calc[0]}%', f'A2: {a_calc[1]}%', f'A3: {a_calc[2]}%', f'7.7% or less?', f'{yes_no(max(a_calc)<=7.7)}'],
        [f'B1: {b_calc[0]}%', f'B2: {b_calc[1]}%', f'B3: {b_calc[2]}%', f'1.7% or less?', f'{yes_no(max(b_calc)<=1.7)}'],
        [f'C1: {round(c_calc[0], 2)}"', f'C2: {round(c_calc[1], 2)}"', f'C3: {round(c_calc[2], 2)}"', f'49.75" or greater?', f'{yes_no(min(c_calc)>=49.75)}'],
        [f'D1: {d_calc[0]}%', '', '', f'9.2% or less?', f'{yes_no(max(d_calc)<=9.2)}'], 
        [f'E1: {e_calc[0]}%', '', '', f'9.2% or less?', f'{yes_no(max(e_calc)<=9.2)}'],
        [f'F1: {f_calc[0]}%', f'F2: {f_calc[1]}%', '', f'1.7% or less?', f'{yes_no(max(f_calc)<=1.7)}'],
        [f'G1: {g_calc[0]}%', f'G2: {g_calc[1]}%', f'G3: {g_calc[2]}%', f'5.2% or less?', f'{yes_no(max(g_calc)<=5.2)}'],
        # H
        [f'H1: {h_calc[0]}%', f'H2: {h_calc[1]}%', f'H3: {h_calc[2]}%', f'5.2% or less?', f'{yes_no(max(h_calc)<=5.2)}'],
    ]

    fig.add_trace(
        go.Table(
            header=dict(values=headers_top, fill_color='lightgrey', align='left'),
            cells=dict(values=[rows_top[i] for i in range(len(headers_top))],fill_color='whitesmoke', align='left')
        ),
        row=2, col=1
    )


    headers_bottom = [
        'Top Landing Cross Slope (x.x%)', 'Top Landing Slope (x.x%)', 'Top Landing Width (inches)', 'Top Landing Depth (inches)'#  'Gutter Slope (x.x%)', 'Gutter Cross Slope (x.x%)
        ]
    rows_bottom = [
        [f'I1: {i_calc[0]}%', f'I2: {i_calc[1]}%', f'I3: {i_calc[2]}%', f'1.7% or less?', f'{yes_no(max(i_calc)<=1.7)}'],
        [f'J1: {j_calc[0]}%', f'J2: {j_calc[1]}%', f'J3: {j_calc[2]}%', f'1.7% or less?', f'{yes_no(max(j_calc)<=1.7)}'],
        [f'K1: {round(k_calc[0], 2)}"', f'K2: {round(k_calc[1], 2)}"', f'K3: {round(k_calc[2], 2)}"', f'49.75" or greater?', f'{yes_no(min(k_calc)>=49.75)}'],
        [f'L1: {round(l_calc[0], 2)}"', f'L2: {round(l_calc[1], 2)}"', f'L3: {round(l_calc[2], 2)}"', f'49.75" or greater?', f'{yes_no(min(l_calc)>=49.75)}'],
        
    ]

    fig.add_trace(
        go.Table(
            header=dict(values=headers_bottom, fill_color='lightgrey', align='left'),
            cells=dict(values=[rows_bottom[i] for i in range(len(headers_bottom))],fill_color='whitesmoke', align='left')
        ),
        row=3, col=1
    )


    fig.update_layout(
        title=f"ADA RAMP MEASUREMENTS ({las_file_path.split('/')[-1]})",
        showlegend=False
    )
    

    fig.write_html(f"{target_dir}/html/{las_file_path.split('/')[-1]}.html")
    # try:
    #     # fig.update_layout(layout) 
    #     fig.write_image(f"output/results_11feb/html/{las_file_path.split('/')[-1].replace('.las', '.png')}", scale=2)
    # except:
    #     print('Error saving top down view image')
    # fig.show()

    # now save to csv
    # file name, a1, a2, a3, b1, b2, b3, c1, c2, c3, d1, e1, f1, f2, g1, g2, g3, h1, h2, h3, i1, i2, i3, j1, j2, j3, k1, k2, k3, l1, l2, l3
    # las_file_path.split('/')[-1]
    # save to csv
    # include header

    with open(f'{target_dir}/csv/ada_ramp_measurements.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([las_file_path.split('/')[-1]] + a_calc + b_calc + c_calc + d_calc + e_calc + f_calc + g_calc + h_calc + i_calc + j_calc + k_calc + l_calc)




if __name__ == '__main__':
    base_path = 'pointclouds/original'
    temp_path = 'pointclouds/modified'
    target_dir = 'output'
    corners_json = 'ramp_corners.json'

    analyze_corners(temp_path, corners_json, target_dir)
    filter_outliers(target_dir, base_path)

    os.makedirs(f'{target_dir}/csv', exist_ok=True)
    os.makedirs(f'{target_dir}/html', exist_ok=True)
    las_file_list = os.listdir(base_path)
    las_file_list = [f.replace(' - Cloud', '') for f in las_file_list]

    # las_file_list = ['woodlandADA_r6_Left_GND_36.las']


    json_file = json.load(open(corners_json, 'r'))

    with open(f'{target_dir}/csv/ada_ramp_measurements.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['File Name', 'A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3', 'D1', 'E1', 'F1', 'F2', 'G1', 'G2', 'G3', 'H1', 'H2', 'H3', 'I1', 'I2', 'I3', 'J1', 'J2', 'J3', 'K1', 'K2', 'K3', 'L1', 'L2', 'L3'])


    # outlier_list = []
    # outlier_list = os.listdir(f'{target_dir}/images_outliers')
    outlier_list = os.listdir(f'{target_dir}/images_outliers')
    outlier_list = [f.replace('.png', '.las') for f in outlier_list]
    print('Outliers:')
    print(outlier_list)
    
    for las_file in tqdm(las_file_list):
        if las_file not in outlier_list:
            las_file_path = os.path.join(base_path, las_file)
            # print(las_file_path)
            try:
                corners = json_file[las_file]['corners']
            except KeyError:
                print(f'No corners found for {las_file}')
                continue
            main(las_file_path, corners, target_dir)
        else:
            print(f'{las_file} is an outlier')
    
    print('Done')

    compare_measurements(target_dir)