import os
import json
import math
import torch
import numpy as np
import open3d as o3d
from PIL import ImageDraw
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    PerspectiveCameras,
    look_at_view_transform,
)


def load_json(file_path):
    """Load data from a JSON file."""
    with open(file_path, "r") as file:
        return json.load(file)


def load_bboxes(room, bbox_dir):
    """Load bounding boxes (GT or predicted)."""
    bbox_file = os.path.join(bbox_dir, f"{room}.json")
    bboxes = load_json(bbox_file)
    return {int(bbox["bbox_id"]): bbox for bbox in bboxes}


def create_point_cloud(scan_pc, device):
    """
    Create a point cloud from scan data.

    Args:
        scan_pc (np.ndarray): The scan data containing points and colors.
        device (str): The device to use for computation.

    Returns:
        Pointclouds: The created point cloud.
    """
    points = torch.tensor(scan_pc[:, :3], dtype=torch.float32)
    colors = torch.tensor(scan_pc[:, 3:], dtype=torch.float32)
    point_cloud = Pointclouds(points=[points], features=[colors]).to(device)
    return point_cloud


def focal_length_to_fov(focal_length):
    """
    For pytorch3d NDC coordinate system
    """
    return torch.rad2deg(2 * torch.atan(1 / focal_length))


def write_bbox_ply(bbox_data, save_path):
    x, y, z, dx, dy, dz = bbox_data

    vertices = np.array([
        [x - dx/2, y - dy/2, z - dz/2],  # 0
        [x + dx/2, y - dy/2, z - dz/2],  # 1
        [x + dx/2, y + dy/2, z - dz/2],  # 2
        [x - dx/2, y + dy/2, z - dz/2],  # 3
        [x - dx/2, y - dy/2, z + dz/2],  # 4
        [x + dx/2, y - dy/2, z + dz/2],  # 5
        [x + dx/2, y + dy/2, z + dz/2],  # 6
        [x - dx/2, y + dy/2, z + dz/2]   # 7
    ])

    edges = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom square
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top square
        [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
    ])

    edge_colors = np.array([[255, 0, 0]] * len(edges))

    ply_file = save_path
    with open(ply_file, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element edge {len(edges)}\n")
        f.write("property int vertex1\nproperty int vertex2\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")  # color
        f.write("end_header\n")

        for v in vertices:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")

        for edge, color in zip(edges, edge_colors):
            f.write(f"{edge[0]} {edge[1]} {color[0]} {color[1]} {color[2]}\n")

    print(f"PLY wireframe saved: {ply_file}")


def setup_camera(
    point_cloud,
    center,
    image_size,
    camera_distance_factor=1.0,
    camera_lift=1.0,
    camera_shift=1.0,
    camera_dist=8.0,
    device="cuda",
    calibrate=True,
    zoom_in=False
):
    """
    Set up the camera for rendering the point cloud.

    Args:
        point_cloud (Pointclouds): The point cloud to render.
        center (np.ndarray): The center of the point cloud.
        image_size (int): The size of the output image.
        camera_distance_factor (float): The factor to adjust camera distance.
        camera_lift (float): The lift to apply to the camera.
        device (str): The device to use for computation.
        calibrate (bool): Whether to calibrate the camera.

    Returns:
        PerspectiveCameras: The set up camera.
    """
    # Compute the bounding box of the point cloud
    min_bounds = point_cloud.points_padded().min(dim=1)[0]
    max_bounds = point_cloud.points_padded().max(dim=1)[0]
    bound_x, bound_y, bound_z = (max_bounds - min_bounds)[0]
    max_bound = max(bound_x, bound_y)

    center = torch.tensor(center, dtype=torch.float32)
    # center[2] += camera_lift
    # camera_position = center + camera_distance_factor * (center - anchor_bbox_3d)
    # camera_position_top = center + torch.tensor([0, 0, camera_lift])
    # camera_position_up = center + torch.tensor([0, bound_y/2 + camera_shift, camera_lift])
    # camera_position_down = center + torch.tensor([0, -(bound_y/2 + camera_shift), camera_lift])
    # camera_position_left = center + torch.tensor([-(bound_x/2 + camera_shift), 0, camera_lift])
    # camera_position_right = center + torch.tensor([bound_x/2 + camera_shift, 0, camera_lift])

    '''
    corners_2d = cameras.transform_points_screen(torch.tensor(corners).cuda(), image_size=(image_size, image_size))
    '''

    # camera parameters
    focal_length = torch.tensor([[2.0, 2.0]]).to(point_cloud.device)  # Initial focal length, shape (1, 2)
    principal_point = torch.tensor([[0.0, 0.0]]).to(point_cloud.device)  # Initial principal point, shape (1, 2)

    # calculate camera FOV
    fov_x = focal_length_to_fov(focal_length[:, 0])
    fov_x_rad = math.radians(fov_x)
    fov_y = focal_length_to_fov(focal_length[:, 1])
    fov_y_rad = math.radians(fov_y)

    camera_dist_top = (max_bound / 2) / math.tan(fov_y_rad / 2)
    # camera_dist_left_right = bound_z * math.sqrt(2) + (bound_x-bound_z)
    if zoom_in:
        camera_dist_left_right = (bound_x / 2) * math.sqrt(2)  # 45°
        camera_dist_up_down = (bound_y / 2) * math.sqrt(2)  # 45°
    else:
        camera_dist_left_right = camera_dist_top
        camera_dist_up_down = camera_dist_top

    R_top, T_top = look_at_view_transform(
        dist=camera_dist_top,
        elev=0,
        azim=0,
        at=center.unsqueeze(0),
        # eye=camera_position_top.unsqueeze(0),
        up=((0, 1, 0),),
    )

    R_left, T_left = look_at_view_transform(
        dist=camera_dist_left_right,  # camera_dist_top  v1
        elev=0,
        azim=-45,
        at=center.unsqueeze(0),
        # eye=camera_position_left.unsqueeze(0),
        up=((1, 0, 0),),
    )

    R_right, T_right = look_at_view_transform(
        dist=camera_dist_left_right,  # camera_dist_top
        elev=0,
        azim=45,
        at=center.unsqueeze(0),
        # eye=camera_position_right.unsqueeze(0),
        up=((-1, 0, 0),),
    )

    R_up, T_up = look_at_view_transform(
        dist=camera_dist_up_down,  # camera_dist_top
        elev=45,
        azim=0,
        at=center.unsqueeze(0),
        # eye=camera_position_up.unsqueeze(0),
        up=((0, -1, 0),),
    )
    
    R_down, T_down = look_at_view_transform(
        dist=camera_dist_up_down,  # camera_dist_top
        elev=-45,
        azim=0,
        at=center.unsqueeze(0),
        # eye=camera_position_down.unsqueeze(0),
        up=((0, 1, 0),),
    )

    # R, T = look_at_view_transform(
    #     at=center.unsqueeze(0),
    #     eye=camera_position_left.unsqueeze(0),
    #     up=((1, 0, 0),),
    # )

    cameras_top = PerspectiveCameras(device=device, R=R_top, T=T_top, focal_length=focal_length, principal_point=principal_point,)
    cameras_down = PerspectiveCameras(device=device, R=R_down, T=T_down, focal_length=focal_length, principal_point=principal_point,)
    cameras_up = PerspectiveCameras(device=device, R=R_up, T=T_up, focal_length=focal_length, principal_point=principal_point,)
    cameras_left = PerspectiveCameras(device=device, R=R_left, T=T_left, focal_length=focal_length, principal_point=principal_point,)
    cameras_right = PerspectiveCameras(device=device, R=R_right, T=T_right, focal_length=focal_length, principal_point=principal_point,)
    
    # cameras = PerspectiveCameras(device=device, R=R, T=T, focal_length=focal_length, principal_point=principal_point,)

    if calibrate:
        if isinstance(image_size, int):
            image_size_tensor = torch.tensor(
                [[image_size, image_size]]
            )  # Convert integer to 2D tensor
        assert image_size_tensor.shape[-1] == 2

        # Get the projection of the point cloud
        points_2d = cameras.transform_points_screen(
            point_cloud.points_padded(), image_size=image_size_tensor
        )
        points_2d = points_2d[..., :2]

        # Compute the bounding box of the projected points
        min_proj = points_2d.min(dim=1)[0]
        max_proj = points_2d.max(dim=1)[0]

        # Adjust focal length and principal point to ensure all points are within the image
        new_focal_length = (
            focal_length
            * (max_proj - min_proj).max()
            / image_size_tensor.to(point_cloud.device)
        )
        new_principal_point = (min_proj + max_proj) / 2

        # Update camera intrinsics
        cameras = PerspectiveCameras(
            device=device,
            R=R,
            T=T,
            focal_length=new_focal_length,
            principal_point=new_principal_point,  # Ensure principal point is 2D
        )
    return cameras_top, cameras_down, cameras_up, cameras_left, cameras_right


def load_scan_pc(scene, ply_path):
    aligned_ply_file = os.path.join(ply_path, f"{scene}.ply")
    pcd = o3d.io.read_point_cloud(aligned_ply_file)
    pc = np.asarray(pcd.points)
    color = np.asarray(pcd.colors)

    scan_pc = np.concatenate((pc, color), axis=1).astype("float32")
    center = np.mean(scan_pc[:, :3], axis=0)

    return scan_pc, center


def draw_label(draw, corners_2d, bbox_id, font, image_size):
    """
    Draw label and bbox_id at the center of the top face of the bounding box.

    Args:
        draw (ImageDraw): ImageDraw object.
        corners_2d (array): 2D coordinates of the bounding box corners.
        bbox_id (int): Bounding box ID.
        font (ImageFont): Font for drawing text.
        image_size (int): Size of the output image.
    """
    # Find the center of the top face
    # center_x = int(
    #     (corners_2d[4][0] + corners_2d[5][0] + corners_2d[6][0] + corners_2d[7][0]) / 4
    # )
    # center_y = int(
    #     (corners_2d[4][1] + corners_2d[5][1] + corners_2d[6][1] + corners_2d[7][1]) / 4
    # )

    # center of all faces
    center_x, center_y = np.mean(corners_2d, axis=0)
    center_x, center_y = int(center_x), int(center_y)

    if 0 <= center_x < image_size and 0 <= center_y < image_size:
        text = f"{bbox_id}"
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        background_x0 = center_x - text_width // 2  #- 2
        background_y0 = center_y - text_height // 2 + 4  # adjusted
        background_x1 = center_x + text_width // 2 #+ 2
        background_y1 = center_y + text_height // 2 + 5  # adjusted
        draw.rectangle(
            [background_x0, background_y0, background_x1, background_y1],
            fill=(255, 255, 255),
        )
        draw.text(
            (center_x - text_width // 2, center_y - text_height // 2),
            text,
            font=font,
            fill=(255, 0, 0),
        )
        return True

    return False


def draw_ids(draw, bboxes, cameras, image_size, font):
    """
    Draw object IDs on the image.

    Args:
        draw (ImageDraw): ImageDraw object.
        bboxes (list): List of bounding boxes.
        cameras (PerspectiveCameras): Camera settings.
        image_size (int): Size of the output image.
        font (ImageFont): Font for drawing text.
    """
    all_id = []
    for bbox in bboxes:
        bbox_id = bbox["bbox_id"]
        x, y, z, w, l, h = bbox["bbox_3d"]

        # Define the eight corners of the 3D bounding box
        corners = [
            [x - w / 2, y - l / 2, z - h / 2],
            [x - w / 2, y + l / 2, z - h / 2],
            [x + w / 2, y - l / 2, z - h / 2],
            [x + w / 2, y + l / 2, z - h / 2],
            [x - w / 2, y - l / 2, z + h / 2],
            [x - w / 2, y + l / 2, z + h / 2],
            [x + w / 2, y - l / 2, z + h / 2],
            [x + w / 2, y + l / 2, z + h / 2],
        ]

        # Project the 3D corners to the 2D image plane
        corners_2d = cameras.transform_points_screen(
            torch.tensor(corners).cuda(), image_size=(image_size, image_size)
        )
        corners_2d = corners_2d[..., :2].cpu().numpy()

        # Check if each corner is within the image boundaries
        valid_corners = [
            (0 <= x < image_size and 0 <= y < image_size) for x, y in corners_2d
        ]

        # Skip drawing if all corners are out of image boundaries
        if not any(valid_corners):
            continue

        # Draw the label and bbox_id
        stat = draw_label(draw, corners_2d, bbox_id, font, image_size)
        if stat:
            all_id.append(bbox_id)
    return all_id
