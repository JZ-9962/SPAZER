# Retrieve scannet image using camera intrinsic parameters
import os
import cv2
import glob
import random
import imageio
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw, ImageFont
from fusion_util import PointCloudToImageMapper, adjust_intrinsic, make_intrinsic, bbox_to_corners
from pc_utils import load_mesh_data, save_mesh, load_bboxes, read_dict


class CameraImage():
    def __init__(
            self,
            scene_id,
            axis_alignment_info_file = "/raid/data/projects/SeeGround/data/scannet/scans_axis_alignment_matrices.json",
            scannet_dir = "/raid/data/datasets/ScanNet/scans",
            posed_image_path = "/raid/data/projects/VLM-Grounder/data/scannet/posed_images",
            visibility_threshold = 0.25, # threshold for the visibility check
        ):
        self.scene_id = scene_id
        self.axis_alignment_info_file = axis_alignment_info_file
        self.scannet_dir = scannet_dir
        self.posed_image_path = posed_image_path

        self._load_data(scene_id)

        ### Load image to determine img_dim
        img_0 = Image.open(self.rgb_images[0])
        self.img_dim = img_0.size

        # scannet parameters
        self.depth_scale = 1000.0
        fx = 577.870605
        fy = 577.870605
        mx = 319.5
        my = 239.5
        # calculate image pixel-3D points correspondances
        intrinsic_ = make_intrinsic(fx=fx, fy=fy, mx=mx, my=my)
        intrinsic_ = adjust_intrinsic(intrinsic_, intrinsic_image_dim=[640, 480], image_dim=self.img_dim)
        self.intrinsic = intrinsic_

        self.point2img_mapper = PointCloudToImageMapper(
            image_dim=self.img_dim,
            visibility_threshold=0.25,
            cut_bound=0,
            intrinsics=self.intrinsic
        )


    def _load_data(self, scene_id):
        # Load image files.
        self.rgb_images = sorted(glob.glob(os.path.join(self.posed_image_path, f"{scene_id}/*.jpg")))
        self.depth_images = sorted(glob.glob(os.path.join(self.posed_image_path, f"{scene_id}/*.png")))
        self.cam_paras = sorted(glob.glob(os.path.join(self.posed_image_path, f"{scene_id}/*.txt")))
        # intrinsic_paras = os.path.join(self.posed_image_path, f"{scene_id}/intrinsic.txt")

        assert len(self.rgb_images) == len(self.depth_images) == len(self.cam_paras) - 1, print("Error in loading posed_images")


    def draw_label(self, draw, corners_2d, bbox_id, font):
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

        if 0 <= center_x < self.img_dim[0] and 0 <= center_y < self.img_dim[1]:
            text = f"{bbox_id}"
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            background_x0 = center_x - text_width // 2 - 2
            background_y0 = center_y - text_height // 2 + 4  # adjusted
            background_x1 = center_x + text_width // 2 + 2
            background_y1 = center_y + text_height // 2 + 12  # adjusted
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


    def get_camera_image_id(
            self,
            bbox3d
        ):

        raise NotImplementedError
        # return image_index, mapped_points / transform
        return


    def get_annotate_image(
            self, 
            bbox3d, 
            bbox_id,
            save_dir="/raid/data/projects/SeeGround/data/scannet/cam_img_infer",
            return_path=False
        ):
        ## Load point cloud data.
        # pc_ply_file = os.path.join(scannet_dir, f"{scene_id}/{scene_id}_vh_clean_2.ply")
        # vertices, colors, faces = load_mesh_data(pc_ply_file)

        self.n_bbox = len(bbox3d)
        bbox_corners = [bbox_to_corners(box) for box in bbox3d]  # (6,3) -> (8,3)
        bbox_corners = np.array(bbox_corners).reshape(-1, 3)  # (n*8, 3)

        valid_corners_all = []
        valid_center_all = []
        inside_mask_all = []
        mapping_coords_all = []
        file_index_all = []

        # Search all images.
        for i in range(len(self.rgb_images)):
            # load image, depth, camera pose
            rgb_image_file = self.rgb_images[i]
            depth_image_file = self.depth_images[i]
            cam_para_file = self.cam_paras[i]
            file_index = rgb_image_file.split("/")[-1].split(".")[0]
            file_index_all.append(file_index)

            # (1296, 968)
            rgb_image = Image.open(rgb_image_file)
            # (640, 480)
            depth = imageio.v2.imread(depth_image_file) / self.depth_scale
            resized_depth = cv2.resize(depth, self.img_dim, interpolation=cv2.INTER_NEAREST)

            # 4x4
            camera_matrix = np.loadtxt(cam_para_file)

            # Get the axis alignment matrix
            scans_axis_alignment_matrices = read_dict(self.axis_alignment_info_file)
            alignment_matrix = scans_axis_alignment_matrices[self.scene_id]
            alignment_matrix = np.array(alignment_matrix, dtype=np.float32).reshape(4, 4)
            alignment_matrix_inv = np.linalg.inv(alignment_matrix)

            # Axis-aligned -> original 
            pts_aligned = np.ones((bbox_corners.shape[0], 4), dtype=bbox_corners.dtype)
            pts_aligned[:, 0:3] = bbox_corners
            original_corners_all = np.dot(pts_aligned, alignment_matrix_inv.T)[:, :3]  # (n*8, 3)
            original_corners_all = original_corners_all.reshape(self.n_bbox, 8, 3)  # (n, 8, 3)
            original_center = np.mean(original_corners_all, axis=1, keepdims=True)  # (n, 1, 3)
            # center(1) + corners(8)
            orignial_corners_center = np.concatenate((original_center, 
                                                      original_corners_all), axis=1)  # (n, 9, 3)
            orignial_corners_center = orignial_corners_center.reshape(-1, 3)  # (n*9, 3)

            # transform coordinates
            inside_mask, mapping_coords = self.point2img_mapper.compute_mapping(
                camera_matrix, 
                orignial_corners_center, 
                depth=resized_depth
            )

            inside_mask = inside_mask.reshape(self.n_bbox, 1+8, -1)
            mapping_coords = mapping_coords.reshape(self.n_bbox, 1+8, -1)  # (n, 9, 3)
            inside_mask_all.append(inside_mask)
            mapping_coords_all.append(mapping_coords)

            # Make sure target object idx is 0
            inside_mask = inside_mask[0]
            n_valid_center = inside_mask[0]
            n_valid_corner = np.sum(inside_mask[1:])
            # n_valid_center = inside_mask[:, 0, :]
            # n_valid_corner = np.sum(inside_mask[:, 1:, :], axis=1)

            valid_center_all.append(n_valid_center)
            valid_corners_all.append(n_valid_corner)

        # Select optimal image index
        max_value = max(valid_corners_all)
        all_max_indices = [i for i, v in enumerate(valid_corners_all) if v == max_value]
        if max_value > 1:
            second_max_value = max_value - 1
            all_second_max_indices = [i for i, v in enumerate(valid_corners_all) if v == second_max_value]
        else:
            all_second_max_indices = []
        
        visible_files = [file_index_all[i] for i in all_max_indices]
        # print(f"max corners: {max_value} ", visible_files)

        # if annotate == "all":
        #     # Annotate all candidate images.
        #     for chosen_idx in all_max_indices:
        #         annotated_image = self.annotate_idx(
        #             chosen_idx, file_index_all, inside_mask_all, mapping_coords_all, bbox_id
        #         )
        # elif annotate == "one":
        #     # Annotate one in candidates.
        #     chosen_idx = random.choice(all_max_indices)
        #     annotated_image = self.annotate_idx(
        #         chosen_idx, file_index_all, inside_mask_all, mapping_coords_all, bbox_id
        #     )
        # else:
        #     print("Wrong input type of annotate.")
        #     raise NotImplemented

        annotated_image_list = []
        for chosen_idx in all_max_indices:
            annotated_res = self.annotate_idx(
                chosen_idx, file_index_all, inside_mask_all, mapping_coords_all, bbox_id
            )
            if annotated_res is not None:
                annotated_image_list.append(annotated_res)

        if len(annotated_image_list) == 0 and len(all_second_max_indices) > 0:  # try to use second max
            print("Use second max indices")
            for chosen_idx in all_second_max_indices:
                annotated_res = self.annotate_idx(
                    chosen_idx, file_index_all, inside_mask_all, mapping_coords_all, bbox_id
                )
                if annotated_res is not None:
                    annotated_image_list.append(annotated_res)


        # choose one annotated image to return
        if len(annotated_image_list) > 0:  # not empty
            annotated_image = random.choice(annotated_image_list)
            # annotate bbox
            # annotated_image = self.draw_bbox(
            #     annotated_image, 
            #     projected_points=mapping_coords_all[chosen_idx][0][1:],
            #     color='red', 
            #     width=2
            # )
            # save
            draw_tgt_id = bbox_id[0]
            save_path = f"{save_dir}/{self.scene_id}/annotated_{draw_tgt_id}.jpg"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            annotated_image.save(save_path)
        else:
            draw_tgt_id = bbox_id[0]
            print(f"No available camera image for target {draw_tgt_id}!")
            annotated_image = None
            save_path = None

        if return_path:
            return save_path
        else:
            return annotated_image


    def clip_and_sanitize_point(self, point, width, height):
        x = point[0]
        y = point[1]
        if x is None or y is None:
            return (0, 0)  # fallback 默认点
        try:
            x = int(round(float(x)))
            y = int(round(float(y)))
        except:
            return (0, 0)
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        return (x, y)


    def draw_bbox(
            self, 
            draw,
            chosen_image, 
            projected_points,
            color='red', 
            width=2
        ):
        # draw = ImageDraw.Draw(chosen_image)
        img_width, img_height = chosen_image.size

        clipped_points = [self.clip_and_sanitize_point(p, img_width, img_height) for p in projected_points]

        # edges = [
        #     (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
        #     (4, 5), (5, 6), (6, 7), (7, 4),  # top
        #     (0, 4), (1, 5), (2, 6), (3, 7)   # vertical
        # ]
        edges = [
            (0, 1), (1, 3), (3, 2), (2, 0),  # bottom
            (4, 5), (5, 7), (7, 6), (6, 4),  # top
            (0, 4), (1, 5), (2, 6), (3, 7)   # vertical
        ]

        for start, end in edges:
            p1, p2 = clipped_points[start], clipped_points[end]
            if p1 != p2:
                draw.line([p1, p2], fill=color, width=width)

        return chosen_image
    

    def annotate_idx(
            self, 
            idx, 
            file_index_all, 
            inside_mask_all, 
            mapping_coords_all, 
            bbox_id
        ):
        chosen_file_idx = file_index_all[idx]
        chosen_image = Image.open(os.path.join(self.posed_image_path, f"{self.scene_id}/{chosen_file_idx}.jpg"))
        # print(chosen_file_idx)

        inside_mask = inside_mask_all[idx]

        center_corners_2d = mapping_coords_all[idx]  # (n, 9, 3)
        draw = ImageDraw.Draw(chosen_image, "RGBA")
        font = ImageFont.truetype("FreeSansBold.ttf", 28, encoding="unic")
        draw_id = bbox_id[0]

        stat_all = []
        # annotated_images = []
        for i in range(self.n_bbox):
            inside_mask_i = inside_mask[i].reshape(-1)  # (9, 1)
            center_corners_i = center_corners_2d[i]
            visible_points = center_corners_i
            # visible_points = center_corners_i[inside_mask_i]

            if True in inside_mask_i:  # visible
                stat = self.draw_label(draw, visible_points, bbox_id[i], font)
            else:
                stat = False
            stat_all.append(stat)

            # if stat:
            #     # draw bbox
            #     self.draw_bbox(
            #         draw,
            #         chosen_image, 
            #         center_corners_i[1:], 
            #         color='red', 
            #         width=2
            #     )
        
        # print(stat_all)
        # if True in stat_all:
        if stat_all[0]:  # Main target is annotated. 
            save_path = f"/raid/tao/projects/agent_3dvg/seeground_from_old/SeeGround/data/scannet/cam_img/{self.scene_id}/{chosen_file_idx}_annotated_{draw_id}.jpg"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            chosen_image.save(save_path)
            # print(save_path, "saved.")

            # annotated_images.append(chosen_image)
            return chosen_image

        return
    
        # For single box.
        # if annotate:
        #     for i in all_max_indices:
        #         file_idx = file_index_all[i]
        #         center_corners_2d = mapping_coords_all[i]
        #         inside_mask_i = inside_mask_all[i]
        #         visible_corner_idx = inside_mask_i.nonzero()[0]

        #         if not np.any(visible_corner_idx == 0):  # if center not visible
        #             visible_corner_idx = np.concatenate((visible_corner_idx, np.array([0])))
        #         visible_corners_2d = center_corners_2d[visible_corner_idx]
                
        #         # Annotate bbox_id
        #         # TODO: annotate all same class
        #         image_i = Image.open(os.path.join(self.posed_image_path, f"{scene_id}/{file_idx}.jpg"))
        #         draw = ImageDraw.Draw(image_i, "RGBA")
        #         font = ImageFont.truetype("/raid/data/projects/SeeGround/data/FreeSansBold.ttf", 28, encoding="unic")

        #         stat = self.draw_label(draw, visible_corners_2d, bbox_id, font)
        #         # for corners_2d in visible_corners_2d:
        #         #     stat = self.draw_label(draw, corners_2d.reshape(1, -1), bbox_id, font)

        #         # TODO: change save path
        #         image_i.save(f"/raid/data/projects/SeeGround/data/scannet/{file_idx}_annotated.jpg")
        #         print(file_idx, "saved.")


if __name__ == "__main__":
    pass

