import os
import sys
import time
import json
import nltk
import torch
import base64
import random
import argparse
import numpy as np
import pandas as pd
import scannet_utils
from PIL import Image
from tqdm import tqdm
from io import BytesIO
from openai import OpenAI
from nltk.stem import PorterStemmer
from collections import defaultdict
from difflib import get_close_matches
from nltk.tokenize import word_tokenize
from PIL import Image, ImageDraw, ImageFont
from openai.types.chat import ChatCompletionMessage
from utils.cam_utils import CameraImage
from utils.pc_utils import (
    setup_camera,
    load_scan_pc,
    create_point_cloud,
    load_bboxes,
    draw_ids
)
from prompts import (
    target_name_select_user_prompt,
    view_selection_sys_prompt,
    view_selection_user_prompt,
    topk_id_user_prompt,
    topk_id_crop_user_prompt,
    object_id_user_prompt,
    object_id_user_prompt_relation,
    all_views
)


# nltk.download("punkt_tab")  # enable in the first run
stemmer = PorterStemmer()

# Parse arguments
parser = argparse.ArgumentParser(description="Process rooms for object detection.")

# required files
parser.add_argument("--scan_id_file", type=str, help="Path to the scan ID file.")
parser.add_argument("--anno_file", type=str, help="Path to the annotation file.")
parser.add_argument("--view_image_dir", type=str, help="Path to the projection image file.")
parser.add_argument("--scanrefer_250_subset", type=str, help="Path to scanrefer_250.csv.")
parser.add_argument("--nr3d_250_subset", type=str, help="Path to nr3d_250.csv.")
parser.add_argument("--aligned_ply_dir", type=str, help="Directory containing ScanNet scans.")
parser.add_argument("--bbox_dir", type=str, help="Directory containing bbox file.")
parser.add_argument("--gt_bbox_dir", type=str, help="Directory containing gt bbox file.")
parser.add_argument("--output_dir", type=str, help="Directory containing output file.")
parser.add_argument("--bbox_crop_dir", type=str, help="Directory containing bbox crop image file.")

# task-related
parser.add_argument("--task_type", type=str, default="subset", help="Use subset or full dataset.")
parser.add_argument("--exp_name", type=str, default="scanrefer", help="Name of the experiment.")
parser.add_argument("--n_topk", type=int, default=5, help="Number of top-k objects to select.")
parser.add_argument("--view_selection", action='store_true', help='Select view')
parser.add_argument("--top_k_prediction", action='store_true', help='Predict top-k objects')
parser.add_argument("--object_id_prediction", action='store_true', help='Predict object id')

# params
parser.add_argument("--seed", type=int, default=42, help=".")
parser.add_argument("--gpt_model_type", type=str, default="gpt-4o")
parser.add_argument("--api_key", type=str, default="")
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--top_p", type=float, default=1.0)
parser.add_argument("--image_size", type=int, default=1024)

parser.add_argument("--resume_file", type=str, default=None, help=".")

args = parser.parse_args()


class Tee:
    def __init__(self, filename, mode="a"):
        self.file = open(filename, mode)
        self.stdout = sys.stdout 

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()
        sys.stdout = self.stdout


def init_gpt_client(api_key: str):
    # GPT client
    client = OpenAI(
        api_key=api_key
    )

    return client


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_json(file_path: str) -> dict:
    """Load data from a JSON file."""
    with open(file_path, "r") as file:
        return json.load(file)
    

def save_to_file(file_path: str, content: str):
    """Save content to a file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as file:
        file.write(content)


def handle_single_matched_target(
        scan_id: str,
        query_text: str,
        matched_targets: set,
        is_unique: bool,
        gt_id: str,
        targets: list,
        gt_bbox: dict,
        targets_ids: list,
        max_value_1: float,
        n_topk: int,
        output_file: str,
        results: list
    ) -> None:
    print(f"Only one matched target {targets_ids}, skip ...")
    results.append(
        {
            "scene_id": scan_id,
            "query": query_text,
            "matched_targets": str(matched_targets),
            "is_unique": is_unique,
            "gt_id": gt_id,
            "pred_bbox": targets[0]["bbox_3d"],
            "gt_bbox": gt_bbox["bbox_3d"],
            "view": None,
            "predicted_id": targets_ids[0],
            "iou": [max_value_1],
            "correct_25": str(max_value_1 > 0.25),
            "correct_50": str(max_value_1 > 0.5),
            "n_top_k": n_topk,
            "in_top_1": str(max_value_1 > 0.25),
            "in_top_k": str(max_value_1 > 0.5),
        }
    )
    save_to_file(output_file, json.dumps(results, indent=4))


def handle_detection_failure(
        scan_id: str,
        query_text: str,
        matched_targets: set,
        is_unique: bool,
        gt_id: str,
        gt_bbox: dict,
        iou_list: list,
        n_topk: int,
        output_file: str,
        results: list
    ) -> None:
    print(f"**** Detection false! {scan_id}, {query_text}")
    results.append(
        {
            "scene_id": scan_id,
            "query": query_text,
            "matched_targets": str(matched_targets),
            "is_unique": is_unique,
            "gt_id": gt_id,
            "predicted_id": None,
            "pred_bbox": None,
            "gt_bbox": gt_bbox["bbox_3d"],
            "view": None,
            "iou": iou_list,
            "correct_25": str(False),
            "correct_50": str(False),
            "n_top_k": n_topk,
            "in_top_1": str(False),
            "in_top_k": str(False),
            "correct_det": str(False)
        }
    )
    save_to_file(output_file, json.dumps(results, indent=4))


def load_ref_data(
        anno_file: str, 
        scan_id_file: str
    ) -> list[dict]:
    with open(anno_file, "r") as f:
        data = json.load(f)

    split_scan_ids = set(x.strip() for x in open(scan_id_file, "r"))
    ref_data = []
    for item in data:
        if item["scene_id"] in split_scan_ids:
            ref_data.append(item)

    print('ref_data ', len(ref_data))
    return ref_data


def load_view_images(img_path: str) -> dict[str, Image.Image]:
    view_img_data = defaultdict()
    for view in all_views:
        # view_img_file = os.path.join(img_path, f"{view}.png")
        # use mesh file
        view_img_file = os.path.join(img_path, f"_mesh_{view}.png")
        view_img = Image.open(view_img_file)
        # view_img_data.append(view_img)
        view_img_data[view] = view_img
    return view_img_data


def encode_PIL_image_to_base64(image: Image.Image) -> str:
    # Save the image to a bytes buffer
    buf = BytesIO()
    image.save(buf, format="JPEG")

    # Get the byte data from the buffer
    byte_data = buf.getvalue()

    # Encode the byte data to base64
    base64_str = base64.b64encode(byte_data).decode("utf-8")
    return base64_str


def resize_image(
        image: Image.Image, 
        max_size: int = 2048
    ) -> Image.Image:
    """
    Resize an image for the longer side to be max_size, while preserving its aspect ratio.

    Args:
        image (PIL.Image.Image): The input image to be resized.
        max_size (int, optional): The maximum size (width or height) for the resized image.
                                  Defaults to 2048.

    Returns:
        PIL.Image.Image: The resized image.
    """
    image_copy = image.copy()

    width, height = image_copy.size
    aspect_ratio = width / height

    if width > height:
        new_width = max_size
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = max_size
        new_width = int(new_height * aspect_ratio)

    resized_image = image_copy.resize((new_width, new_height), Image.LANCZOS)

    return resized_image


def filter_image_url(message_his: list[dict]) -> list[dict]:
    """
    filter all image data in messages
    """
    filtered_message_his = []
    for message in message_his:
        if isinstance(message, dict):
            if isinstance(message["content"], str):
                filtered_message_his.append(
                    {"role": message["role"], "content": message["content"]}
                )
            elif isinstance(message["content"], list):
                new_mes = {
                    "role": message["role"],
                    "content": [
                        item["text"]
                        for item in message["content"]
                        if item["type"] == "text"
                    ],
                }
                if len(new_mes["content"]) == 1:
                    new_mes["content"] = new_mes["content"][0]
                filtered_message_his.append(new_mes)
        elif isinstance(message, ChatCompletionMessage):
            filtered_message_his.append(
                {"role": message.role, "content": [message.content]}
            )
        else:
            raise Exception(f"Unknown message type: {type(message)}")

    return filtered_message_his


def fuzzy_match(
        names: str, 
        object_names: list[str], 
        threshold: float = 0.8
    ) -> set[str]:
    matched_names = set()
    for name in names:
        matches = get_close_matches(name, object_names, n=1, cutoff=threshold)
        if matches:
            matched_names.add(matches[0])
    return matched_names


def stem_match(
        names: str | list[str], 
        object_names: list[str]
    ) -> set[str]:
    matched_names = set()
    if isinstance(names, str):
        names = [names]
    for name in names:
        name_stems = [stemmer.stem(word) for word in word_tokenize(name)]
        for obj_name in object_names:
            obj_name_stems = [stemmer.stem(word) for word in word_tokenize(obj_name)]
            if set(name_stems) & set(obj_name_stems):
                matched_names.add(obj_name)
    return matched_names


def format_gpt_input_view(
        query_text: str, 
        pred_target_class: str, 
        base64Frames: list[str]
    ) -> list[dict]:
    # 1. determine object class
    # 2. annotate id, choose object

    begin_messages = [
        {"role": "system", "content": view_selection_sys_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": view_selection_user_prompt.format(
                        target_class=pred_target_class, 
                        text=query_text
                    )
                },
                *map(
                    lambda x: {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{x}",
                            "detail": "high",
                        },
                    },
                    base64Frames,
                ),
            ],
        },
    ]
    return begin_messages


def format_gpt_input_topk_id(
        pred_target_class: str, 
        query_text: str, 
        base64Frames: list[str], 
        annotated_ids: list[int]
    ) -> list[dict]:
    begin_messages = [
        {"role": "system", "content": view_selection_sys_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": topk_id_user_prompt.format(
                        target_class=pred_target_class, 
                        text=query_text, 
                        object_id_list=str(annotated_ids),
                        n_topk=args.n_topk
                    )
                },
                *map(
                    lambda x: {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{x}",
                            "detail": "high",
                        },
                    },
                    base64Frames,
                ),
            ],
        }
    ]
    return begin_messages


def format_gpt_input_topk_id_crop(
        query_text: str, 
        base64Frames: list[str]
    ) -> list[dict]:
    begin_messages = [
        {"role": "system", "content": view_selection_sys_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": topk_id_crop_user_prompt.format(
                        text=query_text,
                        n_topk=args.n_topk
                    )
                },
                *map(
                    lambda x: {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{x}",
                            "detail": "high",
                        },
                    },
                    base64Frames,
                ),
            ],
        }
    ]
    return begin_messages


def format_gpt_input_object_id(
        pred_target_class: str, 
        query_text: str, 
        base64Frames: list[str], 
        annotated_ids: list[int]
    ) -> list[dict]:
    begin_messages = [
        {"role": "system", "content": view_selection_sys_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    # "text": object_id_user_prompt.format(
                    #     target_class=pred_target_class, 
                    #     text=query_text, 
                    #     object_id_list=str(annotated_ids)
                    # )
                    "text": object_id_user_prompt_relation.format(
                        target_class=pred_target_class, 
                        text=query_text, 
                        object_id_list=str(annotated_ids)
                    )
                },
                *map(
                    lambda x: {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{x}",
                            "detail": "high",
                        },
                    },
                    base64Frames,
                ),
            ],
        }
    ]
    return begin_messages


def get_gpt_response(
        client: OpenAI, 
        begin_messages: list[dict], 
        model_name_or_path: str
    ) -> dict:
    retry = 0
    max_retry = 3
    while retry < max_retry:
        try:
            # result = client.chat.completions.create(
            #     model=model_name_or_path, 
            #     messages=begin_messages, 
            #     max_tokens=1000, 
            #     response_format={"type": "json_object"},
            #     temperature=args.temperature,
            #     top_p=args.top_p
            # )
            result = client.chat.completions.create(
                model=model_name_or_path, 
                messages=begin_messages, 
                max_tokens=1000, 
                response_format={"type": "json_object"},
                temperature=args.temperature
            )
            # result = client.chat.completions.create(
            #     model=model_name_or_path, 
            #     messages=begin_messages, 
            #     max_tokens=1000, 
            #     response_format={"type": "json_object"}
            # )
            response = result.choices[0].message.content
            break
        except Exception as e:
            print(f"VLM failed to return a response. Error: {e} Retrying...")
            time.sleep(5)  # wait for 5 seconds before retrying
            retry += 1
            continue
    return json.loads(response)


def annotate_select_view(
        scan_id: str, 
        select_view: str, 
        targets: list[dict], 
        save_image_path: str, 
        image_path: str
    ) -> tuple[Image.Image, list[int]]:
    # Load point cloud, cameras
    scan_pc, center = load_scan_pc(scan_id, args.aligned_ply_dir)
    point_cloud = create_point_cloud(scan_pc, device="cuda")
    cameras_top, cameras_down, cameras_up, cameras_left, cameras_right = setup_camera(
        center=center,
        image_size=args.image_size,
        camera_dist=8.0,
        device="cuda",
        point_cloud=point_cloud,
        calibrate=False,
    )
    camera_dict = {"top": cameras_top, "down": cameras_down, "up": cameras_up, "left": cameras_left, "right": cameras_right}

    # Annotate bbox on selected view
    # select_view_image = view_images[select_view]
    view_img_file = os.path.join(image_path, f"_mesh_{select_view}.png")
    select_view_image = Image.open(view_img_file)
    select_view_camera = camera_dict[select_view]

    draw = ImageDraw.Draw(select_view_image, "RGBA")
    font = ImageFont.truetype("data/FreeSansBold.ttf", 20, encoding="unic")
    annotated_ids = draw_ids(draw, targets, select_view_camera, image_size=args.image_size, font=font)

    select_view_image.save(save_image_path)

    return select_view_image, annotated_ids


def calc_iou(
        box_a: list[float], 
        box_b: list[float]
    ) -> float:
    """Computes IoU of two axis aligned bboxes.
    Args:
        box_a, box_b: 6D of center and lengths
    Returns:
        iou
    """
    box_a = np.array(box_a)
    box_b = np.array(box_b)

    max_a = box_a[0:3] + box_a[3:6] / 2
    max_b = box_b[0:3] + box_b[3:6] / 2
    min_max = np.array([max_a, max_b]).min(0)

    min_a = box_a[0:3] - box_a[3:6] / 2
    min_b = box_b[0:3] - box_b[3:6] / 2
    max_min = np.array([min_a, min_b]).max(0)
    if not ((min_max > max_min).all()):
        return 0.0

    intersection = (min_max - max_min).prod()
    vol_a = box_a[3:6].prod()
    vol_b = box_b[3:6].prod()
    union = vol_a + vol_b - intersection
    return 1.0 * intersection / union


def resume_results(
        subset_tasks: list[dict], 
        output_file: str
    ) -> tuple[list[dict], list[dict]]:
    if os.path.exists(output_file):
        resumed_results = load_json(output_file)
        results = resumed_results
        resumed_query_list = [res["query"] for res in resumed_results]
        subset_tasks_remain = [task for task in subset_tasks if task["utterance"] not in resumed_query_list]
        num_tasks_remain = len(subset_tasks_remain)
        print(f"{num_tasks_remain} tasks remaining...")
    else:
        subset_tasks_remain = subset_tasks
        results = []
    return subset_tasks_remain, results


def get_max_iou(
        mask3d_bboxes: dict[str, dict], 
        gt_bbox: dict, 
        targets_ids: list[int]
    ) -> tuple[int, int, float, float]:
    # Calculate IoU for all pred bboxes and matched pred bboxes
    all_iou_dict = {}
    target_iou_dict = {}
    for box_data in mask3d_bboxes.values():
        iou_i = calc_iou(gt_bbox["bbox_3d"], box_data["bbox_3d"])
        box_id = box_data["bbox_id"]
        all_iou_dict[str(box_id)] = iou_i
        if box_id in targets_ids:
            target_iou_dict[str(box_id)] = iou_i

    max_ids_all = max(all_iou_dict, key=all_iou_dict.get)
    max_value_all = all_iou_dict[max_ids_all]

    if len(targets) > 0:
        max_ids_1 = max(target_iou_dict, key=target_iou_dict.get)
        max_value_1 = target_iou_dict[max_ids_1]
    else:
        max_ids_1 = None
        max_value_1 = 0.

    print(f"Max IoU all [\"{max_ids_all}\": {max_value_all:.2f}]")
    print(f"Max IoU targets [\"{max_ids_1}\": {max_value_1:.2f}]")

    return max_ids_1, max_ids_all, max_value_1, max_value_all


def calculate_topk_accuracy(results: list[dict]):
    correct_25 = 0
    correct_50 = 0
    num_unique = 0
    num_multiple = 0
    correct_25_unique = 0
    correct_50_unique = 0
    correct_25_multiple = 0
    correct_50_multiple = 0
    acc_25_top1 = 0
    acc_25_topk = 0

    for result in results:
        if result["is_unique"]:
            num_unique += 1
        else:
            num_multiple += 1

        if result["correct_25"][0] == "True" or result["correct_25"] == "True":
            correct_25 += 1
            if result["is_unique"]:
                correct_25_unique += 1
            else:
                correct_25_multiple += 1
        
        if result["correct_50"][0] == "True" or result["correct_50"] == "True":
            correct_50 += 1
            if result["is_unique"]:
                correct_50_unique += 1
            else:
                correct_50_multiple += 1

        if result["in_top_1"] is True or result["in_top_1"] == "True":
            acc_25_top1 += 1
        if result["in_top_k"] is True or result["in_top_k"] == "True":
            acc_25_topk += 1
    
    acc_25 = (correct_25 / len(results)) * 100
    acc_50 = (correct_50 / len(results)) * 100
    acc_25_unique = (correct_25_unique / num_unique) * 100
    acc_50_unique = (correct_50_unique / num_unique) * 100
    acc_25_multiple = (correct_25_multiple / num_multiple) * 100
    acc_50_multiple = (correct_50_multiple / num_multiple) * 100
    acc_25_top1 = (acc_25_top1 / len(results)) * 100
    acc_25_topk = (acc_25_topk / len(results)) * 100
    print(f"Overall: Acc@25: {acc_25:.2f}, Acc@50: {acc_50:.2f}")
    print(f"Unique: Acc@25: {acc_25_unique:.2f}, Acc@50: {acc_50_unique:.2f}")
    print(f"Multiple: Acc@25: {acc_25_multiple:.2f}, Acc@50: {acc_50_multiple:.2f}")
    print(f"Overall Acc@25: Top1: {acc_25_top1:.2f}, TopK: {acc_25_topk:.2f}")


def calculate_accuracy(results: list[dict]):
    # assert len(results) == 250, "Number of results is not correct!"
    correct_25 = 0
    correct_50 = 0
    num_unique = 0
    num_multiple = 0
    correct_25_unique = 0
    correct_50_unique = 0
    correct_25_multiple = 0
    correct_50_multiple = 0

    for result in results:
        if result["is_unique"]:
            num_unique += 1
        else:
            num_multiple += 1

        if result["correct_25"] == "True":
            correct_25 += 1
            if result["is_unique"]:
                correct_25_unique += 1
            else:
                correct_25_multiple += 1

        if result["correct_50"] == "True":
            correct_50 += 1
            if result["is_unique"]:
                correct_50_unique += 1
            else:
                correct_50_multiple += 1

    acc_25 = (correct_25 / len(results)) * 100
    acc_50 = (correct_50 / len(results)) * 100
    acc_25_unique = (correct_25_unique / num_unique) * 100
    acc_50_unique = (correct_50_unique / num_unique) * 100
    acc_25_multiple = (correct_25_multiple / num_multiple) * 100
    acc_50_multiple = (correct_50_multiple / num_multiple) * 100
    print(f"Number of unique: {num_unique}, number of multiple: {num_multiple}")
    print(f"Number of correct unique: {correct_25_unique}, {correct_50_unique}")
    print(f"Number of correct multiple: {correct_25_multiple}, {correct_50_multiple}")
    print(f"Overall: Acc@25: {acc_25:.2f}, Acc@50: {acc_50:.2f}")
    print(f"Unique: Acc@25: {acc_25_unique:.2f}, Acc@50: {acc_50_unique:.2f}")
    print(f"Multiple: Acc@25: {acc_25_multiple:.2f}, Acc@50: {acc_50_multiple:.2f}")


if __name__ == "__main__":
    # Seed
    seed_everything(args.seed)

    # Initialize logging
    log_file = os.path.join(args.output_dir, f"log_{args.gpt_model_type}_{args.exp_name}.txt")
    tee = Tee(log_file)
    sys.stdout = tee

    # Load reference data (250 subset)
    if args.task_type == "subset":
        subset_ref_data = pd.read_csv(args.scanrefer_250_subset)
        subset_tasks = [row for _, row in subset_ref_data.iterrows()]
        print(f"Task number: {len(subset_tasks)}")
    # TODO: full dataset
    elif args.task_type == "full":
        raise NotImplementedError("not implemented yet.")

    # Resume existing results
    output_file = os.path.join(args.output_dir, f"output_{args.gpt_model_type}_{args.exp_name}.json")
    subset_tasks, results = resume_results(subset_tasks, output_file)

    # Resume previous step results
    if args.resume_file:
        print(f"Resume results from {args.resume_file} ...")
        resumed_file = os.path.join(args.output_dir, args.resume_file)
        resumed_results = load_json(resumed_file)
    else:
        resumed_results = None

    # Initialize GPT client
    GPT_client = init_gpt_client(args.api_key)
    print(f"Using model: {args.gpt_model_type}")

    # shuffle
    random.shuffle(subset_tasks)
    # 250 (Not detected: 17, Not matched: 28, No IoU25 targets: 51 - 17 = 34)
    i = 0
    for task in tqdm(subset_tasks):
        # Initialize variables
        iou_list = []
        acc_25_list = [0.]
        acc_50_list = [0.]

        # Get task info
        scan_id = task["scan_id"]
        query_text = task["utterance"]
        gt_id = task["target_id"]
        pred_target_class = task["pred_target_class"]
        is_unique = task["is_unique_scanrefer"]
        view_image_path = os.path.join(args.view_image_dir, scan_id)

        print(f"\nSCENE_ID:{scan_id}\nquery: {query_text}\npred_target_class: {pred_target_class}")
        
        # Load mask3D pred Bbox
        mask3d_bboxes = load_bboxes(scan_id, args.bbox_dir)
        object_names = [obj["target"] for obj in mask3d_bboxes.values()]
        unique_object_names = set(object_names)
        # Load GT 3D Bbox
        gt_bboxes = load_bboxes(scan_id, args.gt_bbox_dir)
        gt_bbox = gt_bboxes[gt_id]
        gt_object_name = gt_bbox["target"]

        # Initialize camera image processor
        camera_image_processor = CameraImage(scan_id)

        # Matching query target name
        matched_targets = fuzzy_match(pred_target_class, object_names).union(
            stem_match(pred_target_class, object_names)
        )
        print(f"Matched target classes: {matched_targets}")

        mask3d_targets_all = [obj for obj in mask3d_bboxes.values()]  # all pred bboxes
        mask3d_ids_all = [obj["bbox_id"] for obj in mask3d_bboxes.values()]  # all pred bbox ids
        targets = [obj for obj in mask3d_bboxes.values() if obj["target"] in matched_targets]  # class-matched bboxes
        targets_ids = [item['bbox_id'] for item in targets]  # matched pred bbox ids
        num_matched_targets = len(targets_ids)
        print(f"Matched targets ids: {targets_ids}")
        
        # Get max IoU for all matched bboxes and all bboxes
        max_ids_1, max_ids_all, max_value_1, max_value_all = get_max_iou(mask3d_bboxes, gt_bbox, targets_ids)

        # Only one matched target, skip
        if num_matched_targets == 1:  # 38 of 250
            handle_single_matched_target(
                scan_id=scan_id,
                query_text=query_text,
                matched_targets=matched_targets,
                is_unique=is_unique,
                gt_id=gt_id,
                targets=targets,
                gt_bbox=gt_bbox,
                targets_ids=targets_ids,
                max_value_1=max_value_1,
                n_topk=args.n_topk,
                output_file=output_file,
                results=results,
            )
            continue

        if max_value_all < 0.25:  # Not successfully detected by Mask3D, skip directly
            handle_detection_failure(
                scan_id=scan_id,
                query_text=query_text,
                matched_targets=matched_targets,
                is_unique=is_unique,
                gt_id=gt_id,
                gt_bbox=gt_bbox,
                iou_list=iou_list,
                n_topk=args.n_topk,
                output_file=output_file,
                results=results,
            )
            print("Skip false detection sample ...")
            continue
        
        # Resume previous step results
        if resumed_results:
            matching_item = next((item for item in resumed_results if item['query'] == query_text), None)
            in_top_k = matching_item["in_top_k"]
            select_view = matching_item["view"]
            predicted_ids = matching_item["predicted_id"]
            if not in_top_k or in_top_k == "False":
                print(f"Skip object id prediction, not in top-k")
                results.append(
                    {
                        "scene_id": scan_id,
                        "query": query_text,
                        "matched_targets": str(matched_targets),
                        "is_unique": is_unique,
                        "gt_id": gt_id,
                        "pred_bbox": None,
                        "gt_bbox": gt_bbox["bbox_3d"],
                        "view": select_view,
                        "predicted_id": None,
                        "iou": 0.,
                        "correct_25": "False",
                        "correct_50": "False"
                    }
                )
                # save results
                save_to_file(output_file, json.dumps(results, indent=4))
                continue

        # 1. --------------------------- View Selection ---------------------------
        if args.view_selection:
            stitched_view_image = Image.open(os.path.join(view_image_path, "stitched_horizontal.png"))
            base64_images = [encode_PIL_image_to_base64(
                resize_image(stitched_view_image, max_size=5160))  # 4096
            ]
            begin_messages = format_gpt_input_view(
                query_text, 
                pred_target_class, 
                base64_images
            )
            response_view = get_gpt_response(
                GPT_client, 
                begin_messages, 
                args.gpt_model_type
            )
            view_reasoning, select_view = response_view["reasoning"], response_view["view"]
            assert select_view is not None, "Select view is None!"
            select_view = all_views[int(select_view)]
            
            print(f"[GPT] select view: {select_view}")
            begin_messages.append({"role": "assistant", "content": view_reasoning})
        elif resumed_results:
            select_view = matching_item["view"]
            begin_messages = []
            print(f"Use previous predicted view: {select_view}")
        else:
            select_view = "top"  # use top view if skip view selection
            begin_messages = []
            print(f"Use Top view by default")
        # ------------------------------------------------------------------------

        # 2. --------------------------- Top-K Object ID ---------------------------
        if args.top_k_prediction:
            # TODO: add additional crop matched targets
            if num_matched_targets > 0 and num_matched_targets <= args.n_topk:
                # Skip top-k selection
                print(f"Use all {num_matched_targets} bboxes {targets_ids}")
                select_object_id = targets_ids
            else:
                # Get selected view image to annotate
                bbox_annotation_image_path = f"{view_image_path}/annotate_box_{select_view}_{pred_target_class}.png"

                if num_matched_targets == 0:
                    # Matching target name failed
                    print(f"Matching failed! Using crop images for object id selection ...")
                    # Annotate all bboxes on select view
                    select_view_image, annotated_ids = annotate_select_view(
                        scan_id=scan_id, 
                        select_view=select_view, 
                        targets=mask3d_targets_all,  # annotated all bboxes
                        save_image_path=bbox_annotation_image_path, 
                        image_path=view_image_path
                    )
                    base64_image_object_id = [encode_PIL_image_to_base64(
                        resize_image(select_view_image, max_size=1024))
                    ]
                    # Get all crop object images
                    bbox_crop_img = Image.open(os.path.join(args.bbox_crop_dir, f"{scan_id}.jpg"))
                    base64_image_object_id.append(
                        encode_PIL_image_to_base64(bbox_crop_img)
                    )
                    begin_messages_object_id = format_gpt_input_topk_id_crop(
                        query_text, 
                        base64_image_object_id
                    )
                    response_id = get_gpt_response(
                        GPT_client, 
                        begin_messages_object_id, 
                        args.gpt_model_type
                    )
                else:
                    # Annotate matched bbox on select view
                    select_view_image, annotated_ids = annotate_select_view(
                        scan_id=scan_id, 
                        select_view=select_view, 
                        targets=targets, 
                        save_image_path=bbox_annotation_image_path, 
                        image_path=view_image_path
                    )
                    base64_image_object_id = [encode_PIL_image_to_base64(
                        resize_image(select_view_image, max_size=1024))
                    ]
                    begin_messages_object_id = format_gpt_input_topk_id(
                        pred_target_class, 
                        query_text, 
                        base64_image_object_id, 
                        annotated_ids
                    )
                    response_id = get_gpt_response(
                        GPT_client, 
                        begin_messages_object_id, 
                        args.gpt_model_type
                    )
                    
                reasoning_id, select_object_id = response_id["reasoning"], response_id["object_id"]
                print(f"[GPT] select object: {select_object_id}")

                begin_messages = filter_image_url(begin_messages)
                begin_messages.append(begin_messages_object_id[1])
                begin_messages.append({"role": "assistant", "content": reasoning_id})
            
            # evaluate top-k
            iou_list = []
            pred_bbox_list = []
            for obj_id in select_object_id:
                try:
                    pred_bbox = mask3d_bboxes[int(obj_id)]
                    iou = calc_iou(gt_bbox["bbox_3d"], pred_bbox["bbox_3d"])
                except:
                    pred_bbox = None
                    iou = 0.
                iou_list.append(iou)
                pred_bbox_list.append(pred_bbox)
                
            formatted_iou_list = [f"{x:.2f}" for x in iou_list]
            acc_25_list = [str(x > 0.25) for x in iou_list]
            acc_50_list = [str(x > 0.5) for x in iou_list]
            print(f"Top-{args.n_topk} IoU is {formatted_iou_list}  ", acc_25_list)

            if acc_25_list[0] == "True":
                in_top_1 = True
            else:
                in_top_1 = False

            if "True" in acc_25_list:
                in_top_k = True
            else:
                in_top_k = False

            print(f"in_top_1: {in_top_1}, in_top_k: {in_top_k}")

            results.append(
                {
                    "scene_id": scan_id,
                    "query": query_text,
                    "matched_targets": str(matched_targets),
                    "is_unique": is_unique,
                    "gt_id": gt_id,
                    "pred_bbox": pred_bbox["bbox_3d"] if pred_bbox else None,
                    "gt_bbox": gt_bbox["bbox_3d"],
                    "view": select_view,
                    "predicted_id": select_object_id,
                    "iou": iou_list,
                    "correct_25": acc_25_list,
                    "correct_50": acc_50_list,
                    "n_top_k": args.n_topk,
                    "in_top_1": str(in_top_1),
                    "in_top_k": str(in_top_k),
                }
            )

            # save results
            save_to_file(output_file, json.dumps(results, indent=4))
            query_name = query_text[:40]

            # save gpt messages
            message_file = os.path.join(args.output_dir, f"{args.gpt_model_type}_messages_{args.exp_name}/{scan_id}/{query_name}.json")
            save_to_file(message_file, json.dumps(filter_image_url(begin_messages), indent=4))
        else:
            print(f"Skip top-k prediction")

        # 3. ------------------------- Object ID Selection -------------------------
        if args.object_id_prediction:
            # Merge predicted ids to targets
            if predicted_ids is not None:
                predicted_ids = [x for x in predicted_ids if x in mask3d_ids_all]
                for pred_id in predicted_ids:
                    if pred_id in mask3d_ids_all and pred_id not in targets_ids:
                        targets_ids.append(pred_id)
                        targets.append(mask3d_bboxes[pred_id])
                        print(f"* append new target: {pred_id}")
            else:
                print(f"predicted_ids is None")
                # Randomly select 5 IDs from either targets_ids or object_ids
                if targets_ids:
                    # If targets_ids is not empty, randomly select 5 IDs from it
                    # If there are fewer than 5 IDs, select all of them
                    num_to_select = min(args.n_topk, len(targets_ids))
                    predicted_ids = random.sample(targets_ids, num_to_select)
                    print(f"Randomly selected {num_to_select} IDs from targets_ids: {predicted_ids}")
                else:
                    # If targets_ids is empty, randomly select 5 IDs from object_ids
                    print(f"targets_ids is empty, randomly select {args.n_topk} IDs from object_ids")
                    num_to_select = min(args.n_topk, len(mask3d_ids_all))
                    predicted_ids = random.sample(mask3d_ids_all, num_to_select)
                    print(f"Randomly selected {num_to_select} IDs from object_ids: {predicted_ids}")

            # Get annotated 3D box on selected view
            bbox_annotation_image_path = f"{view_image_path}/annotate_box_{select_view}_{pred_target_class}.png"
            # Annotate bbox on select view
            select_view_image, annotated_ids = annotate_select_view(
                scan_id=scan_id, 
                select_view=select_view, 
                targets=targets,  # targets / mask3d_targets_all
                save_image_path=bbox_annotation_image_path, 
                image_path=view_image_path
            )
            print(f"annotated_ids: {annotated_ids}")
            
            # Remove repeat ids and keep order
            predicted_ids = list(dict.fromkeys(predicted_ids))
            print(f"pred ids: {predicted_ids}")
            if len(annotated_ids) <= 5:
                if len(annotated_ids) > len(predicted_ids):
                    print(f"use all annotated ids: {annotated_ids}")
                    predicted_ids = annotated_ids

            # Get corresponding camera images
            target_ids_all = [predicted_ids[i:] + predicted_ids[:i] for i in range(len(predicted_ids))]
            all_camera_images = []
            for tgt_id in target_ids_all:
                bbox_3d = [mask3d_bboxes[id]["bbox_3d"] for id in tgt_id if id in targets_ids]
                
                assert len(bbox_3d) == len(tgt_id), f"bbox_3d: {len(bbox_3d)}, tgt_id: {tgt_id}"
                
                camera_image = camera_image_processor.get_annotate_image(
                    bbox3d=bbox_3d,
                    bbox_id=tgt_id
                )
                if camera_image is not None:
                    all_camera_images.append(camera_image)

            # use original size image
            base64_image_object_id = [encode_PIL_image_to_base64(img) for img in all_camera_images]

            # base64_image_object_id = [encode_PIL_image_to_base64(
            #     resize_image(img, max_size=968)
            #     ) for img in all_camera_images
            # ]  # 512  camera_image

            base64_image_object_id.append(encode_PIL_image_to_base64(
                resize_image(select_view_image, max_size=1024))
            )  # 1024  view_image

            begin_messages_object_id = format_gpt_input_object_id(
                pred_target_class, 
                query_text, 
                base64_image_object_id, 
                targets_ids  ## annotated_ids -> predicted_ids -> targets_ids
            )
            begin_messages.append(begin_messages_object_id[1])

            response_id = get_gpt_response(
                GPT_client, 
                begin_messages_object_id, 
                args.gpt_model_type
            )
            reasoning_id, select_object_id = response_id["reasoning"], response_id["object_id"]
            print(f"[GPT] select object: {select_object_id}")

            begin_messages.append({"role": "assistant", "content": reasoning_id})
            
            try:
                pred_bbox = mask3d_bboxes[int(select_object_id)]
                iou = calc_iou(gt_bbox["bbox_3d"], pred_bbox["bbox_3d"])
            except:
                pred_bbox = None
                iou = 0.

            print(f"IoU is {iou:.2f}  ", str(iou > 0.25))

            results.append(
                {
                    "scene_id": scan_id,
                    "query": query_text,
                    "matched_targets": str(matched_targets),
                    "is_unique": is_unique,
                    "gt_id": gt_id,
                    "pred_bbox": pred_bbox["bbox_3d"] if pred_bbox else None,
                    "gt_bbox": gt_bbox["bbox_3d"],
                    "view": select_view,
                    "predicted_id": select_object_id,
                    "iou": iou,
                    "correct_25": str(iou > 0.25),
                    "correct_50": str(iou > 0.5),
                }
            )

            # save results
            save_to_file(output_file, json.dumps(results, indent=4))
            query_name = query_text[:40]

            # save gpt messages
            message_file = os.path.join(args.output_dir, f"{args.gpt_model_type}_messages_{args.exp_name}/{scan_id}/{query_name}.json")
            save_to_file(message_file, json.dumps(filter_image_url(begin_messages), indent=4))


    # Calculate accuracy
    if args.top_k_prediction:
        calculate_topk_accuracy(results)
    if args.object_id_prediction:
        calculate_accuracy(results)
