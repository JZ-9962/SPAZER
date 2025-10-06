#!/bin/bash
set -euo pipefail

# Defaults extracted from Python (NR3D)
SCAN_ID_FILE="data/scannet/scannetv2_val.txt"
ANNO_FILE="datasets/ScanNet/ScanRefer/ScanRefer_filtered_val.json"
VIEW_IMAGE_DIR="data/projection_img/scannet"
SCANREFER_250_SUBSET="projects/VLM-Grounder/outputs/query_analysis/scanrefer_250.csv"
NR3D_250_SUBSET="projects/VLM-Grounder/outputs/query_analysis/nr3d_250.csv"
ALIGNED_PLY_DIR="data/scannet/global_aligned_mesh_clean_2_filtered"
BBOX_DIR="object_lookup_table/nr3d/pred"
GT_BBOX_DIR="object_lookup_table/nr3d/gt"
OUTPUT_DIR="data/output_nr3d"
BBOX_CROP_DIR="data/scannet/bbox_crop_all"
BBOX_CROP_GT_DIR="data/scannet/bbox_crop_all_gt"
CAMERA_VIEW_SAVE_DIR="data/scannet/cam_img_infer_nr3d"
TASK_TYPE="subset"
EXP_NAME="nr3d"
N_TOPK=5
SEED=42
GPT_MODEL_TYPE="gpt-4o"
API_KEY=""
TEMPERATURE=0.2
TOP_P=1.0
IMAGE_SIZE=1024

# Step 1: view selection and top-k prediction
python nr3d_process.py \
  --scan_id_file "$SCAN_ID_FILE" \
  --anno_file "$ANNO_FILE" \
  --view_image_dir "$VIEW_IMAGE_DIR" \
  --scanrefer_250_subset "$SCANREFER_250_SUBSET" \
  --nr3d_250_subset "$NR3D_250_SUBSET" \
  --aligned_ply_dir "$ALIGNED_PLY_DIR" \
  --bbox_dir "$BBOX_DIR" \
  --gt_bbox_dir "$GT_BBOX_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --bbox_crop_dir "$BBOX_CROP_DIR" \
  --bbox_crop_gt_dir "$BBOX_CROP_GT_DIR" \
  --camera_view_save_dir "$CAMERA_VIEW_SAVE_DIR" \
  --task_type "$TASK_TYPE" \
  --exp_name "$EXP_NAME" \
  --n_topk $N_TOPK \
  --view_selection \
  --top_k_prediction \
  --seed $SEED \
  --gpt_model_type "$GPT_MODEL_TYPE" \
  --api_key "$API_KEY" \
  --temperature $TEMPERATURE \
  --top_p $TOP_P \
  --image_size $IMAGE_SIZE

# set resume file from previous output
RESUME_FILE="output_${GPT_MODEL_TYPE}_${EXP_NAME}.json"

# Step 2: object id prediction
python nr3d_process.py \
  --scan_id_file "$SCAN_ID_FILE" \
  --anno_file "$ANNO_FILE" \
  --view_image_dir "$VIEW_IMAGE_DIR" \
  --scanrefer_250_subset "$SCANREFER_250_SUBSET" \
  --nr3d_250_subset "$NR3D_250_SUBSET" \
  --aligned_ply_dir "$ALIGNED_PLY_DIR" \
  --bbox_dir "$BBOX_DIR" \
  --gt_bbox_dir "$GT_BBOX_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --bbox_crop_dir "$BBOX_CROP_DIR" \
  --bbox_crop_gt_dir "$BBOX_CROP_GT_DIR" \
  --camera_view_save_dir "$CAMERA_VIEW_SAVE_DIR" \
  --task_type "$TASK_TYPE" \
  --exp_name "$EXP_NAME" \
  --n_topk $N_TOPK \
  --object_id_prediction \
  --seed $SEED \
  --resume_file "$RESUME_FILE" \
  --gpt_model_type "$GPT_MODEL_TYPE" \
  --api_key "$API_KEY" \
  --temperature $TEMPERATURE \
  --top_p $TOP_P \
  --image_size $IMAGE_SIZE


