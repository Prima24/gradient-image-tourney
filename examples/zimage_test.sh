#!/bin/bash

TASK_ID="6659c5ab-92bd-4ef9-97b5-2abe6c54f49d"
MODEL="gradients-io-tournaments/Z-Image-Turbo"
DATASET_ZIP="https://s3.eu-central-003.backblazeb2.com/gradients-validator/57b77e009a07491e_train_data.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=00362e8d6b742200000000002%2F20260109%2Feu-central-003%2Fs3%2Faws4_request&X-Amz-Date=20260109T113206Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=ea9d19545ec6080a4e9871aa7816a34982d86ceabb79caeea06127653c331e49"
MODEL_TYPE="z-image"
EXPECTED_REPO_NAME="test_zimage-1"

HUGGINGFACE_USERNAME=""
HUGGINGFACE_TOKEN=""
LOCAL_FOLDER="/app/checkpoints/$TASK_ID/$EXPECTED_REPO_NAME"

CHECKPOINTS_DIR="$(pwd)/secure_checkpoints"
OUTPUTS_DIR="$(pwd)/outputs"
mkdir -p "$CHECKPOINTS_DIR"
chmod 700 "$CHECKPOINTS_DIR"
mkdir -p "$OUTPUTS_DIR"
chmod 700 "$OUTPUTS_DIR"

echo "Downloading model and dataset..."
docker run --rm   --volume "$CHECKPOINTS_DIR:/cache:rw"   --name downloader-image   trainer-downloader   --task-id "$TASK_ID"   --model "$MODEL"   --dataset "$DATASET_ZIP"   --task-type "ImageTask"   --model-type "$MODEL_TYPE" 

echo "Starting image training..."
docker run --rm --gpus all   --security-opt=no-new-privileges   --cap-drop=ALL   --memory=32g   --cpus=8   --network none   --env TRANSFORMERS_CACHE=/cache/hf_cache   --volume "$CHECKPOINTS_DIR:/cache:rw"   --volume "$OUTPUTS_DIR:/app/checkpoints/:rw"   --name image-trainer-example   standalone-image-toolkit-trainer   --task-id "$TASK_ID"   --model "$MODEL"   --dataset-zip "$DATASET_ZIP"   --model-type "$MODEL_TYPE"   --expected-repo-name "$EXPECTED_REPO_NAME"   --hours-to-complete 1

echo "Uploading model to HuggingFace..."
docker run --rm --gpus all   --volume "$OUTPUTS_DIR:/app/checkpoints/:rw"   --env HUGGINGFACE_TOKEN="$HUGGINGFACE_TOKEN"   --env HUGGINGFACE_USERNAME="$HUGGINGFACE_USERNAME"   --env TASK_ID="$TASK_ID"   --env EXPECTED_REPO_NAME="$EXPECTED_REPO_NAME"   --env LOCAL_FOLDER="$LOCAL_FOLDER"   --env HF_REPO_SUBFOLDER="checkpoints"   --name hf-uploader   hf-uploader
