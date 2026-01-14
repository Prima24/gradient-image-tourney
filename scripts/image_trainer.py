#!/usr/bin/env python3
"""
image-max   
"""

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import re
import time
import yaml
import toml


# Add project root to python path to import modules
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

import core.constants as cst
import trainer.constants as train_cst
import trainer.utils.training_paths as train_paths
from core.config.config_handler import save_config, save_config_toml
from core.dataset.prepare_diffusion_dataset import prepare_dataset
from core.models.utility_models import ImageModelType



def get_model_path(path: str) -> str:
    if os.path.isdir(path):
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        if len(files) == 1 and files[0].endswith(".safetensors"):
            return os.path.join(path, files[0])
    return path
def merge_model_config(default_config: dict, model_config: dict) -> dict:
    """Merge default config with model-specific overrides."""
    merged = {}

    if isinstance(default_config, dict):
        merged.update(default_config)

    if isinstance(model_config, dict):
        merged.update(model_config)

    return merged if merged else None
def get_config_for_model(lrs_config: dict, model_name: str) -> dict:
    """Get configuration overrides based on model name."""
    if not isinstance(lrs_config, dict):
        return None

    data = lrs_config.get("data")
    default_config = lrs_config.get("default", {})

    if isinstance(data, dict) and model_name in data:
        return merge_model_config(default_config, data.get(model_name))

    if default_config:
        return default_config

    return None

def load_lrs_config(model_type: str, is_style: bool) -> dict:
    """Load the appropriate LRS configuration based on model type and training type"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(script_dir, "lrs")

    if model_type == "flux":
        config_file = os.path.join(config_dir, "flux.json")
    elif is_style:
        config_file = os.path.join(config_dir, "style_config.json")
    else:
        config_file = os.path.join(config_dir, "person_config.json")
    
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load LRS config from {config_file}: {e}", flush=True)
        return None


def create_config(task_id, model_path, model_name, model_type, expected_repo_name, trigger_word: str | None = None):
    """Get the training data directory"""
    train_data_dir = train_paths.get_image_training_images_dir(task_id)

    """Create the diffusion config file"""
    config_template_path, is_style = train_paths.get_image_training_config_template_path(model_type, train_data_dir)

    is_ai_toolkit = model_type in [ImageModelType.Z_IMAGE.value, ImageModelType.QWEN_IMAGE.value]
    
    if is_ai_toolkit:
        with open(config_template_path, "r") as file:
            config = yaml.safe_load(file)
        if 'config' in config and 'process' in config['config']:
            for process in config['config']['process']:
                if 'model' in process:
                    process['model']['name_or_path'] = model_path
                    if 'training_folder' in process:
                        output_dir = train_paths.get_checkpoints_output_path(task_id, expected_repo_name or "output")
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir, exist_ok=True)
                        process['training_folder'] = output_dir
                
                if 'datasets' in process:
                    for dataset in process['datasets']:
                        dataset['folder_path'] = train_data_dir

                if trigger_word:
                    process['trigger_word'] = trigger_word
        
        config_path = os.path.join(train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, f"{task_id}.yaml")
        save_config(config, config_path)
        print(f"Created ai-toolkit config at {config_path}", flush=True)
        return config_path
    else:
        with open(config_template_path, "r") as file:
            config = toml.load(file)

        lrs_config = load_lrs_config(model_type, is_style)
        if lrs_config:
            model_hash = hash_model(model_name)
            lrs_settings = get_config_for_model(lrs_config, model_hash)

            if lrs_settings:
                for optional_key in [
                    "max_grad_norm",
                    "prior_loss_weight",
                    "max_train_epochs",
                    "train_batch_size",
                    "optimizer_args",
                    "unet_lr",
                    "text_encoder_lr",
                    "noise_offset",
                    "min_snr_gamma",
                    "seed",
                    "lr_warmup_steps",
                    "loss_type",
                    "huber_c",
                    "huber_schedule",
                    "optimizer_type",
                    "network_dim",
                    "network_alpha",
                    "network_args",
                    "lr_scheduler_args",
                    "clip_skip",
                    "caption_dropout_rate",
                    "scale_weight_norms",
                    "max_train_steps"
                ]:
                    if optional_key in lrs_settings:
                        config[optional_key] = lrs_settings[optional_key]
            else:
                print(f"Warning: No LRS configuration found for model '{model_name}'", flush=True)
        else:
            print("Warning: Could not load LRS configuration, using default values", flush=True)

        # Update config
        network_config_person = {
            # ANIME & ILLUSTRATION (ID: 9)
            "zenless-lab/sdxl-aam-xl-anime-mix": 9,
            "John6666/nova-anime-xl-pony-v5-sdxl": 9,
            "zenless-lab/sdxl-anima-pencil-xl-v5": 9,
            "cagliostrolab/animagine-xl-4.0": 9,
            "zenless-lab/sdxl-anything-xl": 9,
            "OnomaAIResearch/Illustrious-xl-early-release-v0": 9,
            "John6666/hassaku-xl-illustrious-v10style-sdxl": 9,
            "KBlueLeaf/Kohaku-XL-Zeta": 9,
            "zenless-lab/sdxl-blue-pencil-xl-v7": 9,

            # PHOTOREALISTIC (ID: 69)
            "misri/leosamsHelloworldXL_helloworldXL70": 69,
            "GraydientPlatformAPI/albedobase2-xl": 69,
            "femboysLover/RealisticStockPhoto-fp16": 69,
            "ifmain/UltraReal_Fine-Tune": 69,
            "GraydientPlatformAPI/realism-engine2-xl": 69,
            "SG161222/RealVisXL_V4.0": 69,

            # ARTISTIC / 2.5D / GENERALIST (ID: 99)
            "dataautogpt3/CALAMITY": 99,
            "recoilme/colorfulxl": 99,
            "dataautogpt3/ProteusV0.5": 99,
            "fluently/Fluently-XL-Final": 99,
            "stabilityai/stable-diffusion-xl-base-1.0": 99,
            "openart-custom/DynaVisionXL": 99,
            "Lykon/dreamshaper-xl-1-0": 99,
            "dataautogpt3/ProteusSigma": 99,
            "mann-e/Mann-E_Dreams": 99,
            "Corcelio/mobius": 99,
            "ehristoforu/Visionix-alpha": 99,
            "Lykon/art-diffusion-xl-0.9": 99,
            "stablediffusionapi/omnium-sdxl": 99,
            "GHArt/Lah_Mysterious_SDXL_V4.0_xl_fp16": 99,
            "misri/zavychromaxl_v90": 99,
            "stablediffusionapi/protovision-xl-v6.6": 99,
            "dataautogpt3/TempestV0.1": 99,
            "bghira/terminus-xl-velocity-v2": 99
        }

        network_config_style = {
           # ANIME & ILLUSTRATION (ID: 8) - Same models, different ID for Style tasks
            "zenless-lab/sdxl-aam-xl-anime-mix": 8,
            "John6666/nova-anime-xl-pony-v5-sdxl": 8,
            "zenless-lab/sdxl-anima-pencil-xl-v5": 8,
            "cagliostrolab/animagine-xl-4.0": 8,
            "zenless-lab/sdxl-anything-xl": 8,
            "OnomaAIResearch/Illustrious-xl-early-release-v0": 8,
            "John6666/hassaku-xl-illustrious-v10style-sdxl": 8,
            "KBlueLeaf/Kohaku-XL-Zeta": 8,
            "zenless-lab/sdxl-blue-pencil-xl-v7": 8,

            # PHOTOREALISTIC (ID: 78)
            "misri/leosamsHelloworldXL_helloworldXL70": 78,
            "GraydientPlatformAPI/albedobase2-xl": 78,
            "femboysLover/RealisticStockPhoto-fp16": 78,
            "ifmain/UltraReal_Fine-Tune": 78,
            "GraydientPlatformAPI/realism-engine2-xl": 78,
            "SG161222/RealVisXL_V4.0": 78,

            # ARTISTIC / 2.5D / GENERALIST (ID: 118)
            "dataautogpt3/CALAMITY": 118,
            "recoilme/colorfulxl": 118,
            "dataautogpt3/ProteusV0.5": 118,
            "fluently/Fluently-XL-Final": 118,
            "stabilityai/stable-diffusion-xl-base-1.0": 118,
            "openart-custom/DynaVisionXL": 118,
            "Lykon/dreamshaper-xl-1-0": 118,
            "dataautogpt3/ProteusSigma": 118,
            "mann-e/Mann-E_Dreams": 118,
            "Corcelio/mobius": 118,
            "ehristoforu/Visionix-alpha": 118,
            "Lykon/art-diffusion-xl-0.9": 118,
            "stablediffusionapi/omnium-sdxl": 118,
            "GHArt/Lah_Mysterious_SDXL_V4.0_xl_fp16": 118,
            "misri/zavychromaxl_v90": 118,
            "stablediffusionapi/protovision-xl-v6.6": 118,
            "dataautogpt3/TempestV0.1": 118,
            "bghira/terminus-xl-velocity-v2": 118
        }

        config_mapping_person = {
            9: {
                "network_dim": 128,
                "network_alpha": 64,
                "network_args": ["conv_dim=16", "conv_alpha=8", "dropout=0.005"]
            },
            69: {
                "network_dim": 128,
                "network_alpha": 64,
                "network_args": ["conv_dim=16", "conv_alpha=8", "dropout=0.005"]
            },
            99: {
                "network_dim": 64,
                "network_alpha": 32,
                "network_args": ["conv_dim=8", "conv_alpha=4", "dropout=0.005"]
            },
        }

        config_mapping_style = {
            8: {
                "network_dim": 192,
                "network_alpha": 96,
                "network_args": ["conv_dim=32", "conv_alpha=16", "dropout=0.1"]
            },
            78: {
                "network_dim": 128,
                "network_alpha": 64,
                "network_args": ["conv_dim=16", "conv_alpha=8", "dropout=0.1"]
            },
            118: {
                "network_dim": 128,
                "network_alpha": 64,
                "network_args": ["conv_dim=16", "conv_alpha=8", "dropout=0.1"]
            },
        }

        config["pretrained_model_name_or_path"] = model_path
        config["train_data_dir"] = train_data_dir
        output_dir = train_paths.get_checkpoints_output_path(task_id, expected_repo_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        config["output_dir"] = output_dir

        if model_type == "sdxl":
            if is_style:
                network_config = config_mapping_style[network_config_style[model_name]]
            else:
                network_config = config_mapping_person[network_config_person[model_name]]

            # Count images to adjust dropout dynamically
            num_images = 0
            if os.path.exists(train_data_dir):
                for root, dirs, files in os.walk(train_data_dir):
                    num_images += len([f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))])
            
            print(f"Detected {num_images} images for task {task_id}", flush=True)
            
            config["network_dim"] = network_config["network_dim"]
            config["network_alpha"] = network_config["network_alpha"]
            config["network_args"] = network_config["network_args"]

        config_path = os.path.join(train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, f"{task_id}.toml")
        save_config_toml(config, config_path)
        print(f"config is {config}", flush=True)
        print(f"Created config at {config_path}", flush=True)
        return config_path


def run_training(model_type, config_path):
    print(f"Starting training with config: {config_path}", flush=True)

    is_ai_toolkit = model_type in [ImageModelType.Z_IMAGE.value, ImageModelType.QWEN_IMAGE.value]
    
    if is_ai_toolkit:
        training_command = [
            "python3",
            "/app/ai-toolkit/run.py",
            config_path
        ]
    else:
        if model_type == "sdxl":
            training_command = [
                "accelerate", "launch",
                "--dynamo_backend", "no",
                "--dynamo_mode", "default",
            "--mixed_precision", "bf16",
            "--num_processes", "1",
            "--num_machines", "1",
            "--num_cpu_threads_per_process", "2",
            f"/app/sd-script/{model_type}_train_network.py",
            "--config_file", config_path
        ]
        elif model_type == "flux":
            training_command = [
                "accelerate", "launch",
                "--dynamo_backend", "no",
                "--dynamo_mode", "default",
                "--mixed_precision", "bf16",
                "--num_processes", "1",
                "--num_machines", "1",
                "--num_cpu_threads_per_process", "2",
                f"/app/sd-scripts/{model_type}_train_network.py",
                "--config_file", config_path
            ]

    try:
        print("Starting training subprocess...\n", flush=True)
        process = subprocess.Popen(
            training_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        for line in process.stdout:
            print(line, end="", flush=True)

        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, training_command)

        print("Training subprocess completed successfully.", flush=True)

    except subprocess.CalledProcessError as e:
        print("Training subprocess failed!", flush=True)
        print(f"Exit Code: {e.returncode}", flush=True)
        print(f"Command: {' '.join(e.cmd) if isinstance(e.cmd, list) else e.cmd}", flush=True)
        raise RuntimeError(f"Training subprocess failed with exit code {e.returncode}")

def hash_model(model: str) -> str:
    model_bytes = model.encode('utf-8')
    hashed = hashlib.sha256(model_bytes).hexdigest()
    return hashed 

async def main():
    print("---STARTING IMAGE TRAINING SCRIPT---", flush=True)
    parser = argparse.ArgumentParser(description="Image Model Training Script")
    parser.add_argument("--task-id", required=True, help="Task ID")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--dataset-zip", required=True, help="Link to dataset zip file")
    parser.add_argument("--model-type", required=True, choices=["sdxl", "flux", "qwen-image", "z-image"], help="Model type")
    parser.add_argument("--expected-repo-name", help="Expected repository name")
    parser.add_argument("--trigger-word", help="Trigger word for the training")
    parser.add_argument("--hours-to-complete", type=float, required=True, help="Number of hours to complete the task")
    args = parser.parse_args()

    os.makedirs(train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, exist_ok=True)
    os.makedirs(train_cst.IMAGE_CONTAINER_IMAGES_PATH, exist_ok=True)

    model_path = train_paths.get_image_base_model_path(args.model)

    print("Preparing dataset...", flush=True)

    prepare_dataset(
        training_images_zip_path=train_paths.get_image_training_zip_save_path(args.task_id),
        training_images_repeat=cst.DIFFUSION_SDXL_REPEATS if args.model_type == ImageModelType.SDXL.value else cst.DIFFUSION_FLUX_REPEATS,
        instance_prompt=cst.DIFFUSION_DEFAULT_INSTANCE_PROMPT,
        class_prompt=cst.DIFFUSION_DEFAULT_CLASS_PROMPT,
        job_id=args.task_id,
        output_dir=train_cst.IMAGE_CONTAINER_IMAGES_PATH
    )

    config_path = create_config(
        args.task_id,
        model_path,
        args.model,
        args.model_type,
        args.expected_repo_name,
        args.trigger_word,
    )

    run_training(args.model_type, config_path)


if __name__ == "__main__":
    asyncio.run(main())
