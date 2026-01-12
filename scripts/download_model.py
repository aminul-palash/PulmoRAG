#!/usr/bin/env python
"""Download script for local LLM model (LLaMA-3.1-8B-Instruct GGUF)"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from huggingface_hub import hf_hub_download
from src.logger import logger


# Model configurations
MODEL_CONFIGS = {
    "llama-3.1-8b": {
        "repo_id": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        "filename": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    },
    "mistrallite": {
        "repo_id": "TheBloke/MistralLite-7B-GGUF",
        "filename": "mistrallite.Q4_K_M.gguf",
    },
}

DEFAULT_MODEL = "llama-3.1-8b"


def download_model(
    model_name: str = DEFAULT_MODEL,
    local_dir: str = "models",
) -> str:
    """
    Download a quantized GGUF model from Hugging Face.
    
    Args:
        model_name: Model name (llama-3.1-8b or mistrallite)
        local_dir: Local directory to save the model
        
    Returns:
        Path to the downloaded model file
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")
    
    config = MODEL_CONFIGS[model_name]
    repo_id = config["repo_id"]
    filename = config["filename"]
    
    logger.info(f"Downloading model {filename} from {repo_id}...")
    
    # Create models directory if it doesn't exist
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
        )
        logger.info(f"Successfully downloaded model to: {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        raise


def check_model_exists(
    model_name: str = DEFAULT_MODEL,
    local_dir: str = "models",
) -> bool:
    """Check if the model file already exists."""
    if model_name not in MODEL_CONFIGS:
        return False
    filename = MODEL_CONFIGS[model_name]["filename"]
    model_path = Path(local_dir) / filename
    return model_path.exists()


def get_model_path(
    model_name: str = DEFAULT_MODEL,
    local_dir: str = "models",
) -> Path:
    """Get the path to the model file."""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}")
    filename = MODEL_CONFIGS[model_name]["filename"]
    return Path(local_dir) / filename


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download GGUF model for local inference")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=list(MODEL_CONFIGS.keys()),
        help=f"Model to download (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--model-dir",
        default="models",
        help="Directory to save the model (default: models)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if model exists",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models",
    )
    args = parser.parse_args()
    
    if args.list:
        print("Available models:")
        for name, config in MODEL_CONFIGS.items():
            marker = " (default)" if name == DEFAULT_MODEL else ""
            print(f"  - {name}{marker}: {config['filename']}")
        sys.exit(0)
    
    if check_model_exists(args.model, args.model_dir) and not args.force:
        model_path = get_model_path(args.model, args.model_dir)
        print(f"Model already exists: {model_path}")
        print("Use --force to re-download")
    else:
        file_path = download_model(model_name=args.model, local_dir=args.model_dir)
        print(f"Downloaded model to: {file_path}")
