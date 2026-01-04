#!/usr/bin/env python
"""Download script for local LLM model (MistralLite GGUF)"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from huggingface_hub import hf_hub_download
from src.logger import logger


def download_mistrallite_model(
    repo_id: str = "TheBloke/MistralLite-7B-GGUF",
    filename: str = "mistrallite.Q4_K_M.gguf",
    local_dir: str = "models",
) -> str:
    """
    Download the quantized MistralLite GGUF model from Hugging Face.
    
    Args:
        repo_id: Hugging Face repository ID
        filename: Model filename to download
        local_dir: Local directory to save the model
        
    Returns:
        Path to the downloaded model file
    """
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
    local_dir: str = "models",
    filename: str = "mistrallite.Q4_K_M.gguf",
) -> bool:
    """Check if the model file already exists."""
    model_path = Path(local_dir) / filename
    return model_path.exists()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download MistralLite GGUF model")
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
    args = parser.parse_args()
    
    if check_model_exists(args.model_dir) and not args.force:
        print(f"Model already exists in {args.model_dir}/")
        print("Use --force to re-download")
    else:
        file_path = download_mistrallite_model(local_dir=args.model_dir)
        print(f"Downloaded model to: {file_path}")
