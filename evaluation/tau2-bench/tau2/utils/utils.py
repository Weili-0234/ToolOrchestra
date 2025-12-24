import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
import os
from pathlib import Path
from deepdiff import DeepDiff
from dotenv import load_dotenv
from loguru import logger

res = load_dotenv()
# if not res:
#     logger.warning("No .env file found")

# DATA_DIR = Path('data_dir')
# `tau2` historically relied on REPO_PATH being exported by the container/SLURM env.
# For local runs (no container), make this robust by auto-detecting the repo root.
_repo_path = os.environ.get("REPO_PATH")
if not _repo_path:
    # This file lives at: <REPO>/evaluation/tau2-bench/tau2/utils/utils.py
    # parents[4] -> <REPO>
    _repo_path = str(Path(__file__).resolve().parents[4])
    os.environ["REPO_PATH"] = _repo_path

# In the original container setup, data was in evaluation/data_dir.
# For local runs without that structure, use data/ instead (where the repo actually stores it).
# Allow TAU2_DATA_DIR env var to override if user wants a custom location.
_data_dir_override = os.environ.get("TAU2_DATA_DIR")
if _data_dir_override:
    data_dir = _data_dir_override
else:
    _container_data_dir = os.path.join(_repo_path, "evaluation/data_dir")
    if os.path.isdir(_container_data_dir):
        data_dir = _container_data_dir
    else:
        # Fallback to the actual data/ directory in the repo
        data_dir = os.path.join(_repo_path, "data")

DATA_DIR = Path(data_dir)

def get_dict_hash(obj: dict) -> str:
    """
    Generate a unique hash for dict.
    Returns a hex string representation of the hash.
    """
    hash_string = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(hash_string.encode()).hexdigest()


def show_dict_diff(dict1: dict, dict2: dict) -> str:
    """
    Show the difference between two dictionaries.
    """
    diff = DeepDiff(dict1, dict2)
    return diff


def get_now() -> str:
    """
    Returns the current date and time in the format YYYYMMDD_HHMMSS.
    """
    now = datetime.now()
    return format_time(now)


def format_time(time: datetime) -> str:
    """
    Format the time in the format YYYYMMDD_HHMMSS.
    """
    return time.isoformat()


def get_commit_hash() -> str:
    """
    Get the commit hash of the current directory.
    """
    raise ValueError('debug')
    try:
        commit_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True)
            .strip()
            .split("\n")[0]
        )
    except Exception as e:
        logger.error(f"Failed to get git hash: {e}")
        commit_hash = "unknown"
    return commit_hash
