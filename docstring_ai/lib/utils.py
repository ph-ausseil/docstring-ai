import os
import openai
import argparse
import time
import json
import chromadb
import logging
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import tiktoken
from typing import List, Dict
import hashlib
from dotenv import load_dotenv
from datetime import datetime
from github import Github, GithubException
import subprocess
import sys
import logging
import difflib


def file_has_uncommitted_changes(repo_path: str, file_path: str) -> bool:
    """
    Checks if the specific file has uncommitted changes in the Git repository.
    Returns True if there are uncommitted changes for the file, False otherwise.
    """
    try:
        result = subprocess.run(
            ["git", "-C", repo_path, "diff", "--name-only"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        changed_files = result.stdout.strip().split('\n')
        return os.path.relpath(file_path, repo_path) in changed_files
    except subprocess.CalledProcessError as e:
        logging.error(f"Git command failed while checking uncommitted changes for {file_path}: {e}")
        return False


def prompt_user_confirmation(message: str) -> bool:
    """
    Prompts the user for a yes/no confirmation.
    """
    while True:
        response = input(f"{message} (yes/no): ").strip().lower()
        if response in ['yes', 'y']:
            return True
        elif response in ['no', 'n']:
            return False
        else:
            print("Please respond with 'yes' or 'no'.")


def check_git_repo(repo_path) -> bool:
    """
    Checks if the directory is a Git repository.
    Returns True if Git is available, False otherwise.
    """
    try:
        subprocess.run(
            ["git", "-C", repo_path, "rev-parse", "--is-inside-work-tree"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        logging.info("✅ Git repository detected.")
        return True
    except subprocess.CalledProcessError:
        logging.warning("❌ The specified path is not a Git repository.")
        return False
    except FileNotFoundError:
        logging.warning("❌ Git is not installed or not available in the PATH.")
        return False
    except Exception as e:
        logging.error(f"❌ Error checking Git repository: {e}")
        return False


def has_uncommitted_changes(repo_path) -> bool:
    """
    Checks for uncommitted changes in the Git repository.
    Returns True if there are uncommitted changes, False otherwise.
    """
    try:
        result = subprocess.run(
            ["git", "-C", repo_path, "status", "--porcelain"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        if result.stdout.strip():
            logging.warning("⚠️ Uncommitted changes detected in the repository!")
            print("Consider committing or stashing your changes before running the script.")
            confirm = input("Do you wish to continue? (yes/no): ").strip().lower()
            if confirm != "yes":
                print("Operation aborted by the user.")
                sys.exit(0)
            return True
        else:
            logging.info("✅ No uncommitted changes detected.")
            return False
    except subprocess.CalledProcessError as e:
        logging.error(f"Error checking uncommitted changes: {e}")
        return False


def load_cache(cache_file: str) -> Dict[str, str]:
    """
    Loads the cache from a JSON file.
    Returns a dictionary mapping file paths to their SHA-256 hashes.
    """
    if not os.path.exists(cache_file):
        logging.info(f"No cache file found at '{cache_file}'. Starting with an empty cache.")
        return {}
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache = json.load(f)
        logging.info(f"Loaded cache with {len(cache)} entries.")
        return cache
    except Exception as e:
        logging.error(f"Error loading cache file '{cache_file}': {e}")
        return {}


def save_cache(cache_file: str, cache: Dict[str, str]):
    """
    Saves the cache to a JSON file.
    """
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2)
        logging.info(f"Cache saved with {len(cache)} entries.")
    except Exception as e:
        logging.error(f"Error saving cache file '{cache_file}': {e}")



def get_python_files(repo_path: str) -> List[str]:
    """Recursively retrieve all Python files in the repository."""
    python_files = []
    for root, dirs, files in os.walk(repo_path):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
    return python_files


def sort_files_by_size(file_paths: List[str]) -> List[str]:
    """Sort files in ascending order based on their file size."""
    sorted_files = sorted(file_paths, key=lambda x: os.path.getsize(x))
    logging.info("Files sorted by size (ascending).")
    return sorted_files



def compute_sha256(file_path: str) -> str:
    """
    Computes the SHA-256 hash of a file.
    """
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            # Read and update hash string value in blocks of 4K
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception as e:
        logging.error(f"Error computing SHA-256 for {file_path}: {e}")
        return ""



def traverse_repo(repo_path: str, pr_depth: int) -> Dict[int, List[str]]:
    """
    Traverse the repository and categorize folders based on their depth.
    Returns a dictionary mapping depth levels to lists of folder paths.
    """
    folder_dict = {}
    for root, dirs, files in os.walk(repo_path):
        # Calculate the current folder's depth relative to repo_path
        rel_path = os.path.relpath(root, repo_path)
        if rel_path == ".":
            depth = 0
        else:
            depth = rel_path.count(os.sep) + 1  # +1 because rel_path doesn't start with sep
        if depth <= pr_depth:
            folder_dict.setdefault(depth, []).append(root)
    return folder_dict


def create_backup(file_path: str):
    """
    Creates a backup of the given file with a timestamp to prevent overwriting existing backups.
    """
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    backup_path = f"{file_path}.{timestamp}.bak"
    try:
        with open(file_path, 'r', encoding='utf-8') as original_file, \
             open(backup_path, 'w', encoding='utf-8') as backup_file:
            backup_file.write(original_file.read())
        logging.info(f"Backup created at {backup_path}")
    except Exception as e:
        logging.error(f"Error creating backup for {file_path}: {e}")


def show_diff(original_code: str, modified_code: str) -> str:
    """
    Generates a unified diff between the original and modified code.
    """
    original_lines = original_code.splitlines(keepends=True)
    modified_lines = modified_code.splitlines(keepends=True)
    diff = difflib.unified_diff(original_lines, modified_lines, fromfile='original', tofile='modified')
    return ''.join(diff)