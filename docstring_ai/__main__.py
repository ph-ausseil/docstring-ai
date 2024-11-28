"""
This module automates the process of adding docstrings to Python files,
integrating with GitHub to create pull requests (PRs). It uses OpenAI's API 
for generating docstrings and CLI arguments for configuration.

Modules:
- argparse: For parsing command-line arguments.
- os: For file and environment operations.
- openai: To interact with the OpenAI API.
- chromadb: For embedding and storing code context.
- LOG: For LOG messages and errors.
- datetime: For handling date and time operations.
- subprocess: To run shell commands.
- sys: For system-specific parameters and functions.
- dotenv: To load environment variables from a .env file.

Functions:
- main: The entry point of the script that handles argument parsing and execution flow.
"""
from github import Github
import subprocess
import re
from pathlib import Path
import os
import openai
import argparse
import time
import json
import chromadb
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
from docstring_ai.lib.docstring_utils import (
    parse_classes,
    extract_class_docstring,
    extract_description_from_docstrings
)
from docstring_ai.lib.process import process_files_and_create_prs
from docstring_ai.lib.utils import (
    check_git_repo,
    repo_has_uncommitted_changes,
    load_cache,
    save_cache,
    get_python_files,
    sort_files_by_size,
    prompt_user_confirmation,
)
from docstring_ai.lib.prompt_utils import create_file_with_docstring
from docstring_ai.lib.config import CACHE_FILE_NAME, CONTEXT_SUMMARY_PATH, setup_logging

# Load environment variables from .env file
load_dotenv()
setup_logging()

import subprocess
import re
import requests


def is_git_repo(folder_path):
    """Check if the folder is a Git repository."""
    try:
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=folder_path, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False

def get_remote_url(folder_path):
    """Retrieve the remote URL for the Git repository."""
    try:
        return subprocess.check_output(["git", "remote", "get-url", "origin"], cwd=folder_path, stderr=subprocess.DEVNULL).strip().decode()
    except subprocess.CalledProcessError:
        return None

def parse_github_url(remote_url):
    """Extract user and repository name from a GitHub remote URL."""
    match = re.search(r"github\.com[:/](.+?)/(.+?)(?:\.git)?$", remote_url)
    if match:
        return match.groups()
    return None, None

def determine_pr_target(path: str, args) -> (bool, str):
    """
    Determine whether to enable PR creation and the target GitHub repository.

    Args:
        path (str): Path to the repository or folder.
        args (Namespace): Parsed CLI arguments.

    Returns:
        (bool, str): A tuple where the first element indicates whether PR creation is enabled,
                     and the second is the GitHub repository (owner/repo) if applicable.
    """
    use_repo_config = args.use_repo_config
    if is_git_repo(path):
        remote_url = get_remote_url(path)
        if remote_url:
            user, repo = parse_github_url(remote_url)
            if user and repo and use_repo_config:
                return True, f"{user}/{repo}"

            if user and repo:
                print(f"The folder {str(Path(path).absolute())} is part of the GitHub repository: {user}/{repo}")
                proceed = input(f"Do you want to create a pull request on the repository {user}/{repo}? (yes/no): ").strip().lower()
                if proceed == "yes":
                    return True, f"{user}/{repo}"

    if args.pr:
        return True, args.pr

    if os.getenv('GITHUB_REPO'):
        proceed = input(f"Do you want to use the folder {os.getenv('GITHUB_REPO')} as GitHub repository? (yes/no): ").strip().lower()
        if proceed == "yes":
            return True, os.getenv("GITHUB_REPO")

    return False, None

def determine_target_branch(path: str, args) -> str:
    """
    Determine the target branch for the PR.

    Args:
        path (str): Path to the repository or folder.
        args (Namespace): Parsed CLI arguments.

    Returns:
        str: The name of the target branch.
    """

    try:
        current_branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=path).strip().decode()
        if args.use_repo_config: 
            return current_branch

        print(f"The current branch in the repository is: {current_branch}")
        proceed = input(f"Do you want to use '{current_branch}' as the target branch? (yes/no): ").strip().lower()
        if proceed == "yes":
            return current_branch
    except subprocess.CalledProcessError:
        if args.target_branch:
            return args.target_branch

        if os.getenv('GITHUB_TARGET_BRANCH'):
            return os.getenv('GITHUB_TARGET_BRANCH')

    return None

def main():
    """
    The main function that serves as the entry point of the script.

    This function sets up the command-line interface for configuring the 
    process of adding docstrings to Python files. It handles user input, 
    validates arguments, and orchestrates the docstring generation and 
    GitHub integration process.
    """
    parser = argparse.ArgumentParser(
        description="Automate adding docstrings to Python files and integrate with GitHub for PR creation."
    )

    # CLI Arguments
    parser.add_argument("--path", required=True, help="Path to the repository or folder containing Python files.")
    parser.add_argument("--api_key", help="OpenAI API key. Defaults to the OPENAI_API_KEY environment variable.")
    parser.add_argument("--manual", action="store_true", help="Enable manual validation circuits for review.")
    parser.add_argument("--no-cache", action="store_true", help="Execute the script without cached values.")
    parser.add_argument("--help-flags", action="store_true", help="List and describe all available flags.")
    parser.add_argument("--pr-depth", type=int, default=2, help="Depth level for creating PRs per folder. Default is 2.")
    parser.add_argument("--use-repo-config", help="Use if the --path is a git repo exit, it will use git config (and overide any of the following parametter).")
    parser.add_argument("--pr", help="GitHub repository for PR creation (e.g., owner/repository).")
    parser.add_argument("--target-branch", help="Target branch of the PR.")
    parser.add_argument("--github-token", help="GitHub personal access token. Defaults to the GITHUB_TOKEN environment variable.")
    parser.add_argument("--branch-name", help="Branch name for the PR. Auto-generated if not provided.")
    parser.add_argument("--pr-name", help="Custom name for the pull request. Defaults to '-- Add docstrings for files in `path`'.")

    # Parse arguments
    args = parser.parse_args()

    # If --help-flags is used, display flag descriptions and exit
    if args.help_flags:
        print("Available Flags:\n")
        print("  --path             (Required) Path to the repository or folder containing Python files.")
        print("  --api_key          OpenAI API key. Defaults to the OPENAI_API_KEY environment variable.")
        print("  --manual           Enable manual validation circuits for review.")
        print("  --no-cache         Execute the script without cached values by deleting cache files.")
        print("  --use-repo-config  Use if the --path is a git repo exit, it will use git config (and overide any of the following parametter).")
        print("  --pr               GitHub repository for PR creation (e.g., owner/repository).")
        print("  --github-token     GitHub personal access token. Defaults to the GITHUB_TOKEN environment variable.")
        print("  --branch-name      Branch name for the PR. Auto-generated if not provided.")
        print("  --pr-name          Custom name for the pull request. Defaults to '-- Add docstrings for files in `path`'.")
        print("  --pr-depth         Depth level for creating PRs per folder. Default is 2.")
        return

    # Handle the --no-cache flag
    if args.no_cache:
        # Paths for cache and context summary
        cache_file = os.path.join(args.path, CACHE_FILE_NAME)
        context_summary_file = os.path.join(args.path, CONTEXT_SUMMARY_PATH)

        # Delete cache files if they exist
        if os.path.exists(cache_file):
            os.remove(cache_file)
            print(f"Deleted cache file: {CACHE_FILE_NAME}")
        else:
            print(f"No cache file found: {CACHE_FILE_NAME}")

        if os.path.exists(context_summary_file):
            os.remove(context_summary_file)
            print(f"Deleted context summary file: {CONTEXT_SUMMARY_PATH}")
        else:
            print(f"No context summary file found: {CONTEXT_SUMMARY_PATH}")

    # Retrieve API key
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OpenAI API key must be provided via --api_key or the OPENAI_API_KEY environment variable.")
        exit(1)

    # Path validation
    path = args.path
    if not os.path.exists(path):
        print(f"Error: The specified path '{path}' does not exist.")
        exit(1)

 
    # GitHub integration
    github_token = args.github_token or os.getenv("GITHUB_TOKEN")
    use_repo_config = args.use_repo_config
    pr_depth = args.pr_depth
    branch_name = args.branch_name or "docstring-ai"
    pr_name = args.pr_name or "Add docstrings for folder"
    manual = args.manual

    # Determine PR target
    pr_enabled, github_repo = determine_pr_target(path, args)
    if not pr_enabled:
        print("\n⚠️ WARNING: You are running the script without GitHub PR creation.")
        print("Modified files will be directly edited in place. Proceed with caution!")
        if not prompt_user_confirmation("Do you wish to continue?"):
            print("Operation aborted by the user.")
            sys.exit(0)
    else:
        if not github_token:
            print("Error: GitHub token must be provided via --github-token or the GITHUB_TOKEN environment variable.")
            exit(1)
        if not github_repo:
            print("Error: GitHub repository must be provided via --pr or the GITHUB_REPO environment variable.")
            exit(1)

           # Determine target branch
        target_branch = determine_target_branch(path, args)
        if not target_branch:
            print("Error: Unable to determine the target branch. Please provide a target branch using --target-branch.")
            exit(1)

        print(f"GitHub PR enabled for repository: {github_repo}")
        print(f"Target branch: {target_branch}")
        print(f"Using branch: {branch_name}")
        print(f"PR Name: {pr_name}")
        print(f"GitHub token: {'[HIDDEN]' if github_token else 'NOT SET'}")
        print(f"PR Depth: {pr_depth}")




    # Manual validation
    if manual:
        print("Manual validation circuits are enabled. You will be prompted to review changes before they are applied or PRs are created.")

    # Process files and handle PRs
    process_files_and_create_prs(
        repo_path= path,
        api_key=api_key,
        create_pr= pr_enabled,
        github_token=github_token, 
        github_repo=github_repo, 
        branch_name=branch_name, 
        pr_name=pr_name, 
        pr_depth=pr_depth, 
        manual=manual,
        target_branch=target_branch,
    )


if __name__ == "__main__":
    main()
