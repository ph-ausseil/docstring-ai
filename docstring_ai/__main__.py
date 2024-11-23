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
from docstring_ai.lib.logger import logging, LOG, show_file_progress
import difflib
from docstring_ai.lib.docstring_utils import (
    parse_classes,
    extract_class_docstring,
    extract_description_from_docstrings
)
from docstring_ai.lib.process import process_files_and_create_prs
from docstring_ai.lib.utils import (
    check_git_repo,
    has_uncommitted_changes,
    load_cache,
    save_cache,
    get_python_files,
    sort_files_by_size,
    prompt_user_confirmation,
)
from docstring_ai.lib.prompt_utils import add_docstrings

# Load environment variables from .env file
load_dotenv()
logging.setLevel(logging.DEBUG)

def main():
    """
    The main function that serves as the entry point of the script.

    This function sets up the command-line interface for configuring the 
    process of adding docstrings to Python files. It handles user input, 
    validates arguments, and orchestrates the docstring generation and 
    GitHub integration process.

    It accepts command-line arguments related to file paths, OpenAI API key, 
    GitHub repository information, and other configuration options for 
    creating pull requests.

    Raises:
        SystemExit: If there are errors in argument parsing or invalid input.
    """
    parser = argparse.ArgumentParser(
        description="Automate adding docstrings to Python files and integrate with GitHub for PR creation."
    )

    # CLI Arguments
    parser.add_argument("--path", required=True, help="Path to the repository or folder containing Python files.")
    parser.add_argument("--api_key", help="OpenAI API key. Defaults to the OPENAI_API_KEY environment variable.")
    parser.add_argument("--pr", help="GitHub repository for PR creation (e.g., owner/repository).")
    parser.add_argument("--github-token", help="GitHub personal access token. Defaults to the GITHUB_TOKEN environment variable.")
    parser.add_argument("--branch-name", help="Branch name for the PR. Auto-generated if not provided.")
    parser.add_argument("--pr-name", help="Custom name for the pull request. Defaults to '-- Add docstrings for files in `path`'.")
    parser.add_argument("--pr-depth", type=int, default=2, help="Depth level for creating PRs per folder. Default is 2.")
    parser.add_argument("--manual", action="store_true", help="Enable manual validation circuits for review.")
    parser.add_argument("--help-flags", action="store_true", help="List and describe all available flags.")

    # Parse arguments
    args = parser.parse_args()

    # If --help-flags is used, display flag descriptions and exit
    if args.help_flags:
        print("Available Flags:\n")
        print("  --path           (Required) Path to the repository or folder containing Python files.")
        print("  --api_key        OpenAI API key. Defaults to the OPENAI_API_KEY environment variable.")
        print("  --pr             GitHub repository for PR creation (e.g., owner/repository).")
        print("  --github-token   GitHub personal access token. Defaults to the GITHUB_TOKEN environment variable.")
        print("  --branch-name    Branch name for the PR. Auto-generated if not provided.")
        print("  --pr-name        Custom name for the pull request. Defaults to '-- Add docstrings for files in `path`'.")
        print("  --pr-depth      Depth level for creating PRs per folder. Default is 2.")
        print("  --manual         Enable manual validation circuits for review.")
        print("  --help-flags     List and describe all available flags.")
        return

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
    github_repo = args.pr or os.getenv("GITHUB_REPO")
    pr_depth = args.pr_depth
    branch_name = args.branch_name or f"feature/docstring-updates-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    pr_name = args.pr_name or f"-- Add docstrings for files in `{path}`"
    manual = args.manual

    if not github_repo:
        print("\n⚠️ WARNING: You are running the script without GitHub PR creation.")
        print("Modified files will be directly edited in place. Proceed with caution!")
        if not prompt_user_confirmation("Do you wish to continue?"):
            print("Operation aborted by the user.")
            sys.exit(0)

    if github_repo:
        if not github_token:
            print("Error: GitHub token must be provided via --github-token or the GITHUB_TOKEN environment variable.")
            exit(1)
        if not github_repo:
            print("Error: GitHub repository must be provided via --pr or the GITHUB_REPO environment variable.")
            exit(1)

        print(f"GitHub PR enabled for repository: {github_repo}")
        print(f"Using branch: {branch_name}")
        print(f"PR Name: {pr_name}")
        print(f"GitHub token: {'[HIDDEN]' if github_token else 'NOT SET'}")
        print(f"PR Depth: {pr_depth}")

    # Manual validation
    if manual:
        print("Manual validation circuits are enabled. You will be prompted to review changes before they are applied or PRs are created.")

    # Process files and handle PRs
    process_files_and_create_prs(
        path, api_key, args.pr is not None, github_token, 
        github_repo, branch_name, pr_name, pr_depth, manual
    )

if __name__ == "__main__":
    main()
