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
import re
import uuid
from datetime import datetime
from github import Github, GithubException
import subprocess
import logging

def sanitize_branch_name(name: str) -> str:
    """
    Sanitizes the branch name by replacing invalid characters with underscores and appending a UUID.
    
    Args:
        name (str): The original branch name.
    
    Returns:
        str: The sanitized and unique branch name.
    """
    
    sanitized = re.sub(r'[^A-Za-z0-9_-]+', '_', name.replace('/', '-'))
    return f"{sanitized}"

def create_github_pr(repo_path: str, github_token: str, github_repo: str, branch_name: str, pr_name: str) -> None:
    """
    Creates a GitHub pull request for the specified repository, branch, and pull request name.

    This function automates Git operations to create a new branch, commit changes,
    and push them to a remote repository on GitHub. It also gathers the files that
    have changed and includes them in the pull request body.

    Args:
        repo_path (str): The local path to the GitHub repository.
        github_token (str): The GitHub Access Token used for authentication.
        github_repo (str): The GitHub repository identifier in the format 'owner/repo'.
        branch_name (str): The base name for the new branch to create for the pull request.
        pr_name (str): The title of the pull request.

    Raises:
        GithubException: If there is an issue with the GitHub API (e.g., permission issues).
        subprocess.CalledProcessError: If any Git command fails during execution.
        Exception: For any other unexpected errors that may occur.
    """
    try:
        g = Github(github_token)
        repo = g.get_repo(github_repo)

        # Sanitize and generate unique branch name
        sanitized_branch_name = sanitize_branch_name(branch_name)
        unique_suffix = uuid.uuid4().hex[:8]  # 8-character unique suffix
        full_branch_name = f"{sanitized_branch_name}_{unique_suffix}"

        # Create a new branch from the default branch
        default_branch = repo.default_branch
        source = repo.get_branch(default_branch)
        ref = f"refs/heads/{full_branch_name}"

        try:
            repo.get_git_ref(ref)
            logging.info(f"Branch '{full_branch_name}' already exists.")
        except GithubException as e:
            if e.status == 404:
                repo.create_git_ref(ref=ref, sha=source.commit.sha)
                logging.info(f"Branch '{full_branch_name}' created.")
            else:
                logging.error(f"Failed to create branch '{full_branch_name}': {e}")
                raise e

        # Commit and push changes
        commit_message = "[Docstring-AI] ✨ Add docstrings via Docstring-AI script"
        commit_and_push_changes(repo_path, full_branch_name, commit_message)

        # Gather changed files
        changed_files = get_changed_files(repo_path, full_branch_name)

        if not changed_files:
            logging.warning("No Python files have changed. Pull Request will not be created.")
            return

        # Create Pull Request with list of changed files in the body
        pr_body = "Automated docstring additions.\n\n**Files Changed:**\n"
        for file in changed_files:
            pr_body += f"- `{file}`\n"

        pr = repo.create_pull(
            title="[Docstring-AI] " + pr_name,
            body=pr_body,
            head=full_branch_name,
            base=default_branch
        )
        logging.info(f"Pull Request created: {pr.html_url}")
    except GithubException as e:
        logging.error(f"GitHub API error: {e.data.get('message', e)}")
        raise e
    except subprocess.CalledProcessError as e:
        logging.error(f"Git command failed: {e}")
        raise e
    except Exception as e:
        logging.error(f"Error creating GitHub PR: {e}")
        raise e


def commit_and_push_changes(repo_path: str, branch_name: str, commit_message: str) -> None:
    """
    Commits and pushes changes to the specified branch in the given repository.

    This function manages Git operations to switch to the specified branch,
    add all changes, make a commit with the provided message, fetch and merge
    remote changes, and then push the changes to the remote repository.

    Args:
        repo_path (str): The local path to the GitHub repository.
        branch_name (str): The name of the branch to which changes will be committed.
        commit_message (str): The commit message to use when committing changes.

    Raises:
        subprocess.CalledProcessError: If any Git command fails during execution.
    """
    try:
        # Checkout or create the branch
        subprocess.run(
            ["git", "-C", repo_path, "checkout", "-B", branch_name],
            check=True
        )
        logging.info(f"Checked out to branch '{branch_name}'.")

        # Add all changes
        subprocess.run(
            ["git", "-C", repo_path, "add", "."],
            check=True
        )
        logging.info("Added all changes to staging.")

        # Check if there are changes to commit
        result = subprocess.run(
            ["git", "-C", repo_path, "diff", "--cached", "--exit-code"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        if result.returncode == 0:
            logging.info("No changes to commit.")
            return  # Exit the function as there are no changes

        # Commit changes
        subprocess.run(
            ["git", "-C", repo_path, "commit", "-m", commit_message],
            check=True
        )
        logging.info(f"Committed changes with message: {commit_message}")

        # Fetch remote changes for the branch
        subprocess.run(
            ["git", "-C", repo_path, "fetch", "origin", branch_name],
            check=True
        )
        logging.info(f"Fetched latest changes from remote branch '{branch_name}'.")

        # Merge remote changes into local branch
        subprocess.run(
            ["git", "-C", repo_path, "merge", f"origin/{branch_name}"],
            check=True
        )
        logging.info(f"Merged remote branch 'origin/{branch_name}' into '{branch_name}'.")

        # Push changes to remote repository
        subprocess.run(
            ["git", "-C", repo_path, "push", "-u", "origin", branch_name],
            check=True
        )
        logging.info(f"Changes pushed to branch '{branch_name}' on remote.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Git command failed: {e}")
        raise e

def get_changed_files(repo_path: str, branch_name: str) -> List[str]:
    """
    Retrieves a list of changed Python files in the given repository since the last commit.

    This function uses Git to determine which Python files have changed in the local repository
    by comparing the current state of the working directory to the last commit on the specified branch.

    Args:
        repo_path (str): The local path to the GitHub repository.
        branch_name (str): The name of the branch to compare against.

    Returns:
        List[str]: A list of changed Python files.
    """
    try:
        # Get the diff between the local branch and its upstream
        result = subprocess.run(
            ["git", "-C", repo_path, "diff", "--name-only", f"origin/{branch_name}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        changed_files = result.stdout.strip().split('\n')
        changed_files = [file for file in changed_files if file.endswith('.py') and file]
        logging.info(f"Changed Python files: {changed_files}")
        return changed_files
    except subprocess.CalledProcessError as e:
        logging.error(f"Git command failed while retrieving changed files: {e.stderr}")
        return []
