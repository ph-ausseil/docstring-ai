import os
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
from pathlib import Path

def sanitize_branch_name(name: str) -> str:
    """
    Sanitizes the branch name by replacing invalid characters with underscores.
    
    Args:
        name (str): The original branch name.
    
    Returns:
        str: The sanitized branch name.
    """
    # Replace '/' with '-' to flatten branch hierarchy
    sanitized = name.replace('/', '-')
    # Replace any character that's not alphanumeric, '-', or '_' with '_'
    sanitized = re.sub(r'[^A-Za-z0-9_-]+', '_', sanitized)
    return sanitized

def generate_unique_suffix() -> str:
    """
    Generates a unique suffix using UUID4.
    
    Returns:
        str: An 8-character unique suffix.
    """
    return uuid.uuid4().hex[:8]

def create_github_pr(
    repo_path: str,
    github_token: str,
    github_repo: str,
    branch_base_name: str,
    pr_name: str,
    target_branch: str  # Target branch of the PR; After a PR is sent, this branch should be checked out
) -> bool:
    """
    Creates a GitHub pull request for the specified repository, branch, and pull request name.
    
    This function automates Git operations to create a new branch, commit changes,
    and push them to a remote repository on GitHub. It also gathers the files that
    have changed and includes them in the pull request body.
    
    Args:
        repo_path (str): The local path to the GitHub repository.
        github_token (str): The GitHub Access Token used for authentication.
        github_repo (str): The GitHub repository identifier in the format 'owner/repo'.
        branch_base_name (str): The base name for the new branch to create for the pull request.
        pr_name (str): The title of the pull request.
        target_branch (str): The target branch for the PR.
    
    Returns:
        bool: True if the PR was created successfully, False otherwise.
    """
    current_branch = None
    try:

        # Step 3: Initialize GitHub API
        g = Github(github_token)
        repo = g.get_repo(github_repo)

        # Step 4: Sanitize and generate unique branch name
        sanitized_branch_name = sanitize_branch_name(branch_base_name)
        unique_suffix = generate_unique_suffix()
        full_branch_name = f"{sanitized_branch_name}_{unique_suffix}"

        logging.info(f"Generated unique branch name: '{full_branch_name}'")

        # Step 5: Create a new branch from the target branch
        try:
            source = repo.get_branch(target_branch)
            ref = f"refs/heads/{full_branch_name}"
            try:
                repo.get_git_ref(ref)
                logging.info(f"Branch '{full_branch_name}' already exists on remote.")
            except GithubException as e:
                if e.status == 404:
                    repo.create_git_ref(ref=ref, sha=source.commit.sha)
                    logging.info(f"Branch '{full_branch_name}' created on remote.")
                else:
                    logging.error(f"Failed to create branch '{full_branch_name}': {e}")
                    return False
        except GithubException as e:
            logging.error(f"Failed to retrieve target branch '{target_branch}': {e}")
            return False

        # Step 6: Commit and push changes
        if not commit_and_push_changes(repo_path, full_branch_name, "[Docstring-AI] ✨ Add docstrings via Docstring-AI script"):
            logging.error("Failed to commit and push changes.")
            return False

        # Step 7: Gather changed files compared to the target branch
        changed_files = get_changed_files(repo_path, full_branch_name, target_branch)
        if not changed_files:
            logging.warning("No Python files have changed after commit. Pull Request will not be created.")
            return False

        # Step 8: Create Pull Request with list of changed files in the body
        pr_body = create_pull_request_body(changed_files)
        try:
            pr = repo.create_pull(
                title="[Docstring-AI] " + pr_name,
                body=pr_body,
                head=full_branch_name,
                base=target_branch
            )
            logging.info(f"Pull Request created: {pr.html_url}")
        except GithubException as e:
            logging.error(f"GitHub API error while creating PR: {e.data.get('message', e)}")
            return False

        # Step 9: Checkout the target branch regardless of PR creation success
        if not checkout_branch(repo_path, target_branch):
            logging.warning(f"Failed to checkout to target branch '{target_branch}'.")
            # Not returning False here since PR creation was successful
        else:
            logging.info(f"Successfully checked out to target branch '{target_branch}'.")

        return True

    except Exception as e:
        logging.error(f"Unexpected error in create_github_pr: {e}")
        # Attempt to checkout target_branch even in case of unexpected errors
        if not checkout_branch(repo_path, target_branch):
            logging.warning(f"Failed to checkout to target branch '{target_branch}' after an error.")
        return False


def get_python_files(repo_path: str) -> List[str]:
    """
    Retrieves a list of all Python files in the given repository.
    
    Args:
        repo_path (str): The local path to the GitHub repository.
    
    Returns:
        List[str]: A list of Python file paths.
    """
    python_files = []
    for root, dirs, files in os.walk(repo_path):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for file in files:
            if file.endswith('.py'):
                full_path = os.path.join(root, file)
                python_files.append(os.path.relpath(full_path, repo_path))
    logging.info(f"Total Python files found: {len(python_files)}")
    return python_files


def create_pull_request_body(changed_files: List[str]) -> str:
    """
    Creates the body content for the pull request listing the changed files.
    
    Args:
        changed_files (List[str]): List of changed Python file paths.
    
    Returns:
        str: The formatted PR body.
    """
    pr_body = "Automated docstring additions.\n\n**Files Changed:**\n"
    for file in changed_files:
        pr_body += f"- `{file}`\n"
    return pr_body


def commit_and_push_changes(repo_path: str, branch_name: str, commit_message: str) -> bool:
    """
    Commits and pushes changes to the specified branch in the given repository.
    
    This function manages Git operations to switch to the specified branch,
    add all changes, make a commit with the provided message, fetch and merge
    remote changes, and then push the changes to the remote repository.

    Args:
        repo_path (str): The local path to the GitHub repository.
        branch_name (str): The name of the branch to which changes will be committed.
        commit_message (str): The commit message to use when committing changes.
    
    Returns:
        bool: True if commit and push were successful, False otherwise.
    """
    try:
        # Checkout or create the branch locally
        subprocess.run(
            ["git", "-C", repo_path, "checkout", "-B", branch_name],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        logging.info(f"Checked out to branch '{branch_name}' locally.")

        # Add all changes
        subprocess.run(
            ["git", "-C", repo_path, "add", "."],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        logging.info("Added all changes to staging.")

        # Log git status
        if not log_git_status(repo_path):
            return False

        # Check if there are changes to commit
        result = subprocess.run(
            ["git", "-C", repo_path, "diff", "--cached", "--exit-code"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        if result.returncode == 0:
            logging.info("No changes to commit.")
            return True  # No changes to commit, but not an error

        # Commit changes
        subprocess.run(
            ["git", "-C", repo_path, "commit", "-m", commit_message],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        logging.info(f"Committed changes with message: '{commit_message}'")

        # Display the latest commit for verification
        commit_log = subprocess.run(
            ["git", "-C", repo_path, "log", "-1"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        logging.info(f"Latest Commit:\n{commit_log.stdout}")

        # Fetch remote changes for the branch
        subprocess.run(
            ["git", "-C", repo_path, "fetch", "origin", branch_name],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        logging.info(f"Fetched latest changes from remote branch '{branch_name}'.")

        # Merge remote changes into local branch
        subprocess.run(
            ["git", "-C", repo_path, "merge", f"origin/{branch_name}"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        logging.info(f"Merged remote branch 'origin/{branch_name}' into '{branch_name}'.")

        # Push changes to remote repository
        subprocess.run(
            ["git", "-C", repo_path, "push", "-u", "origin", branch_name],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        logging.info(f"Changes pushed to branch '{branch_name}' on remote.")

        # Display the remote commit history for verification
        remote_log = subprocess.run(
            ["git", "-C", repo_path, "log", "origin/" + branch_name, "--oneline"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Log git status after push
        if not log_git_status(repo_path):
            return False

        return True

    except subprocess.CalledProcessError as e:
        logging.error(f"Git command failed: {e.stderr.decode().strip()}")
        return False

def get_changed_files(repo_path: str, branch_name: str, base_branch: str) -> List[str]:
    """
    Retrieves a list of changed Python files in the given repository between the base branch and the feature branch.
    
    Args:
        repo_path (str): The local path to the GitHub repository.
        branch_name (str): The name of the feature branch.
        base_branch (str): The name of the base branch to compare against.
    
    Returns:
        List[str]: A list of changed Python files.
    """
    try:
        # Fetch the latest changes for both base and feature branches
        subprocess.run(
            ["git", "-C", repo_path, "fetch", "origin", base_branch],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        subprocess.run(
            ["git", "-C", repo_path, "fetch", "origin", branch_name],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        logging.info(f"Fetched latest changes from base branch '{base_branch}' and feature branch '{branch_name}'.")

        # Get the diff between the remote base branch and remote feature branch
        result = subprocess.run(
            ["git", "-C", repo_path, "diff", "--name-only", f"origin/{base_branch}..origin/{branch_name}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        changed_files = result.stdout.strip().split('\n')
        changed_files = [file for file in changed_files if file.endswith('.py') and file]
        if changed_files:
            logging.info(f"Changed Python files: {changed_files}")
        else:
            logging.info("No Python files have changed.")
        return changed_files
    except subprocess.CalledProcessError as e:
        logging.error(f"Git command failed while retrieving changed files: {e.stderr.strip()}")
        return []


def log_git_status(repo_path: str) -> bool:
    """
    Logs the current git status.
    
    Args:
        repo_path (str): The local path to the GitHub repository.
    
    Returns:
        bool: True if git status was logged successfully, False otherwise.
    """
    try:
        status = subprocess.run(
            ["git", "-C", repo_path, "status"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        logging.info(f"Git Status:\n{status.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to retrieve git status: {e.stderr.strip()}")
        return False

def checkout_branch(repo_path: str, branch_name: str) -> bool:
    """
    Checks out the specified branch in the repository.
    
    Args:
        repo_path (str): The local path to the GitHub repository.
        branch_name (str): The name of the branch to check out.
    
    Returns:
        bool: True if checkout was successful, False otherwise.
    """
    try:
        subprocess.run(
            ["git", "-C", repo_path, "checkout", branch_name],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        logging.info(f"Checked out to branch '{branch_name}'.")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to checkout branch '{branch_name}': {e.stderr.decode().strip()}")
        return False

