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
        branch_name (str): The name of the new branch to create for the pull request.
        pr_name (str): The title of the pull request.

    Raises:
        GithubException: If there is an issue with the GitHub API (e.g., permission issues).
        subprocess.CalledProcessError: If any Git command fails during execution.
        Exception: For any other unexpected errors that may occur.
    """
    try:
        g = Github(github_token)
        repo = g.get_repo(github_repo)

        # Create a new branch from the default branch
        default_branch = repo.default_branch
        source = repo.get_branch(default_branch)
        ref = f"refs/heads/{branch_name}"

        try:
            repo.get_git_ref(ref)
            logging.info(f"Branch '{branch_name}' already exists.")
        except GithubException as e:
            if e.status == 404:
                repo.create_git_ref(ref=ref, sha=source.commit.sha)
                logging.info(f"Branch '{branch_name}' created.")
            else:
                raise e

        # Commit and push changes
        commit_message = "[Docstring-AI] ✨ Add docstrings via Docstring-AI script"
        commit_and_push_changes(repo_path, branch_name, commit_message)

        # Gather changed files
        changed_files = get_changed_files(repo_path)

        # Create Pull Request with list of changed files in the body
        pr_body = "Automated docstring additions.\n\n**Files Changed:**\n"
        for file in changed_files:
            pr_body += f"- `{file}`\n"

        pr = repo.create_pull(
            title="[Docstring-AI] " + pr_name,
            body=pr_body,
            head=branch_name,
            base=default_branch
        )
        logging.info(f"Pull Request created: {pr.html_url}")
    except GithubException as e:
        logging.error(f"GitHub API error: {e.data['message']}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Git command failed: {e}")
    except Exception as e:
        logging.error(f"Error creating GitHub PR: {e}")


def commit_and_push_changes(repo_path: str, branch_name: str, commit_message: str) -> None:
    """
    Commits and pushes changes to the specified branch in the given repository.

    This function manages Git operations to switch to the specified branch,
    add all changes, make a commit with the provided message, and then push
    the changes to the remote repository.

    Args:
        repo_path (str): The local path to the GitHub repository.
        branch_name (str): The name of the branch to which changes will be committed.
        commit_message (str): The commit message to use when committing changes.

    Raises:
        subprocess.CalledProcessError: If any Git command fails during execution.
    """
    try:
        subprocess.run(
            ["git", "-C", repo_path, "checkout", "-B", branch_name],
            check=True
        )
        subprocess.run(
            ["git", "-C", repo_path, "add", "."],
            check=True
        )
        subprocess.run(
            ["git", "-C", repo_path, "commit", "-m", commit_message],
            check=True
        )
        subprocess.run(
            ["git", "-C", repo_path, "push", "-u", "origin", branch_name],
            check=True
        )
        logging.info(f"Changes committed and pushed to branch '{branch_name}'.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Git command failed: {e}")
        raise e


def get_changed_files(repo_path: str) -> List[str]:
    """
    Retrieves a list of changed files in the given repository since the last commit.

    This function uses Git to determine which files have changed in the local repository
    by comparing the current state of the working directory to the last commit.

    Args:
        repo_path (str): The local path to the GitHub repository.

    Returns:
        List[str]: A list of changed files, filtered to include only Python files ('.py').

    Raises:
        subprocess.CalledProcessError: If the Git command fails during execution.
    """
    try:
        result = subprocess.run(
            ["git", "-C", repo_path, "diff", "--name-only", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        changed_files = result.stdout.strip().split('\n')
        changed_files = [file for file in changed_files if file.endswith('.py')]
        logging.info(f"Changed Python files: {changed_files}")
        return changed_files
    except subprocess.CalledProcessError as e:
        logging.error(f"Git command failed while retrieving changed files: {e}")
        return []
