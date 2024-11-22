import os
import openai
import argparse
import time
import json
import ast
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


def create_github_pr(repo_path, github_token, github_repo, branch_name, pr_name):
    """
    Creates a GitHub pull request for the specified repository, branch, and PR name.
    Automatically gathers changed files and includes them in the PR body.
    Automates Git operations: checkout, add, commit, push.
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


def commit_and_push_changes(repo_path, branch_name, commit_message):
    """
    Commits and pushes changes to the specified branch.
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


def get_changed_files(repo_path) -> List[str]:
    """
    Retrieves a list of changed files in the repository.
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
