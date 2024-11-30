

import json
import os
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from functools import partial
from datetime import datetime

import openai
import tiktoken
from tqdm import tqdm

from docstring_ai.lib.utils import (
    create_backup
)
from docstring_ai.lib.llm_utils import (
    create_file_with_docstring,
    generate_file_description,
    upload_files_to_openai
)
from docstring_ai.lib.chroma_utils import embed_and_store_files
from pydantic import BaseModel, Field



class TreeConfig(BaseModel):
    base_path: Path = Field(description="Base Path of the project")
    repositories_to_ignore: List[str] = Field(default = [] , description="List of repository paths to ignore.")
    extensions_to_ignore: List[str] = Field(default = ['.daicache'], description="List of file extensions to ignore.")
    apply_gitignore_policy: Optional[bool] = Field(default=True, description="Whether to apply .gitignore rules.")


def dump_tree(path: str, config: TreeConfig, prefix: str = ""):
    path = Path(path)

    # Compute the relative path for comparison with repositories_to_ignore
    relative_path = str(path.relative_to(config.base_path))
    # Check if the current path is in the repositories to ignore
    if relative_path in [str(Path(ignored)) for ignored in config.repositories_to_ignore]:
        return ""  # Skip this repository
    # Initialize the tree string representation
    tree_str_representation = f"{prefix}{path.name}/\n"

    # Load .gitignore rules if applicable
    gitignore_patterns = []
    if config.apply_gitignore_policy and (path / ".gitignore").exists():
        with open(path / ".gitignore") as f:
            gitignore_patterns = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    for p in path.iterdir():
        # Skip directories or files that match .gitignore or configuration rules
        if any(p.match(pattern) for pattern in gitignore_patterns):
            continue
        if p.is_dir() and str(p.resolve()) in [str(Path(config.base_path).joinpath(ignored).resolve()) for ignored in config.repositories_to_ignore]:
            continue
        if p.is_file() and any(p.suffix == ext for ext in config.extensions_to_ignore):
            continue

        # Recursive call for directories or add file to the tree
        if p.is_dir():
            tree_str_representation += dump_tree(p, prefix=prefix + "  ", config=config)
        else:
            tree_str_representation += f"{prefix}  {p.name}\n"

    return tree_str_representation



def generate_folder_descriptions(repo_path: Path, file_tree: str, generate_readme: bool = False, manual: bool = True):
    logging.debug("To be implemented : This function will create a summary of what is contained in a specific Folder")
    return {}

def generate_files_descriptions(
    files_to_describe: List[str],
    output_dir: Path,
    assistant_id: str,
    thread_id: str,
    context_summary: List[Dict],
    collection,
    api_key: str,
    repo_path: str,
    project_tree: str,
    directory_descriptions: Dict[str, str]
    ):
    file_descriptions_list = []
    for file in files_to_describe:
        relative_path = str(Path(os.path.relpath(file, repo_path)))
        if not any(str(Path(entry["file"])) == relative_path for entry in context_summary):
            try:


                file_description = generate_file_description(
                    assistant_id=assistant_id,
                    thread_id=thread_id,
                    project_tree=project_tree,
                    directory_descriptions=directory_descriptions,
                    file_path=file
                )

                # Save description
                description_file_path = output_dir / Path(file).with_suffix('.txt')
                description_file_path.parent.mkdir(parents=True, exist_ok=True)
                with open(description_file_path, 'w', encoding='utf-8') as f:
                    f.write(file_description)

                file_descriptions_list.append(str(description_file_path))
                context_summary.append({"file": relative_path, "description": file_description})

            except Exception as e:
                logging.error(f"Failed to generate description for {file}: {e}")
    return file_descriptions_list

def generate_descriptions(
    files_to_describe: List[str],
    output_dir: Path,
    assistant_id: str,
    thread_id: str,
    context_summary: List[Dict],
    collection,
    api_key: str,
    repo_path: Path  # Add repo_path parameter
) -> List[str]:
    """
    Generates detailed descriptions for files, embeds them into ChromaDB, and uploads to OpenAI, updating the Assistant's resources.

    Args:
        files_to_process (List[str]): List of file paths to generate descriptions for.
        output_dir (Path): Directory to store description files.
        assistant_id (str): OpenAI Assistant ID.
        thread_id (str): OpenAI Thread ID.
        context_summary (List[Dict]): Current context summary.
        collection: ChromaDB collection.
        api_key (str): OpenAI API key.
        repo_path (str): Repository path for computing relative paths.

    Returns:
        List[str]: List of successfully uploaded description file IDs.
    """
    description_file_ids = []
    repo_path = Path(repo_path)
    tree_config = TreeConfig(base_path= repo_path)
    project_tree = dump_tree(path= repo_path, config= tree_config)

    directory_descriptions = generate_folder_descriptions(repo_path =repo_path, file_tree = project_tree)

    file_descriptions_list = generate_files_descriptions(
            files_to_describe=files_to_describe,
            output_dir=output_dir,
            assistant_id=assistant_id,
            thread_id=thread_id,
            context_summary=context_summary,
            collection=collection,
            api_key=api_key,
            repo_path=repo_path,
            project_tree=project_tree,
            directory_descriptions=directory_descriptions
        )
    # Embed and upload descriptions
    if file_descriptions_list:
        embed_and_store_files(collection, file_descriptions_list, tags={"file_type": "description"})
        description_file_ids = upload_files_to_openai(file_descriptions_list)

    return description_file_ids
