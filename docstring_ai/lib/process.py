"""
This module processes Python files to add docstrings using OpenAI's Assistant,
embeds the files in ChromaDB, and integrates with GitHub for pull request creation.

Functions:
- process_files_and_create_prs: Processes Python files, adds docstrings, and creates pull requests.
"""

import json
import os
from pathlib import Path
import logging
from docstring_ai.lib.logger import show_file_progress
import openai
import datetime
import tiktoken
from docstring_ai.lib.utils import (
    ensure_docstring_header,
    check_git_repo,
    has_uncommitted_changes,
    file_has_uncommitted_changes,
    load_cache,
    save_cache,
    get_python_files,
    sort_files_by_size,
    prompt_user_confirmation,
    show_diff,
    compute_sha256,
    traverse_repo,
    create_backup
)
from docstring_ai.lib.prompt_utils import (
    initialize_assistant,
    update_assistant_tool_resources,
    create_thread,
    construct_few_shot_prompt,
    add_docstrings,
    generate_file_description
)
from docstring_ai.lib.chroma_utils import (
    initialize_chroma,
    get_or_create_collection,
    embed_and_store_files,
    get_relevant_context,
    store_class_summary
)
from docstring_ai.lib.docstring_utils import (
    parse_classes,
    extract_class_docstring,
    extract_description_from_docstrings
)
from docstring_ai.lib.github_utils import create_github_pr
from docstring_ai import (
    MAX_TOKENS, 
    CHROMA_COLLECTION_NAME, 
    CACHE_FILE_NAME, 
    DATA_PATH, 
    CONTEXT_SUMMARY_PATH
    )

def process_files_and_create_prs(
    repo_path: str, 
    api_key: str, 
    create_pr: bool, 
    github_token: str, 
    github_repo: str, 
    branch_name: str, 
    pr_name: str, 
    pr_depth: int, 
    manual: bool
) -> None:
    """
    Processes Python files in the specified repository, adds docstrings using OpenAI's Assistant,
    and creates pull requests on GitHub if specified.

    1. **Setup and Initialization**
    Step 1: Verify the presence of a Git repository and check for uncommitted changes.
    Step 2: Initialize ChromaDB for context-aware file embedding and retrieval.
    Step 3: Load a cache file to track previously processed files.

    2. **File Discovery and Preparation**
    Step 4: Retrieve all Python files in the repository.
    Step 5: Sort files by size to optimize processing order.
    Step 6: `filter_files_by_hash` - Compute SHA-256 hashes and filter out unchanged files using the cache.

    3. **Embedding and Assistant Setup**
    Step 7: Embed selected Python files into ChromaDB for efficient context storage.
    Step 9: `upload_files_to_openai` - Upload files to OpenAI and update the Assistant's resources.
    Step 8: Initialize an OpenAI Assistant instance for docstring generation.
    Step 10: Create a new OpenAI thread for interaction and processing.

    4. **Docstring Generation and Processing**
    Step 11: `process_single_file` - Process each selected Python file to generate and add docstrings.
    Step 12: Support manual approval of changes, if enabled.
    Step 13: Save a context summary of processed files for future reference.

    5. **Cache Update and Pull Request Creation (Optional)**
    Step 14: Update the cache with the latest file states.
    Step 15: Traverse the repository by folder depth and prepare for pull request creation.
    Step 16: Create pull requests for the processed files (if enabled).

    Args:
        repo_path (str): The path of the Git repository to process.
        api_key (str): The OpenAI API key for making requests.
        create_pr (bool): Flag indicating if pull requests should be created.
        github_token (str): The GitHub token for authentication.
        github_repo (str): The repository where pull requests will be made.
        branch_name (str): The name of the branch for the pull request.
        pr_name (str): The name of the pull request to create.
        pr_depth (int): The maximum depth to categorize folders for PR creation.
        manual (bool): Flag indicating if manual approval is required for changes.

    Returns:
        None: This function performs its operations and does not return a value.

    Raises:
        Exception: Various exceptions may occur during file processing, API calls,
                    or Git operations, which are logged accordingly.
    """
    ###
    ### INITIATE
    ###
    openai.api_key = api_key

    # Check if Git is present
    git_present = check_git_repo(repo_path)

    # Check for uncommitted changes if Git is present
    # uncommitted_changes = False
    # if git_present:
    #     uncommitted_changes = has_uncommitted_changes(repo_path)

    # Initialize ChromaDB
    logging.info("\nInitializing ChromaDB...")
    chroma_client = initialize_chroma()
    collection = get_or_create_collection(chroma_client, CHROMA_COLLECTION_NAME)

    # Load cache
    cache_path = os.path.join(repo_path, CACHE_FILE_NAME)
    cache = load_cache(cache_path)

    # Step 1: Retrieve all Python files
    python_files = get_python_files(repo_path)
    logging.info(f"Found {len(python_files)} Python files to process.")

    if not python_files:
        logging.info("No Python files found. Exiting.")
        return

    # Step 2: Sort files by size (ascending)
    python_files_sorted = sort_files_by_size(python_files)

    # Step 3: Compute SHA-256 hashes and filter out unchanged files
    logging.info("\nChecking file hashes...")
    files_to_process = filter_files_by_hash(python_files_sorted, repo_path, cache)

    logging.info(f"\n{len(files_to_process)} files to process after cache check.")

    if not files_to_process:
        logging.info("No files need processing. Exiting.")
        return

    # Step 4: Traverse repository and categorize folders based on pr_depth
    logging.info("\nTraversing repository to categorize folders based on pr_depth...")
    folder_dict = traverse_repo(repo_path, pr_depth)
    logging.info(f"Found {len(folder_dict)} depth levels up to {pr_depth}.")
    
    ###
    ### EMBED
    ###

    # Load Context Summary
    context_summary_full_path = os.path.join(repo_path, CONTEXT_SUMMARY_PATH)
    context_summary = []  # Initialize context_summary as an empty list by default
    if os.path.exists(context_summary_full_path):
        try:
            with open(context_summary_full_path, 'r', encoding='utf-8') as f:
                context_summary = json.load(f)
            logging.info(f"Loaded context summary with {len(context_summary)} entries.")
        except Exception as e:
            logging.error(f"Error loading context summary: {e}")

    # Step 5: Embed and store files in ChromaDB
    logging.info("\nEmbedding and storing Python files in ChromaDB...")
    embed_and_store_files(collection, files_to_process)

    # Step 6: Upload files to OpenAI and update Assistant's tool resources
    logging.info("\nUploading files to OpenAI...")
    file_ids = upload_files_to_openai(files_to_process)

    if not file_ids:
        logging.error("No files were successfully uploaded to OpenAI. Exiting.")
        return

    ###
    ### START ASSISTANT 
    ###
    # Step 7: Initialize Assistant
    logging.info("\nInitializing Assistant...")
    assistant_id = initialize_assistant(api_key)
    if not assistant_id:
        logging.error("Assistant initialization failed. Exiting.")
        return

    # Update Assistant's tool resources with OpenAI file IDs
    update_assistant_tool_resources(
        api_key=api_key,
        assistant_id=assistant_id,
        file_ids=file_ids
        )

    # Step 8: Create a Thread
    logging.info("\nCreating a new Thread...")
    thread_id = create_thread(
        api_key=api_key, 
        assistant_id=assistant_id
        )
    if not thread_id:
        logging.error("Thread creation failed. Exiting.")
        return

    # Ensure the './data/' directory exists
    output_dir = DATA_PATH
    output_dir.mkdir(parents=True, exist_ok=True)  # Creates the directory if it doesn't exist
    # Step 9 : Create Missing Context 
    file_descriptions_list = []
    for file in files_to_process: 
        if not any(entry["file"] == file for entry in context_summary): 
            logging.info(f"Generating detailed description for {file}...")
            with open(file, 'r', encoding='utf-8') as f:
                file_description = generate_file_description(
                    assistant_id=assistant_id,
                    thread_id=thread_id, 
                    file_content=f.read()
                )

            file_path = Path(output_dir) / Path(file).with_suffix('.txt')
            file_path.parent.mkdir(parents=True, exist_ok=True) 
            # Create a file with descriptions
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(file_description)
            file_descriptions_list.append(file_path)

            context_summary.append({
                "file": file,
                "description": file_description
            })
            # Append to context_summary for future use

    # TODO: Add context to both vector stores
    # Step 5: Embed and store files in ChromaDB
    logging.info("\nEmbedding and storing Python files in ChromaDB...")
    embed_and_store_files(collection, file_descriptions_list)

    # Step 6: Upload files to OpenAI and update Assistant's tool resources
    logging.info("\nUploading files to OpenAI...")
    file_ids.extend(upload_files_to_openai(file_descriptions_list))
    update_assistant_tool_resources(api_key, assistant_id, file_ids)
 
    # Step 9: Process Each Python File for Docstrings using the decorated function
    logging.info("\nProcessing Python files to add docstrings...")
    process_single_file(
        files=files_to_process,
        repo_path=repo_path,
        assistant_id=assistant_id,
        thread_id=thread_id,
        collection=collection,
        context_summary=context_summary,
        cache=cache,
        manual=manual
    )

    # Step 10: Save Context Summary
    try:
        with open(context_summary_full_path, "w", encoding='utf-8') as f:
            json.dump(context_summary, f, indent=2)
        logging.info(f"\nDocstring generation completed. Context summary saved to '{context_summary_full_path}'.")
    except Exception as e:
        logging.error(f"Error saving context summary: {e}")

    # Step 11: Save Cache
    save_cache(cache_path, cache)

    # Step 12: Create Pull Requests Based on pr_depth
    if create_pr and git_present and github_token and github_repo:
        if manual:
            # Show summary of PRs to be created and ask for confirmation
            print("\nPull Requests to be created for the following folders:")
            for depth, folders in folder_dict.items():
                for folder in folders:
                    print(f"- Depth {depth}: {folder}")
            if not prompt_user_confirmation("Do you want to proceed with creating these Pull Requests?"):
                logging.info("Pull Request creation aborted by the user.")
                return

        logging.info("\nCreating Pull Requests based on pr_depth...")
        for depth, folders in folder_dict.items():
            for folder in folders:
                # Collect all Python files in the folder
                pr_files = get_python_files(folder)
                if not pr_files:
                    continue  # Skip folders with no Python files

                # Generate a unique branch name for the folder
                folder_rel_path = os.path.relpath(folder, repo_path).replace(os.sep, "_")
                folder_branch_name = f"feature/docstrings-folder-{folder_rel_path}-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

                # Generate PR name
                folder_pr_name = f"-- Add docstrings for folder `{folder_rel_path}`" if not pr_name else pr_name

                # Create GitHub PR
                create_github_pr(
                    repo_path=repo_path, 
                    github_token=github_token, 
                    github_repo=github_repo, 
                    branch_name=folder_branch_name, 
                    pr_name=folder_pr_name
                )

@show_file_progress(desc="Processing Python Files", leave=True)
def process_single_file(
    file_path: str,
    repo_path: str,
    assistant_id: str,
    thread_id: str,
    collection,
    context_summary: list,
    cache: dict,
    manual: bool
) -> None:
    """
    Processes a single Python file: adds docstrings, updates context, and handles caching.

    **Steps:**

    1. **File Reading and Initial Setup**
       - Compute the file's relative path within the repository.
       - Read the file content into memory.
       - Log errors and skip processing if the file cannot be read.

    2. **Class Parsing and Context Retrieval**
       - Parse the file for class definitions and dependencies.
       - If classes are found, retrieve relevant context summaries from ChromaDB to support docstring generation.

    3. **Few-Shot Prompt Construction**
       - Construct a few-shot prompt using the retrieved context and examples from the ChromaDB collection.

    4. **File Description Generation**
       - Check if the file's description is already cached.
       - If not cached, generate a detailed file description using the OpenAI Assistant and append it to the context summary.

    5. **Docstring Generation**
       - Use the OpenAI Assistant to generate docstrings for the code, applying the few-shot prompt and contextual information.

    6. **Manual Validation (Optional)**
       - If `manual` is set to True, display the differences between the original and modified code for user approval before saving changes.

    7. **File Update**
       - Create a backup of the original file if Git is not present or if the file has uncommitted changes.
       - Overwrite the file with the modified code containing the new docstrings.

    8. **Context Summary and Cache Updates**
       - Generate an updated file description after docstring addition.
       - Update the cache with the new file hash to avoid reprocessing unchanged files.
       - Update the context summary with the new description.

    9. **Class Summaries**
       - Extract and store summaries for any modified classes in the file.

    Args:
        file_path (str): The path to the Python file.
        repo_path (str): The repository path.
        assistant_id (str): The OpenAI assistant ID.
        thread_id (str): The thread ID for OpenAI interactions.
        collection: The ChromaDB collection.
        context_summary (list): The context summary list.
        cache (dict): The cache dictionary.
        manual (bool): Flag indicating if manual approval is required.

    Returns:
        None
    """
    try:
        relative_path = os.path.relpath(file_path, repo_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            original_code = f.read()
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {e}")
        return

    # Check if file is cached and has existing description
    cached_entry = next((item for item in context_summary if item["file"] == relative_path), None)
    if cached_entry:
        file_description = cached_entry.get("description", "")
        logging.info(f"Using cached description for {file_path}.")

    # Parse classes and identify dependencies
    classes = parse_classes(file_path)
    if not classes:
        logging.info(f"No classes found in {file_path}. Skipping context retrieval.")
        context = ""
    else:
        # Retrieve relevant context summaries from ChromaDB
        context = get_relevant_context(collection, classes, max_tokens=MAX_TOKENS // 2)  # Allocate half tokens to context
        logging.info(f"Retrieved context with {len(tiktoken.get_encoding('gpt4').encode(context))} tokens.")

    # Construct few-shot prompt
    few_shot_prompt = construct_few_shot_prompt(
        collection= collection, 
        classes=classes, 
        max_tokens=MAX_TOKENS,
        context= context
        )

    # Add docstrings using Assistant's API
    modified_code = add_docstrings(
        assistant_id=assistant_id,
        thread_id=thread_id,
        code=original_code,
        context=few_shot_prompt
    )

    if hasattr(process_single_file, 'last_modified_code') and process_single_file.last_modified_code == modified_code:
        print(f"Old file path = {process_single_file.last_file_path}")
        exit(f"file_path = {file_path}")

    process_single_file.last_file_path = file_path
    process_single_file.last_modified_code = modified_code

    if modified_code and modified_code != original_code:
        # Ensure the header is added if not present
        modified_code = ensure_docstring_header(modified_code)
        # Show diff and ask for validation if manual flag is enabled
        if manual:
            diff = show_diff(original_code, modified_code)
            print(f"\n--- Diff for {file_path} ---\n{diff}\n--- End of Diff ---\n")
            if not prompt_user_confirmation(f"Do you approve changes for {file_path}?"):
                logging.info(f"Changes for {file_path} were not approved by the user.")
                return  # Skip applying changes

        try:
            # Backup original file only if needed
            if not check_git_repo(repo_path):
                # No Git: Always create a backup
                create_backup(file_path)
            elif check_git_repo(repo_path) and file_has_uncommitted_changes(repo_path, file_path):
                # Git is present: Backup if the file has uncommitted changes
                create_backup(file_path)

            # Update the file with modified code
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(modified_code)
            logging.info(f"Updated docstrings in {file_path}")

            # **New Step: Get Detailed File Description After Adding Docstrings**
            logging.info(f"Generating updated description for {file_path} after adding docstrings...")
            updated_file_description = generate_file_description(
                assistant_id=assistant_id,
                thread_id=thread_id, 
                file_content=modified_code
            )
            logging.info(f"Updated description for {file_path}")

            # Update context_summary with the updated description
            if cached_entry:
                # Update existing entry
                cached_entry["description"] = updated_file_description
                logging.info(f"Updated cached description for {file_path}.")
            else:
                # Append new entry
                context_summary.append({
                    "file": relative_path,
                    "description": updated_file_description
                })

            # Update cache with new hash
            new_hash = compute_sha256(file_path)
            cache[relative_path] = new_hash
            logging.info(f"Updated cache for file: {file_path}")

            # Store class summaries if any
            modified_classes = parse_classes(file_path)
            for class_name in modified_classes.keys():
                # Extract the docstring for each class
                class_docstring = extract_class_docstring(modified_code, class_name)
                if class_docstring:
                    summary = class_docstring.strip().split('\n')[0]  # First line as summary
                    store_class_summary(collection, relative_path, class_name, summary)

        except Exception as e:
            logging.error(f"Error updating file {file_path}: {e}")
    else:
        logging.info(f"No changes made to {file_path}.")
        # Update cache even if no changes to prevent reprocessing unchanged files
        current_hash = compute_sha256(file_path)
        cache[relative_path] = current_hash

@show_file_progress(desc="Checking file hashes", leave=True)
def filter_files_by_hash(file_path, repo_path, cache):
    """
    Filters a single file based on SHA-256 hash and cache.

    Args:
        file_path (str): Path to the file.
        repo_path (str): Path to the repository.
        cache (dict): Dictionary storing file hashes.

    Returns:
        str: File path if it needs processing, otherwise None.
    """
    current_hash = compute_sha256(file_path)
    cached_hash = cache.get(os.path.relpath(file_path, repo_path))
    if current_hash != cached_hash:
        return file_path  # Return files that need processing
    return None  # Skip unchanged files


@show_file_progress(desc="Uploading files to OpenAI", leave=True)
def upload_files_to_openai(file_path):
    """
    Uploads a single file to OpenAI and returns the file ID.

    Args:
        file_path (str): Path to the file.

    Returns:
        str: File ID from OpenAI or None on failure.
    """
    try:
        with open(file_path, "rb") as f:
            response = openai.files.create(
                file=f,
                purpose="assistants"
            )
        logging.info(f"Updated file : {file_path}")
        return response.id  # Return the file ID
    except Exception as e:
        logging.error(f"Error uploading {file_path} to OpenAI: {e}")
        return None
