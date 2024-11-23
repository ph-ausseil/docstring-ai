"""
This module processes Python files to add docstrings using OpenAI's Assistant,
embeds the files in ChromaDB, and integrates with GitHub for pull request creation.

Functions:
- process_files_and_create_prs: Processes Python files, adds docstrings, and creates pull requests.
"""

import json
import os
import logging
import openai
import datetime
import tiktoken
from docstring_ai.lib.utils import (
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
from docstring_ai import MAX_TOKENS, CHROMA_COLLECTION_NAME, CACHE_FILE_NAME, CONTEXT_SUMMARY_PATH

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
    openai.api_key = api_key

    # Check if Git is present
    git_present = check_git_repo(repo_path)

    # Check for uncommitted changes if Git is present
    uncommitted_changes = False
    if git_present:
        uncommitted_changes = has_uncommitted_changes(repo_path)

    # Initialize ChromaDB
    logging.info("\nInitializing ChromaDB...")
    chroma_client = initialize_chroma()
    collection = get_or_create_collection(chroma_client, CHROMA_COLLECTION_NAME)

    # Load cache
    cache_path = os.path.join(repo_path, CACHE_FILE_NAME)
    cache = load_cache(cache_path)

    # Load Context Summary
    context_summary_full_path = os.path.join(repo_path, CONTEXT_SUMMARY_PATH)
    if os.path.exists(context_summary_full_path):
        try:
            with open(context_summary_full_path, 'r', encoding='utf-8') as f:
                context_summary = json.load(f)
            logging.info(f"Loaded context summary with {len(context_summary)} entries.")
        except Exception as e:
            logging.error(f"Error loading context summary: {e}")
            context_summary = []
    else:
        logging.info(f"No existing context summary found at '{context_summary_full_path}'. Starting fresh.")
        context_summary = []

    # Step 1: Retrieve all Python files
    python_files = get_python_files(repo_path)
    logging.info(f"Found {len(python_files)} Python files to process.")

    if not python_files:
        logging.info("No Python files found. Exiting.")
        return

    # Step 2: Sort files by size (ascending)
    python_files_sorted = sort_files_by_size(python_files)

    # Step 3: Compute SHA-256 hashes and filter out unchanged files
    files_to_process = []
    for file_path in python_files_sorted:
        current_hash = compute_sha256(file_path)
        cached_hash = cache.get(os.path.relpath(file_path, repo_path))
        if current_hash == cached_hash:
            logging.info(f"Skipping unchanged file: {file_path}")
        else:
            files_to_process.append(file_path)

    logging.info(f"\n{len(files_to_process)} files to process after cache check.")

    if not files_to_process:
        logging.info("No files need processing. Exiting.")
        return

    # Step 4: Embed and store files in ChromaDB
    logging.info("\nEmbedding and storing Python files in ChromaDB...")
    embed_and_store_files(collection, files_to_process)

    # Step 5: Initialize Assistant
    logging.info("\nInitializing Assistant...")
    assistant_id = initialize_assistant(api_key)
    if not assistant_id:
        logging.error("Assistant initialization failed. Exiting.")
        return

    # Step 6: Upload files to OpenAI and update Assistant's tool resources
    logging.info("\nUploading files to OpenAI and updating Assistant's tool resources...")
    file_ids = []  # List to store file IDs returned by OpenAI

    for file_path in files_to_process:
        try:
            # Upload the file to OpenAI
            with open(file_path, "rb") as f:
                response = openai.files.create(
                    file=f,
                    purpose="assistants"
                )
            file_id = response.id
            if file_id:
                file_ids.append(file_id)
                logging.info(f"Uploaded {file_path} to OpenAI with file ID: {file_id}")
            else:
                logging.warning(f"Failed to retrieve file ID for {file_path}")
        except Exception as e:
            logging.error(f"Error uploading {file_path} to OpenAI: {e}")

    if not file_ids:
        logging.error("No files were successfully uploaded to OpenAI. Exiting.")
        return

    # Update Assistant's tool resources with OpenAI file IDs
    update_assistant_tool_resources(api_key, assistant_id, file_ids)

    # Step 7: Create a Thread
    logging.info("\nCreating a new Thread...")
    thread_id = create_thread(api_key, assistant_id)
    if not thread_id:
        logging.error("Thread creation failed. Exiting.")
        return

    # Step 8: Traverse repository and categorize folders based on pr_depth
    logging.info("\nTraversing repository to categorize folders based on pr_depth...")
    folder_dict = traverse_repo(repo_path, pr_depth)
    logging.info(f"Found {len(folder_dict)} depth levels up to {pr_depth}.")

    # Step 9: Process Each Python File for Docstrings
    logging.info("\nProcessing Python files to add docstrings...")
    for idx, file_path in enumerate(files_to_process, 1):
        logging.info(f"\nProcessing file {idx}/{len(files_to_process)}: {file_path}")
        try:
            relative_path = os.path.relpath(file_path, repo_path)
            with open(file_path, 'r', encoding='utf-8') as f:
                original_code = f.read()
        except Exception as e:
            logging.error(f"Error reading file {file_path}: {e}")
            continue

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
        few_shot_prompt = construct_few_shot_prompt(collection, classes, max_tokens=MAX_TOKENS)

        # Check if file is cached and has existing description
        cached_entry = next((item for item in context_summary if item["file"] == relative_path), None)
        if cached_entry:
            file_description = cached_entry.get("description", "")
            logging.info(f"Using cached description for {file_path}.")
        else:
            # **New Step: Get Detailed File Description Using Assistant's API**
            logging.info(f"Generating detailed description for {file_path}...")
            file_description = generate_file_description(
                assistant_id=assistant_id,  # Modify as needed if using assistant objects
                thread_id=thread_id, 
                file_content=original_code
            )
            logging.info(f"Description for {file_path}: {file_description}")
            # Append to context_summary for future use
            context_summary.append({
                "file": relative_path,
                "description": file_description
            })

        # Add docstrings using Assistant's API
        modified_code = add_docstrings(
            assistant_id=assistant_id,
            thread_id=thread_id,
            code=original_code,
            context=few_shot_prompt
        )

        if modified_code and modified_code != original_code:
            # Show diff and ask for validation if manual flag is enabled
            if manual:
                diff = show_diff(original_code, modified_code)
                print(f"\n--- Diff for {file_path} ---\n{diff}\n--- End of Diff ---\n")
                if not prompt_user_confirmation(f"Do you approve changes for {file_path}?"):
                    logging.info(f"Changes for {file_path} were not approved by the user.")
                    continue  # Skip applying changes

            try:
                # Backup original file only if needed
                if not git_present:
                    # No Git: Always create a backup
                    create_backup(file_path)
                elif git_present and file_has_uncommitted_changes(repo_path, file_path):
                    # Git is present: Backup if the file has uncommitted changes
                    create_backup(file_path)

                # Update the file with modified code
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(modified_code)
                logging.info(f"Updated docstrings in {file_path}")

                # **New Step: Get Detailed File Description After Adding Docstrings**
                logging.info(f"Generating updated description for {file_path} after adding docstrings...")
                updated_file_description = generate_file_description(
                    assistant_id=assistant_id,  # Modify as needed if using assistant objects
                    thread_id=thread_id, 
                    file_content=modified_code
                )
                logging.info(f"Updated description for {file_path}: {updated_file_description}")

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
                folder_branch_name = f"feature/docstrings-folder-{folder_rel_path}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

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
