import os
import logging
import openai
import datetime
import tiktoken
from src.utils import (
    check_git_repo,
    has_uncommitted_changes,
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
from src.prompt_utils import (
    initialize_assistant,
    update_assistant_tool_resources,
    create_thread,
    construct_few_shot_prompt

)
from src.chroma_utils import (
    initialize_chroma,
    get_or_create_collection,
    embed_and_store_files,
    get_relevant_context,
    store_class_summary
)
from src.docstring_utils import (
    add_docstrings_to_code,
    parse_classes,
    extract_class_docstring,
    extract_description_from_docstrings
)
from src.github_utils import create_github_pr
from src import MAX_TOKENS, CHROMA_COLLECTION_NAME, CACHE_FILE_NAME


def process_files_and_create_prs(repo_path: str, api_key: str, create_pr: bool, github_token: str, github_repo: str, branch_name: str, pr_name: str, pr_depth: int, manual: bool):
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

    # Step 6: Update Assistant's tool_resources with uploaded file IDs
    logging.info("\nUpdating Assistant's tool resources...")
    # Retrieve all file IDs from ChromaDB
    file_ids = [doc['id'] for doc in collection.get()['ids']]
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
    context_summary = []
    for idx, file_path in enumerate(files_to_process, 1):
        logging.info(f"\nProcessing file {idx}/{len(files_to_process)}: {file_path}")
        try:
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
            logging.info(f"Retrieved context with {len(tiktoken.get_encoding('gpt4o').encode(context))} tokens.")

        # Construct few-shot prompt
        few_shot_prompt = construct_few_shot_prompt(collection, classes, max_tokens=MAX_TOKENS)

        # Add docstrings using Assistant's API
        modified_code = add_docstrings_to_code(api_key, assistant_id, thread_id, original_code, few_shot_prompt)

        if modified_code and modified_code != original_code:
            # Show diff and ask for validation if manual flag is enabled
            if manual:
                diff = show_diff(original_code, modified_code)
                print(f"\n--- Diff for {file_path} ---\n{diff}\n--- End of Diff ---\n")
                if not prompt_user_confirmation(f"Do you approve changes for {file_path}?"):
                    logging.info(f"Changes for {file_path} were not approved by the user.")
                    continue  # Skip applying changes

            try:
                # Backup original file if Git is not present or if there are uncommitted changes
                if not git_present or uncommitted_changes:
                    create_backup(file_path)

                if manual:
                    # Ask for confirmation before applying changes
                    if not prompt_user_confirmation(f"Do you want to apply changes to {file_path}?"):
                        logging.info(f"Changes for {file_path} were not applied by the user.")
                        continue

                # Update the file with modified code
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(modified_code)
                logging.info(f"Updated docstrings in {file_path}")

                # Extract description from docstrings
                description = extract_description_from_docstrings(modified_code)
                context_summary.append({
                    "file": os.path.relpath(file_path, repo_path),
                    "description": description
                })

                # Parse classes again to extract summaries
                modified_classes = parse_classes(file_path)
                for class_name in modified_classes.keys():
                    # Extract the docstring for each class
                    class_docstring = extract_class_docstring(modified_code, class_name)
                    if class_docstring:
                        summary = class_docstring.strip().split('\n')[0]  # First line as summary
                        store_class_summary(collection, os.path.relpath(file_path, repo_path), class_name, summary)

                # Update cache with new hash
                new_hash = compute_sha256(file_path)
                cache[os.path.relpath(file_path, repo_path)] = new_hash
                logging.info(f"Updated cache for file: {file_path}")

            except Exception as e:
                logging.error(f"Error updating file {file_path}: {e}")
        else:
            logging.info(f"No changes made to {file_path}.")
            # Update cache even if no changes to prevent reprocessing unchanged files
            current_hash = compute_sha256(file_path)
            cache[os.path.relpath(file_path, repo_path)] = current_hash

        if idx > 5 :
            exit("ixd > 5")

    # Step 10: Save Context Summary
    context_summary_path = os.path.join(repo_path, "context_summary.json")
    try:
        with open(context_summary_path, "w", encoding='utf-8') as f:
            json.dump(context_summary, f, indent=2)
        logging.info(f"\nDocstring generation completed. Context summary saved to '{context_summary_path}'.")
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
                    repo_path = repo_path, 
                    github_token = github_token, 
                    github_repo = github_repo, 
                    branch_name = folder_branch_name, 
                    pr_name= folder_pr_name
                    )
