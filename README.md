

> [!IMPORTANT]
> **Safe to use (create PR & Backup) but will add docstring to a few files and crashes. It is in development not production ready**

# Docstring-AI 🤖✨

![License](https://img.shields.io/github/license/ph-ausseil/docstring-ai)
![Python Version](https://img.shields.io/badge/python-3.7%2B-blue.svg)

> **Automate and Enhance Your Python Documentation with AI-Powered Precision**

---

## 📜 What is Docstring-AI?

Docstring-AI is an advanced tool that **automates the generation of high-quality Python docstrings** for your codebase. Built with OpenAI's GPT-4o-mini and ChromaDB, it transforms your documentation process, improving code clarity, maintainability, and collaboration.

### 🌟 Key Features

- **Automated Docstring Generation**: Leverages AI to generate comprehensive and context-aware docstrings.
- **Automatic Detection of Git Repository**: Detects if a folder is a Git repository and integrates seamlessly.
- **Integration with ChromaDB**: Uses vector search for similarity-based context enrichment.
- **Efficient Caching**: Reduces API calls by implementing a SHA-256-based caching mechanism.
- **Incremental Knowledge Building**: Processes files incrementally, building context to improve docstring quality over time.
- **GitHub Integration**: Automatically creates pull requests with documentation updates.
- **Manual Validation with Diff Preview**: Enables optional manual review to ensure changes align with your requirements.


---

## 📚 Table of Contents

1. [📜 What is Docstring-AI?](#-what-is-docstring-ai)
2. [🌟 Key Features](#-key-features)
3. [🚀 Getting Started](#-getting-started)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
4. [🏃‍♂️ Running Docstring-AI](#️-running-docstring-ai)
   - [Basic Usage](#basic-usage)
   - [With GitHub Pull Request Creation](#with-github-pull-request-creation)
   - [Enabling Manual Validation](#enabling-manual-validation)
5. [📝 Understanding Flags](#-understanding-flags)
6. [🔍 Detailed Explanation of `--pr-depth`](#-detailed-explanation-of---pr-depth)
7. [🔍 Manual Validation Workflow](#-manual-validation-workflow)
8. [🛤️ Roadmap](#️-roadmap)
9. [📚 Additional Information](#-additional-information)
10. [☕ Support the Project](#-support-the-project)
11. [🤝 Contributing](#-contributing)
12. [📄 License](#-license)
13. [🧑‍💻 Contact](#-contact)
---

## 🚀 Getting Started

### Prerequisites

- **Poetry**: Install via pip
```bash
pip install poetry
```
  
- **OpenAI API Key**: Obtain from the [OpenAI Dashboard](https://platform.openai.com/account/api-keys).
  
- **Git (Optional)**: For repositories where you want to create pull requests. Install from [Git Downloads](https://git-scm.com/downloads).

---

### Installation

1. **Clone the Repository**
```bash
git clone https://github.com/yourusername/docstring-ai.git
cd docstring-ai
```

2. **Install Poetry**
```bash
pip install poetry
```

3. **Configure Environment Variables**
    - Copy the template and add your OpenAI API key:
```bash
cp .env.template .env
```
    - Open the `.env` file and set your `OPENAI_API_KEY`. If you plan to use GitHub PR features, also set `GITHUB_TOKEN` and `GITHUB_REPO` as needed.

4. **Install Project Dependencies**
```bash
poetry install
```

---

## 🏃‍♂️ Running Docstring-AI

### 🏃‍♂️ Basic Usage

Run Docstring-AI to automatically generate docstrings for your Python repository:

```bash
poetry run . --path=/path/to/repo
```

> [!TIP]
> Use the `--use-repo-config` flag for seamless Git integration. It automatically detects the repository configuration.

```bash
poetry run . --path=/path/to/repo --use-repo-config
```

---

### 📦 With GitHub Pull Request Creation

Enable GitHub integration to create pull requests for your changes:

```bash
poetry run . --path=/path/to/repo --pr=yourusername/yourrepo --github-token=YOUR_GITHUB_TOKEN
```

---

### 🛠️ Enabling Manual Validation

Enable manual review of changes to ensure they align with your requirements:

```bash
poetry run . --path=/path/to/repo --manual
```

---

### 🛡️ Combining Flags

Combine multiple flags for greater flexibility, such as GitHub integration and manual validation:

```bash
poetry run . --path=/path/to/repo --pr=yourusername/yourrepo --github-token=YOUR_GITHUB_TOKEN --manual
```

---

## 📝 Understanding Flags

### General Flags

| Flag             | Description                                                                                                          |
|------------------|----------------------------------------------------------------------------------------------------------------------|
| `--path`         | **(Required)** Path to the repository or folder containing Python files.                                            |
| `--api_key`      | OpenAI API key. Defaults to the `OPENAI_API_KEY` environment variable if not provided.                              |
| `--manual`       | Enable manual validation with diff preview before committing changes or creating PRs.                               |
| `--help-flags`   | List and describe all available flags.                                                                               |

---

### GitHub Integration Flags

| Flag             | Description                                                                                                          |
|------------------|----------------------------------------------------------------------------------------------------------------------|
| `--pr`           | GitHub repository for PR creation in the format `owner/repository`. Overrides the `GITHUB_REPO` environment variable if provided. |
| `--target-branch`| Repository branch that is the target of the created PR (Default: Suggest the current branch).                        |
| `--github-token` | GitHub personal access token. Defaults to the `GITHUB_TOKEN` environment variable if not provided.                  |
| `--branch-name`  | Branch name for the PR. Auto-generated if not provided.                                                             |
| `--pr-name`      | Custom name for the pull request. Defaults to `-- Add docstrings for files in 'path'.`                              |
| `--pr-depth`     | Depth level for creating PRs per folder. Defaults to `2`.                                                           |

---

### Advanced Flags

| Flag             | Description                                                                                                          |
|------------------|----------------------------------------------------------------------------------------------------------------------|
| `--use-repo-config` | Use the repository’s Git configuration for GitHub integration. Overrides other GitHub-related parameters.         |
| `--no-cache`     | Execute the script without cached values by deleting cache files.                                                   |


---

## 🔍 Detailed Explanation of Git Flags

The Git-related flags control how Docstring-AI integrates with GitHub and your repository’s configuration. Understanding their precedence and behavior ensures optimal usage for pull request creation.

### How It Works:

1. **`--use-repo-config` (Highest Priority)**:  
   If this flag is used, the tool will detect and prioritize the repository’s Git configuration. This overrides other GitHub-related flags (`--pr`, `--github-token`, etc.) and environment variables.
   - Example: If `--use-repo-config` is specified, the detected repository URL and branch will be used regardless of other provided flags.

2. **Explicit Flags (`--pr`, `--github-token`, etc.)**:  
   If `--use-repo-config` is not provided, the tool defaults to explicitly specified flags. These flags override values from the `.env` file or environment variables.
   - Example: Specifying `--pr=myuser/myrepo` and `--github-token=TOKEN` will use these values instead of those in `.env`.

3. **Environment Variables (`GITHUB_REPO`, `GITHUB_TOKEN`, etc.)**:  
   If neither `--use-repo-config` nor explicit flags are provided, the tool falls back to environment variables set in the `.env` file.

4. **Fallback Behavior**:  
   If no configuration (flags or environment variables) is available, GitHub pull request creation is skipped. Files will be directly modified in place.

---

### Priority and Behavior Overview

| Configuration Method       | Priority     | Description                                                                                      |
|----------------------------|--------------|--------------------------------------------------------------------------------------------------|
| `--use-repo-config`        | **Highest**  | Detects repository and branch automatically, overriding all other settings.                     |
| Explicit Flags (`--pr`, etc.) | High       | Manually specifies repository and GitHub token, overriding environment variables.               |
| Environment Variables      | Medium       | Reads `GITHUB_REPO` and `GITHUB_TOKEN` from `.env` or system environment.                       |
| Fallback                   | Lowest      | Skips GitHub integration entirely, modifying files locally.                                     |

---

## 🔍 Priority & Overwrite Relationships of Git Flags

Docstring-AI uses a structured hierarchy to determine which Git configuration to use for repository and branch management. The behavior is designed to provide flexibility while leveraging existing configurations effectively.

---

### GitHub Repository (`--pr`)

> [!NOTE]
> By default, the application tries to detect the local repository and suggest. User confirmation will be requested.


> [!TIP]
> `--use-repo-config` skips the confirmations process.

1. **`--use-repo-config` (Highest Priority)**:  
   - Automatically detects the repository from the local Git configuration and uses it.  
   - Overrides all other configurations, including `--pr` and `GITHUB_REPO`.  


2. **Local Repository Detection**:  
   - If `--use-repo-config` is not enabled, the tool attempts to detect if the provided path is part of a local Git repository.  
   - The detected repository is suggested for confirmation before proceeding.  

3. **`--pr` Flag**:  
   - Manually specify the repository using the `--pr` flag.  
   - Overrides the `GITHUB_REPO` environment variable if provided.

4. **Environment Variable (`GITHUB_REPO`)**:  
   - Used as a fallback if no other configuration is specified.


---

### GitHub Token (`--github-token`)
1. **Command-Line Flag**:  
   - Highest priority when provided via the `--github-token` flag.  
   - Overrides the `GITHUB_TOKEN` environment variable.  

2. **Environment Variable (`GITHUB_TOKEN`)**:  
   - Used as a fallback if no explicit token is provided.

> [!WARNING]
> Ensure a GitHub token is provided via `--github-token` or `GITHUB_TOKEN` when using GitHub integration to avoid authentication issues.

---

### Target Branch (`--target-branch`)
1. **Default Behavior**:  
   - Defaults to the current branch detected from the local Git repository.  

2. **Overrides**:  
   - Can be explicitly set using the `--target-branch` flag or via the `GITHUB_TARGET_BRANCH` environment variable.

> [!TIP]
> Default branch detection ensures that pull requests target the branch you're actively working on, reducing manual configuration.

---

### Branch Name (`--branch-name`)
1. **Default Behavior**:  
   - Automatically generated if not specified.  
     Example: `feature/docstring-updates-YYYYMMDDHHMMSS`.  

2. **Overrides**:  
   - Can be manually specified using the `--branch-name` flag to ensure consistency or adhere to naming conventions.

---

### Summary Table

| Configuration     | Priority (Highest to Lowest)                                                | Default Behavior                          |
|-------------------|----------------------------------------------------------------------------|-------------------------------------------|
| Repository (`--pr`) | 1. `--use-repo-config` → 2. Detect Local Config → 3. `--pr` → 4. `GITHUB_REPO` | Detects local repository and suggests it. |
| Token (`--github-token`) | 1. `--github-token` → 2. `GITHUB_TOKEN`                               | None. Must be explicitly provided.        |
| Target Branch (`--target-branch`) | 1. `--target-branch` → 2. `GITHUB_TARGET_BRANCH`              | Defaults to the current branch.           |
| Branch Name (`--branch-name`) | 1. `--branch-name`                                              | Auto-generated.                           |

---

### Example Scenario

If you run the following command:
```bash
poetry run . --path=/path/to/repo --pr=myuser/myrepo
```


## 🔍 Detailed Explanation of `--pr-depth`

The `--pr-depth` argument controls how changes are grouped into pull requests based on folder structure. It allows for flexibility in organizing PRs, especially for large repositories.

### How it Works:

- **Depth `0`**: Each Python file will have its own pull request. This is ideal for repositories with minimal interdependencies between files.
- **Depth `1`**: All Python files within the same immediate folder (and its subfolders) will be grouped into a single PR.
- **Depth `2` (Default)**: Python files are grouped by their second-level folder hierarchy. For example:
```
/repo
├── folderA/
│   ├── subfolderA1/
│   ├── subfolderA2/
├── folderB/
    ├── subfolderB1/
```
  - Files in `folderA` and its subfolders are grouped into one PR.
  - Files in `folderB` and its subfolders are grouped into another PR.

### Use Cases:

1. **Small Repositories**: Use `--pr-depth=0` for fine-grained PRs that isolate changes per file.
2. **Medium Repositories**: Use `--pr-depth=1` to consolidate related changes within folders.
3. **Large Repositories**: Use the default `--pr-depth=2` to group changes at higher levels, reducing the number of PRs.

---

## 🔍 Manual Validation Workflow

When the `--manual` flag is enabled, Docstring-AI introduces an interactive review process to ensure that all changes are intentional and meet your quality standards.

### Steps:

1. **Diff Preview**:
    - After generating modified code with added docstrings, Docstring-AI displays a unified diff between the original and modified files.
    - Example:
```diff
--- original
+++ modified
@@ -1,3 +1,5 @@
 def example_function(param1, param2):
+    """
+    Adds two parameters.
+
     # Function implementation
```

2. **User Confirmation**:
    - You are prompted to approve or reject the changes for each file.
    - Respond with `yes` to apply the changes or `no` to skip them.

3. **Applying Changes**:
    - Approved changes are either directly applied to the files or prepared for PR creation, depending on your setup.

4. **PR Confirmation** (if Git is present):
    - Before creating pull requests, you are prompted to confirm the creation of PRs for the categorized folders.

---

## 🛤️ Roadmap

### 1. **Core Functionality**
   - **Description**: Establish the foundational features that enable Docstring-AI to traverse Python repositories, generate docstrings using OpenAI's Assistants API, manage context with ChromaDB, and implement a SHA-256 caching mechanism to optimize performance.
   - **Current Status**: Completed.

### 2. **PyPI Module**
   - **Description**: Distribute Docstring-AI as a **PyPI package**, allowing developers to install and use it effortlessly within their projects. This makes it accessible for both local and global installations, facilitating seamless integration into various development environments.

### 3. **GitHub Action Package**
   - **Description**: Package Docstring-AI as a **GitHub Action** to automate docstring generation within your CI/CD pipeline. This integration ensures that every commit or pull request is automatically documented, maintaining consistent and up-to-date documentation across the repository.

### 4. **Granularity**
   - **Description**: Leveraging Structured Outputs , choose which docstring to generate : Modules, Classes, Functions & Constants levels

### 5. **Console improvements** : 
   - **Description**: Use of the tdmq wrapper & enhance the logging class

---

## 📚 Additional Information

### 🔧 Configuration

- **Environment Variables**:
  - `OPENAI_API_KEY`: Your OpenAI API key.
  - `GITHUB_TOKEN`: Your GitHub personal access token.
  - `GITHUB_REPO`: GitHub repository in the format `owner/repository`.

- **.env File**:
  - Create a `.env` file by copying the template:
```bash
cp .env.template .env
```
  - Populate the necessary environment variables in the `.env` file.

### 🛠️ Development

- **Logging**:
  - All operations and errors are logged to `docstring_ai.log` for easy debugging and monitoring.

- **Caching**:
  - Implements a SHA-256 caching mechanism (`docstring_cache.json`) to track changes and optimize performance by avoiding redundant processing.

### 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

### 📄 License

This project is licensed under the [MIT License](https://github.com/yourusername/docstring-ai/blob/main/LICENSE).

---

## 🧑‍💻 Contact

For any questions, suggestions, or feedback, please reach out to me on LinkedIn

<!-- 

## ☕ Support the Project


If you found this project useful and want to show your support, you can:

<!--
### 💖 Buy Me a Coffee

[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-Support-orange)](https://www.buymeacoffee.com/yourusername)

 Replace "yourusername" with your actual Buy Me a Coffee username.

Your support helps keep this project alive and well-maintained. Thank you for contributing to its growth!

---

### 💰 Bitcoin Tipping

If you prefer crypto, feel free to send Bitcoin to the following address:

**Bitcoin Address**: `your-bitcoin-address-here`


You can also scan the QR code below to send Bitcoin directly:

![Bitcoin QR Code](path/to/qr-code.png)

---

Thank you for your generosity and for supporting open-source software! ❤️
