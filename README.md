# Docstring-AI 🤖✨

![License](https://img.shields.io/github/license/yourusername/docstring-ai)
![Python Version](https://img.shields.io/badge/python-3.7%2B-blue.svg)

> **Automate and Enhance Your Python Documentation with AI-Powered Precision**

---

## 📜 What is Docstring-AI?

Docstring-AI is an intelligent tool designed to **automate the generation of comprehensive docstrings** for your Python codebase. Leveraging the power of OpenAI's GPT-4o-mini and ChromaDB, Docstring-AI ensures that your functions, classes, and modules are well-documented, enhancing code readability, maintainability, and overall quality.

### 🌟 Key Features

- **Automated Docstring Generation**
- **Intelligent Context Management**: With Vector search for similarity adding context (ChromaDB).
- **Efficient Caching Mechanism**: Catching mechanism for cost efficiency.
- **Knowledge Accumulation**: Algorith that start with small file, build up knowledge to give more context to file being processed over time.
- **Manual Validation with Diff Preview**: Option to manualy review each change (Back up system in place anyway)

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

### Basic Usage

To run Docstring-AI and add docstrings to your Python repository:

```bash
poetry run . --path=/path/to/repo
```

### With GitHub Pull Request Creation

To enable automatic creation of GitHub pull requests for your changes:

```bash
poetry run . --path=/path/to/repo --pr=yourusername/yourrepo --github-token=YOUR_GITHUB_TOKEN
```

### Enabling Manual Validation

To enable manual review and validation of changes before they are applied or submitted:

```bash
poetry run . --path=/path/to/repo --manual
```

Combine flags as needed:

```bash
poetry run . --path=/path/to/repo --pr=yourusername/yourrepo --github-token=YOUR_GITHUB_TOKEN --manual
```

---

## 📝 Understanding Flags

| Flag             | Description                                                                                                                                                           |
|------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--path`         | **(Required)** Path to the repository or folder containing Python files.                                                                                                |
| `--api_key`      | OpenAI API key. Defaults to the `OPENAI_API_KEY` environment variable if not provided.                                                                                 |
| `--pr`           | GitHub repository for PR creation in the format `owner/repository`. Overrides the `GITHUB_REPO` environment variable if provided.                                                  |
| `--target-branch`| Repositiry branch that is the target of the created PR (Default: Sugest the current branch)                                         |
| `--github-token` | GitHub personal access token. Defaults to the `GITHUB_TOKEN` environment variable if not provided.                                                                     |
| `--branch-name`  | Branch name for the PR. Auto-generated if not provided.                                                                                                                |
| `--pr-name`      | Custom name for the pull request. Defaults to `-- Add docstrings for files in 'path'.`                                                                                  |
| `--pr-depth`     | Depth level for creating PRs per folder. Defaults to `2`.                                                                                                              |
| `--manual`       | Enable manual validation with diff preview before committing changes or creating PRs.                                                                                   |
| `--help-flags`   | List and describe all available flags.                                                                                                                                |

---

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
