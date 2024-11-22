# Docstring-AI 🤖✨

![License](https://img.shields.io/github/license/yourusername/docstring-ai)
![Python Version](https://img.shields.io/badge/python-3.7%2B-blue.svg)

> **Automate and Enhance Your Python Documentation with AI-Powered Precision**

---

## 📜 What is Docstring-AI?

Docstring-AI is an intelligent tool designed to **automate the generation of comprehensive docstrings** for your Python codebase. Leveraging the power of OpenAI's GPT-4o-mini and ChromaDB, Docstring-AI ensures that your functions, classes, and modules are well-documented, enhancing code readability, maintainability, and overall quality.

### 🌟 Key Features

- **Automated Docstring Generation**: Automatically add clear and concise docstrings to your Python code, explaining the purpose, parameters, return values, and exceptions of functions and classes.
  
- **Intelligent Context Management**: Utilize ChromaDB to embed and retrieve contextual information, ensuring that docstrings are accurate and consistent across the entire codebase.
  
- **Efficient Caching Mechanism**: Saving Costs

- **Knowledge Accumulation**: Build a robust knowledge base over time by maintaining context summaries for each class, enhancing the AI's ability to generate relevant and precise documentation.
  
- **Future Integrations**: Planned PyPI module, GitHub Action integration, and distribution as afor seamless incorporation into your development workflow.

---

## 🚀 Getting Started

### Prerequisites

- **Poetry** : `pip install poetry`
- **OpenAI API Key**: Obtain from the [OpenAI Dashboard](https://platform.openai.com/account/api-keys).

### Installation 

### 1. **Install Tool & Prerequisites**

1. Download
  ```bash
  git clone https://github.com/yourusername/docstring-ai.git
  cd docstring-ai
  ```

2. Install poetry
  
  `pip install poetry`

3. Add your API Key

    Open .env.template file,
    Add your OpenAI API
    Rename the file .env

4. Install The project

    `poetry install`

### 2. **Run**

  ```bash
  poetry run . --path=`YOUR PROJECT OR FOLDER PATH`
  ```

  ```bash
  poetry run . --path=`YOUR PROJECT PATH` --pr
  ```

---

## 🛤️ Roadmap

### 1. **Core Functionality**
   - **Description**: Establish the foundational features that enable Docstring-AI to traverse Python repositories, generate docstrings using OpenAI's Assistants API, manage context with ChromaDB, and implement a SHA-256 caching mechanism to optimize performance.
   - **Current Status**: Completed.


### 2. **Man-in-the-Middle Validation with Diff Preview**
   - **Description**: Introduce an interactive layer that allows developers to **preview and validate changes** before they are committed to the codebase. This feature will display diffs of proposed docstring additions, ensuring that all modifications meet quality standards.
   - **Planned Features**:
     - Generate diffs between original and modified files.
     - Provide a user-friendly interface for approving or rejecting changes.
     - Ensure code integrity by allowing manual oversight of AI-generated documentation.


### 3. **PyPI Module**
   - **Description**: Distribute Docstring-AI as a **PyPI package**, allowing developers to install and use it effortlessly within their projects. This makes it accessible for both local and global installations, facilitating seamless integration into various development environments.
   - **Planned Features**:
     - Installable via `pip install docstring-ai`.
     - Command-line interface (CLI) for flexible usage.
     - Comprehensive documentation and usage guides for easy adoption.


### 3. **GitHub Action Package**
   - **Description**: Package Docstring-AI as a **GitHub Action** to automate docstring generation within your CI/CD pipeline. This integration ensures that every commit or pull request is automatically documented, maintaining consistent and up-to-date documentation across the repository.
   - **Planned Features**:
     - Easy setup via `.github/workflows` configuration.
     - Automatic execution on specified triggers (e.g., push, pull_request).
     - Actionable logs and reports on docstring additions and modifications.

