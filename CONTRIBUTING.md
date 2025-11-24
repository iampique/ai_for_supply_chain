# Contributing to Automotive Supply Chain AI

Thank you for your interest in contributing to this project! This document outlines the contribution process and guidelines to ensure a smooth collaboration experience.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Branch Protection Rules](#branch-protection-rules)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Requirements](#testing-requirements)
- [Commit Message Guidelines](#commit-message-guidelines)
- [Documentation](#documentation)

## 🤝 Code of Conduct

By participating in this project, you agree to:

- Be respectful and inclusive
- Welcome constructive feedback
- Focus on what is best for the community
- Show empathy towards other contributors

## 🚀 Getting Started

### 1. Fork the Repository

```bash
# Fork the repository on GitHub, then clone your fork
git clone git@github.com:YOUR_USERNAME/ai_for_supply_chain.git
cd ai_for_supply_chain
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
# Edit .env with your credentials (see README.md for details)
```

### 3. Create a Branch

**Important**: Never commit directly to `master` branch. Always create a feature branch.

```bash
# Update master branch
git checkout master
git pull origin master

# Create a new branch for your feature
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
# or
git checkout -b docs/update-readme
```

## 🔒 Branch Protection Rules

This repository has branch protection rules enabled for the `master` branch. Contributors should be aware of the following:

### Protected Branch: `master`

The following rules apply to the `master` branch:

1. **Direct Pushes Prohibited**: No one can push directly to `master` (including repository admins)
2. **Pull Request Required**: All changes must go through a Pull Request
3. **Review Required**: At least one approval from a maintainer is required before merging
4. **Status Checks Required**: All CI/CD checks must pass before merging
5. **Up-to-Date Required**: Branch must be up-to-date with `master` before merging
6. **No Force Pushes**: Force pushes and branch deletion are not allowed

### What This Means for Contributors

- ✅ Create a feature branch from `master`
- ✅ Make your changes and commit to your branch
- ✅ Push your branch to your fork
- ✅ Open a Pull Request targeting `master`
- ✅ Wait for review and address feedback
- ✅ Once approved, maintainers will merge your PR

- ❌ Do not try to push directly to `master`
- ❌ Do not force push to `master`
- ❌ Do not delete the `master` branch

## 🔄 Development Workflow

### 1. Keep Your Branch Updated

Regularly sync your branch with the master branch:

```bash
# Fetch latest changes
git fetch origin

# Rebase your branch on top of master
git checkout feature/your-feature-name
git rebase origin/master

# If conflicts occur, resolve them and continue
git rebase --continue
```

### 2. Make Your Changes

- Write clean, readable code
- Follow the coding standards (see below)
- Add tests for new functionality
- Update documentation as needed

### 3. Test Your Changes

```bash
# Run basic import tests
python -c "from src.agents.parts_discovery_agent import PartsDiscoveryAgent; print('✓ OK')"

# Test your specific changes
python -m src.demo_agentic_workflow  # If modifying workflow
# Or run relevant demo scripts
```

### 4. Commit Your Changes

Follow the commit message guidelines (see below).

```bash
git add .
git commit -m "feat: add new feature description"
```

## 📝 Pull Request Process

### Before Submitting

- [ ] Code follows the project's coding standards
- [ ] Tests pass locally
- [ ] Documentation is updated (if needed)
- [ ] Commit messages follow the guidelines
- [ ] Branch is up-to-date with `master`
- [ ] No merge conflicts

### Creating a Pull Request

1. **Push your branch** to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Open a Pull Request** on GitHub:
   - Go to the repository on GitHub
   - Click "New Pull Request"
   - Select your branch and target `master`
   - Fill out the PR template (if available)

3. **PR Title Format**:
   ```
   type: Brief description
   ```
   Examples:
   - `feat: Add supplier risk scoring visualization`
   - `fix: Resolve Qdrant connection timeout issue`
   - `docs: Update README with new setup instructions`

4. **PR Description** should include:
   - What changes were made
   - Why the changes were needed
   - How to test the changes
   - Any breaking changes
   - Screenshots (if UI changes)

### PR Review Process

1. **Automated Checks**: GitHub Actions will run tests and checks
2. **Code Review**: A maintainer will review your code
3. **Feedback**: Address any requested changes
4. **Approval**: Once approved, your PR will be merged

### After Approval

- Maintainers will merge your PR
- Your branch can be deleted after merging
- Your contribution will be visible in the project history

## 💻 Coding Standards

### Python Style Guide

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide
- Use type hints for function parameters and return types
- Maximum line length: 100 characters
- Use meaningful variable and function names

### Code Organization

```python
# Imports: Standard library, third-party, local
import json
from pathlib import Path
from typing import Dict, List

from qdrant_client import QdrantClient
from loguru import logger

from src.config import settings
```

### Documentation

- Add docstrings to all functions and classes
- Use Google-style docstrings:
  ```python
  def example_function(param: str) -> bool:
      """
      Brief description of the function.
      
      Args:
          param: Description of parameter
      
      Returns:
          Description of return value
      
      Raises:
          ValueError: When parameter is invalid
      """
  ```

### Error Handling

- Use specific exception types
- Log errors appropriately
- Provide helpful error messages

## 🧪 Testing Requirements

### Before Submitting a PR

- [ ] All existing tests pass
- [ ] New functionality has tests (if applicable)
- [ ] Code runs without errors
- [ ] No linter errors

### Running Tests

```bash
# Test imports
python -c "from src.agents.parts_discovery_agent import PartsDiscoveryAgent; print('✓ OK')"

# Test Qdrant connection
python -c "from src.qdrant_client import get_qdrant_client; get_qdrant_client(); print('✓ Connected')"

# Run demo scripts to verify functionality
python -m src.demo_acorn_comparison
```

## 📝 Commit Message Guidelines

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

### Format

```
<type>: <subject>

<body>

<footer>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

### Examples

```
feat: Add ACORN algorithm comparison demo

Implements comprehensive demonstration of Qdrant 1.16 ACORN
algorithm with three different filter selectivity scenarios.

Closes #123
```

```
fix: Resolve Qdrant connection timeout issue

Increased timeout from 10s to 30s to handle slow connections.
Added retry logic for transient failures.

Fixes #456
```

```
docs: Update README with contribution guidelines

Adds CONTRIBUTING.md and updates README with contribution
process and branch protection information.
```

## 📚 Documentation

### When to Update Documentation

- Adding new features
- Changing existing functionality
- Fixing bugs that affect user experience
- Adding new configuration options

### Documentation Files

- `README.md`: Main project documentation
- `CONTRIBUTING.md`: This file
- `TEST_WORKFLOW.md`: Testing guide
- Code docstrings: Inline documentation

## 🐛 Reporting Issues

### Before Reporting

1. Check if the issue already exists
2. Verify it's not a configuration issue
3. Try to reproduce the issue

### Issue Template

When creating an issue, include:

- **Description**: Clear description of the issue
- **Steps to Reproduce**: How to reproduce the issue
- **Expected Behavior**: What should happen
- **Actual Behavior**: What actually happens
- **Environment**: Python version, OS, etc.
- **Logs**: Relevant error messages or logs

## ✅ Checklist for Contributors

Before submitting your PR, ensure:

- [ ] Code follows PEP 8 style guide
- [ ] All tests pass
- [ ] Documentation is updated
- [ ] Commit messages follow guidelines
- [ ] Branch is up-to-date with `master`
- [ ] PR description is clear and complete
- [ ] No sensitive data (API keys, passwords) in code
- [ ] `.env` file is not committed

## 🙏 Thank You!

Your contributions make this project better for everyone. We appreciate your time and effort!

## 📧 Questions?

If you have questions about contributing:

- Open an issue with the `question` label
- Check existing issues and discussions
- Review the README.md for setup instructions

---

**Happy Contributing! 🚀**

