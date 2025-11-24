# Branch Protection Setup Guide

This guide explains how to set up branch protection rules for the `master` branch on GitHub.

## 🔒 Setting Up Branch Protection Rules

### Step 1: Navigate to Repository Settings

1. Go to your GitHub repository: `https://github.com/iampique/ai_for_supply_chain`
2. Click on **Settings** (top right, requires admin access)
3. In the left sidebar, click **Branches**

### Step 2: Add Branch Protection Rule

1. Under **Branch protection rules**, click **Add rule** or **Add branch protection rule**
2. In the **Branch name pattern** field, enter: `master`

### Step 3: Configure Protection Settings

Enable the following settings:

#### ✅ Required Settings

- [x] **Require a pull request before merging**
  - [x] Require approvals: `1` (or more)
  - [x] Dismiss stale pull request approvals when new commits are pushed
  - [x] Require review from Code Owners (if you have CODEOWNERS file)

- [x] **Require status checks to pass before merging**
  - [x] Require branches to be up to date before merging
  - [ ] (Optional) Add specific status checks if you have CI/CD

- [x] **Require conversation resolution before merging**
  - Ensures all PR comments are addressed

- [x] **Do not allow bypassing the above settings**
  - Prevents even admins from bypassing rules

#### ✅ Recommended Settings

- [x] **Restrict who can push to matching branches**
  - Leave empty (no one can push directly)

- [x] **Do not allow force pushes**
  - Prevents rewriting history

- [x] **Do not allow deletions**
  - Prevents accidental branch deletion

- [x] **Require linear history**
  - (Optional) Ensures clean commit history

### Step 4: Save the Rule

Click **Create** or **Save changes** at the bottom of the page.

## 📋 Summary of Protection Rules

Once configured, the `master` branch will have:

| Rule | Description |
|------|-------------|
| **Direct Pushes** | ❌ Prohibited for everyone (including admins) |
| **Pull Request Required** | ✅ All changes must go through PRs |
| **Review Required** | ✅ At least 1 approval needed |
| **Status Checks** | ✅ All CI/CD checks must pass |
| **Up-to-Date** | ✅ Branch must be synced with master |
| **Force Push** | ❌ Not allowed |
| **Branch Deletion** | ❌ Not allowed |

## 🔧 Additional Configuration (Optional)

### Code Owners

Create `.github/CODEOWNERS` file to automatically request reviews:

```
# Default owners
* @iampique

# Specific paths
/src/agents/ @iampique
/docs/ @iampique
```

### Required Status Checks

If you set up GitHub Actions CI/CD, you can require specific checks:

- `lint` - Code linting
- `test` - Unit tests
- `build` - Build verification

## ✅ Verification

After setting up protection rules:

1. Try to push directly to `master` (should fail)
2. Create a test branch and PR
3. Verify that PR requires approval
4. Verify that status checks run

## 📝 Notes

- Branch protection rules only apply to the `master` branch
- Contributors can still create branches and push to their forks
- All merges must go through Pull Requests
- Maintainers can still merge approved PRs

## 🆘 Troubleshooting

**Issue**: "You don't have permission to push to this branch"
- **Solution**: This is expected! Create a branch and open a PR instead.

**Issue**: "Required status check is not set"
- **Solution**: Either disable status checks requirement or set up CI/CD workflows.

**Issue**: "Can't merge: branch is out of date"
- **Solution**: Update your branch with `git rebase origin/master` or merge master into your branch.

---

For contributors, see [CONTRIBUTING.md](../CONTRIBUTING.md) for the development workflow.

