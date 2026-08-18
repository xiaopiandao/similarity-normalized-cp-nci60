# Manual GitHub upload

## Browser method

1. Create an empty GitHub repository without adding a README, `.gitignore`, or license.
2. Extract this ZIP locally.
3. On the new repository page, choose **uploading an existing file**.
4. Upload the extracted contents so that `README.md`, `scripts/`, `experiments/`, `tests/`, and `splits/` appear at the repository root.
5. Commit the upload to a new branch such as `reproducibility-code`, then open a pull request to `main` if desired.

GitHub's browser uploader may be inconvenient for a package with many files. GitHub Desktop is the recommended manual alternative: add the extracted folder as a local repository, publish it, create the branch, and push.

## Files that should remain private or untracked

Do not add the local `data/`, `results/`, `models/`, `checkpoints/`, `tmp/`, or manuscript submission directories. The included `.gitignore` blocks these paths and common large binary formats.
