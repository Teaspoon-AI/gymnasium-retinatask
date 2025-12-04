  1. Push to GitHub:
  git remote add origin https://github.com/Teaspoon-AI/gymnasium-retinatask.git
  git push -u origin main
  2. Enable GitHub Actions - Workflows will automatically run on PRs and releases
  3. Create First Release:
  git tag v0.1.0
  git push origin v0.1.0
  3. This will trigger:
    - Documentation build and deployment
    - PyPI package build (publish requires PYPI_API_TOKEN secret)
