# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial implementation of the Retina Task environment
- Support for three evaluation modes: single_pattern, batch, and full_evaluation
- Two reward types: paper (fitness function from original paper) and simple (negative error)
- Pattern validation utilities via RetinaPatterns class
- Comprehensive test suite with 14 tests
- Example scripts:
  - Random agent example
  - Perfect agent example (baseline)
  - Pattern analysis script
- Full Sphinx documentation with Furo theme
- GitHub Actions workflows for CI/CD:
  - Pre-commit checks on PRs
  - Build and publish to PyPI on releases
  - Documentation versioning on tags
- Pre-commit hooks configuration
- Farama Foundation standard files:
  - Code of Conduct
  - Contributing guidelines
  - Pull request template
  - Funding configuration

## [0.1.0] - 2025-12-01

### Added
- Initial release of gymnasium-retinatask
- Clean, ML-agnostic implementation of the Left & Right Retina Problem
- Gymnasium-compatible API following Farama Foundation standards
- Support for Python 3.10-3.13
- MIT License
