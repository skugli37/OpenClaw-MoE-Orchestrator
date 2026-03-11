# Security Policy

## Supported Scope

The supported production surface is:

- `src/openclaw_moe_orchestrator/`
- `.github/workflows/`
- `configs/`
- production wrappers in `scripts/`

Code under `experiments/` is not treated as production runtime.

## Reporting

If you discover a security issue, report it privately through GitHub Security Advisories for this repository or contact the maintainer account before opening a public issue.

## Secrets Handling

- Do not commit `.env` files or credentials.
- Production workflows rely on GitHub-provided tokens or environment-level secrets.
- The `Security` workflow scans tracked files for likely secrets on pushes, pull requests, and a weekly schedule.

## Dependency Policy

- Production dependencies are pinned in `pyproject.toml`.
- CI runs lint and tests.
- The `Security` workflow performs dependency auditing against the production dependency manifest.
