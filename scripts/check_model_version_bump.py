"""
CI gate (warning-only): flags when batlab/models/gbrt.py changed in the
last commit without src/bundle_cache.py's MODEL_VERSION also changing.

Why this matters: src/bundle_cache.py's disk cache stores trained model
bundles keyed by MODEL_VERSION (among other things). If the model-training
code changes but MODEL_VERSION isn't bumped, a stale cached bundle can
silently keep serving predictions from the old model code after deploy --
nothing else re-validates that the cached output shape still matches what
the new code would produce.

FEATURE_VERSION used to have this same drift risk, duplicated between
src/bundle_cache.py and batlab/validation/manifest.py, kept in sync only by
this same kind of CI warning. It now lives once in
batlab/features/engineering.py and both modules import it directly, so it's
structurally impossible for it to drift -- nothing to check for that half
anymore. MODEL_VERSION genuinely can't be centralized the same way (nothing
else depends on gbrt.py's output shape), so it still needs a human to
remember to bump it, and this script still only warns (exit 1, but ci.yml
runs it with continue-on-error: true -- visible in the CI log, not blocking).

Extracted from ci.yml's inline `python -c "..."` block so the detection
logic (check_needs_bump()) is unit-testable -- see
tests/test_check_model_version_bump.py, which exists because this exact
class of "CI check silently never actually running/catching anything" bug
has hit this project before (pytest itself, historically).

Run locally:  python scripts/check_model_version_bump.py
"""

import subprocess
import sys

MODEL_FILE = "batlab/models/gbrt.py"
BUNDLE_CACHE_FILE = "src/bundle_cache.py"


def get_changed_files(base: str = "HEAD~1", head: str = "HEAD") -> list[str]:
    """Files changed between base and head. Empty list if the diff can't be
    computed (e.g. no prior commit, shallow clone)."""
    result = subprocess.run(
        ["git", "diff", "--name-only", base, head],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return []
    return [line for line in result.stdout.strip().splitlines() if line]


def check_needs_bump(changed_files: list[str]) -> bool:
    """True if MODEL_FILE changed without BUNDLE_CACHE_FILE also changing --
    i.e. the warning should fire."""
    model_changed  = MODEL_FILE in changed_files
    bundle_changed = BUNDLE_CACHE_FILE in changed_files
    return model_changed and not bundle_changed


def main() -> int:
    changed = get_changed_files()
    if not changed:
        print("MODEL_VERSION check skipped (no prior commit or diff unavailable).")
        return 0

    if check_needs_bump(changed):
        print(f"WARNING: {MODEL_FILE} changed but {BUNDLE_CACHE_FILE}")
        print("(app disk-cache MODEL_VERSION) was not updated. Stale cached")
        print("bundles may silently serve predictions from the old model code.")
        print(f"Bump MODEL_VERSION in {BUNDLE_CACHE_FILE}.")
        return 1

    print("MODEL_VERSION check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
