"""
Download the UI segmentation dataset from Roboflow.
Run once to populate the ./dataset/ folder.

Requirements:
    pip install roboflow python-dotenv
"""

import os
import sys
from pathlib import Path

# Check dependencies
missing = []
try:
    from roboflow import Roboflow
except ImportError:
    missing.append("roboflow")
try:
    from dotenv import load_dotenv
except ImportError:
    missing.append("python-dotenv")

if missing:
    print("Missing dependencies. Install with:")
    print(f"  pip install {' '.join(missing)}")
    sys.exit(1)

load_dotenv()

api_key = os.getenv("ROBOFLOW_API_KEY")
if not api_key:
    print("ERROR: ROBOFLOW_API_KEY not found in .env")
    sys.exit(1)

WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE", "cvtesting-3rnup")
PROJECT = os.getenv("ROBOFLOW_PROJECT", "segmentation-ui-oh51c")
VERSION = int(os.getenv("ROBOFLOW_VERSION", "1"))
FORMAT = os.getenv("ROBOFLOW_FORMAT", "yolov8")
DEST = os.getenv("ROBOFLOW_DEST", "./dataset")
AUTO_GENERATE_VERSION = os.getenv("ROBOFLOW_AUTO_GENERATE_VERSION", "1").lower() not in {
    "0",
    "false",
    "no",
}
OVERWRITE_EXISTING = os.getenv("ROBOFLOW_OVERWRITE", "1").lower() not in {
    "0",
    "false",
    "no",
}

print(f"Connecting to Roboflow workspace: {WORKSPACE}")
rf = Roboflow(api_key=api_key)

project = rf.workspace(WORKSPACE).project(PROJECT)

def _extract_version_numbers(version_objects):
    version_numbers = []
    for item in version_objects or []:
        value = getattr(item, "version", None)
        if value is None and isinstance(item, dict):
            value = item.get("version")
        try:
            version_numbers.append(int(value))
        except (TypeError, ValueError):
            continue
    return sorted(set(version_numbers))


versions = project.versions()
available_versions = _extract_version_numbers(versions)

if available_versions:
    joined = ", ".join(str(v) for v in available_versions)
    print(f"Available versions: {joined}")
else:
    if AUTO_GENERATE_VERSION:
        print("No dataset versions found. Generating initial version automatically...")
        try:
            generated = int(
                project.generate_version(
                    settings={
                        "augmentation": {},
                        "preprocessing": {},
                    }
                )
            )
            available_versions = [generated]
            print(f"Generated version {generated}.")
        except Exception as e:
            print(f"ERROR: Failed to auto-generate initial version: {e}")
            print("Try creating a version manually in the Roboflow UI.")
            sys.exit(1)
    else:
        print("ERROR: No dataset versions were returned for this project.")
        print("Possible causes:")
        print("  - Wrong workspace or project slug")
        print("  - API key does not have access to this project")
        print("  - No version has been generated/exported in Roboflow yet")
        print("\nVerify in Roboflow UI that at least one dataset version exists,")
        print(f"and confirm slugs: workspace='{WORKSPACE}', project='{PROJECT}'.")
        print("If desired, enable auto-generation with ROBOFLOW_AUTO_GENERATE_VERSION=1.")
        sys.exit(1)

target_version = VERSION
if target_version not in available_versions:
    latest = available_versions[-1]
    print(
        f"Requested version {target_version} is unavailable. "
        f"Falling back to latest available version {latest}."
    )
    target_version = latest

print(f"Downloading version {target_version} in {FORMAT} format → {DEST}")
try:
    dataset = project.version(target_version).download(
        FORMAT, location=DEST, overwrite=OVERWRITE_EXISTING
    )
except Exception as e:
    print(f"ERROR: Download failed for version {target_version}: {e}")
    if target_version != available_versions[-1]:
        latest = available_versions[-1]
        print(f"Retrying with latest version {latest}...")
        dataset = project.version(latest).download(
            FORMAT, location=DEST, overwrite=OVERWRITE_EXISTING
        )
    else:
        raise

print("\nDownload complete.")
print(f"Dataset location: {dataset.location}")
print(f"Overwrite enabled: {OVERWRITE_EXISTING}")

# Post-download verification summary (helps catch "folder exists but looks empty" confusion)
root = Path(dataset.location)
if root.exists():
    train_imgs = list((root / "train" / "images").glob("*"))
    train_lbls = list((root / "train" / "labels").glob("*.txt"))
    valid_exists = (root / "valid").exists()
    test_exists = (root / "test").exists()
    print(
        "Verification summary:"
        f" train/images={len(train_imgs)},"
        f" train/labels={len(train_lbls)},"
        f" valid_split={'yes' if valid_exists else 'no'},"
        f" test_split={'yes' if test_exists else 'no'}"
    )
