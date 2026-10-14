"""
ParkiSense — Dataset Downloader
=================================
Downloads and organises publicly available walking / gait datasets suitable
for Parkinson's Disease research.

Datasets attempted
------------------

1. **PhysioNet — Gait in Parkinson's Disease (GaitPDB)**
   • URL  : https://physionet.org/static/published-projects/gaitpdb/
   • Type : Raw sensor time-series (.txt) — stride intervals, force plates
   • Use  : Provides validated Parkinson / control labels; not video-based
   • License: ODC-BY 1.0

2. **Zenodo — Walking Gait Video Dataset** (if publicly available)
   • Type : Video (.mp4)
   • Destination: data/raw/videos/{control,parkinson}/

3. **PhysioNet — Gait Analysis — Parkinson's Disease**
   • URL  : https://physionet.org/content/gait-maturation-db/1.0.0/
   • Type : Text signal files

NOTE
----
Most academic Parkinson walking video datasets require institutional access
or a data-use agreement (e.g., CARE-PD, PD-WALK).  The downloader will
fetch any dataset that is fully open.  If a dataset requires login, the
script will print the access URL so you can download it manually and place
it in the correct directory.

Usage (CLI):
    python scripts/download_dataset.py

    # Custom destination:
    python scripts/download_dataset.py --data-dir /path/to/data/raw

Usage (Python API):
    from scripts.download_dataset import DatasetDownloader
    downloader = DatasetDownloader()
    downloader.download_all()
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
import tarfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project root (scripts/ lives one level below project root)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils import get_logger, ensure_dir

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Default output directory
# ---------------------------------------------------------------------------
DEFAULT_DATA_DIR = str(_PROJECT_ROOT / "data" / "raw")

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
# Each entry: (name, url, label_subfolder, description, requires_agreement)
_DATASETS: List[Dict] = [
    {
        "name": "GaitPDB (PhysioNet)",
        "url": "https://physionet.org/static/published-projects/gaitpdb/gait-in-parkinsons-disease-1.0.0.zip",
        "dest_subdir": "gaitpdb",
        "description": "Gait sensor data — stride intervals, force data for Parkinson and control subjects.",
        "requires_agreement": False,
        "is_video": False,
    },
    {
        "name": "Gait Maturation DB (PhysioNet)",
        "url": "https://physionet.org/static/published-projects/gait-maturation-db/gait-maturation-database-1.0.0.zip",
        "dest_subdir": "gait_maturation",
        "description": "Gait signal database; useful for reference signal processing patterns.",
        "requires_agreement": False,
        "is_video": False,
    },
]

# Datasets that need manual download (restricted access)
_MANUAL_DATASETS: List[Dict] = [
    {
        "name": "CARE-PD Dataset",
        "url": "https://zenodo.org/record/care-pd (requires registration)",
        "instructions": (
            "1. Register at https://zenodo.org\n"
            "2. Search for 'CARE-PD walking video'\n"
            "3. Download and place .mp4 files in:\n"
            "     data/raw/videos/parkinson/  (Parkinson subjects)\n"
            "     data/raw/videos/control/    (Healthy controls)"
        ),
    },
    {
        "name": "PD-WALK Dataset",
        "url": "https://ieee-dataport.org/open-access/pd-walk",
        "instructions": (
            "1. Visit https://ieee-dataport.org/open-access/pd-walk\n"
            "2. Download and extract the dataset\n"
            "3. Place videos in data/raw/videos/{control,parkinson}/"
        ),
    },
    {
        "name": "Daphnet Gait Dataset (UCI)",
        "url": "https://archive.ics.uci.edu/ml/datasets/Daphnet+Freezing+of+Gait",
        "instructions": (
            "1. Visit the UCI link above\n"
            "2. Download 'dataset_fog_release.tar.gz'\n"
            "3. Contains accelerometer signals (not video) — useful for sensor-based baselines"
        ),
    },
]


# ---------------------------------------------------------------------------
# DatasetDownloader
# ---------------------------------------------------------------------------

class DatasetDownloader:
    """
    Downloads, extracts, and organises gait datasets.

    Args:
        data_dir: Root destination directory (default: data/raw/).
        chunk_size: HTTP streaming chunk size in bytes.
        timeout:    Request timeout in seconds.
    """

    def __init__(
        self,
        data_dir:   str = DEFAULT_DATA_DIR,
        chunk_size: int = 8192,
        timeout:    int = 30,
    ) -> None:
        self.data_dir   = Path(data_dir)
        self.chunk_size = chunk_size
        self.timeout    = timeout

        # Ensure the standard video tree exists
        ensure_dir(self.data_dir / "videos" / "control")
        ensure_dir(self.data_dir / "videos" / "parkinson")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download_all(self) -> None:
        """Download every dataset in the registry."""
        logger.info("═" * 60)
        logger.info("  ParkiSense Dataset Downloader")
        logger.info("═" * 60)

        succeeded = []
        failed    = []

        for ds in _DATASETS:
            logger.info("\n▶ %s", ds["name"])
            logger.info("  %s", ds["description"])
            try:
                dest = self.data_dir / ds["dest_subdir"]
                self._download_and_extract(ds["url"], dest, ds["name"])
                succeeded.append(ds["name"])
            except Exception as exc:
                logger.warning("  ❌ Failed: %s", exc)
                failed.append((ds["name"], str(exc)))

        self._print_manual_instructions()
        self._print_summary(succeeded, failed)

    def download_one(self, name: str) -> None:
        """Download a single dataset by name."""
        for ds in _DATASETS:
            if ds["name"].lower() == name.lower():
                dest = self.data_dir / ds["dest_subdir"]
                self._download_and_extract(ds["url"], dest, ds["name"])
                return
        raise ValueError(f"Unknown dataset: '{name}'. "
                         f"Available: {[d['name'] for d in _DATASETS]}")

    def list_datasets(self) -> None:
        """Print all available datasets."""
        print("\n── Automatically downloadable ─────────────────────────")
        for i, ds in enumerate(_DATASETS, 1):
            print(f"  {i}. {ds['name']}")
            print(f"     {ds['description']}")
            print(f"     Video: {ds['is_video']}  |  URL: {ds['url'][:70]}…")
        print("\n── Manual download required ────────────────────────────")
        for i, ds in enumerate(_MANUAL_DATASETS, 1):
            print(f"  {i}. {ds['name']}  →  {ds['url']}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _download_and_extract(self, url: str, dest_dir: Path, name: str) -> None:
        """Stream-download *url* and extract the archive to *dest_dir*."""
        ensure_dir(dest_dir)

        filename = Path(urlparse(url).path).name
        archive_path = dest_dir / filename

        # ── Download ──────────────────────────────────────────────────
        logger.info("  Downloading %s …", url)
        try:
            response = requests.get(
                url,
                stream=True,
                timeout=self.timeout,
                headers={"User-Agent": "ParkiSense-Downloader/1.0"},
            )
            response.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Cannot reach {url} — check your internet connection."
            )
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response else "?"
            raise RuntimeError(
                f"HTTP {status} downloading {name}. "
                "Dataset may require registration — see manual instructions."
            )

        total_bytes = int(response.headers.get("content-length", 0))
        with open(archive_path, "wb") as f:
            with tqdm(
                total=total_bytes,
                unit="B",
                unit_scale=True,
                desc=f"  Downloading {filename}",
                ncols=80,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=self.chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        logger.info("  ✅ Downloaded → %s", archive_path)

        # ── Extract ───────────────────────────────────────────────────
        self._extract_archive(archive_path, dest_dir)
        os.remove(archive_path)  # Remove archive after extraction
        logger.info("  ✅ Extracted → %s", dest_dir)

    @staticmethod
    def _extract_archive(archive_path: Path, dest_dir: Path) -> None:
        """Extract a .zip or .tar.gz archive."""
        suffix = "".join(archive_path.suffixes)

        if suffix in (".zip",):
            with zipfile.ZipFile(archive_path, "r") as zf:
                members = zf.namelist()
                for member in tqdm(members, desc="  Extracting", unit="file", ncols=80):
                    zf.extract(member, dest_dir)

        elif suffix in (".tar.gz", ".tgz", ".tar.bz2", ".tar"):
            mode = "r:gz" if "gz" in suffix else ("r:bz2" if "bz2" in suffix else "r")
            with tarfile.open(archive_path, mode) as tf:
                members = tf.getmembers()
                for member in tqdm(members, desc="  Extracting", unit="file", ncols=80):
                    tf.extract(member, dest_dir)
        else:
            logger.warning("Unknown archive format: %s — skipping extraction.", suffix)

    def _print_manual_instructions(self) -> None:
        """Print manual download instructions for restricted datasets."""
        logger.info("\n" + "─" * 60)
        logger.info("  MANUAL DOWNLOAD REQUIRED (restricted datasets)")
        logger.info("─" * 60)
        for ds in _MANUAL_DATASETS:
            logger.info("\n  📦 %s", ds["name"])
            logger.info("     URL: %s", ds["url"])
            for line in ds["instructions"].strip().split("\n"):
                logger.info("     %s", line)

    @staticmethod
    def _print_summary(succeeded: List[str], failed: List[Tuple]) -> None:
        logger.info("\n" + "═" * 60)
        logger.info("  Download Summary")
        logger.info("═" * 60)
        logger.info("  ✅ Succeeded : %d", len(succeeded))
        for name in succeeded:
            logger.info("     • %s", name)
        if failed:
            logger.info("  ❌ Failed    : %d", len(failed))
            for name, reason in failed:
                logger.info("     • %s — %s", name, reason)
        logger.info("═" * 60)
        logger.info(
            "\n  Next step: Place your walking videos in\n"
            "    data/raw/videos/control/   — healthy subjects\n"
            "    data/raw/videos/parkinson/ — Parkinson subjects\n\n"
            "  Then run:\n"
            "    python src/dataset/create_dataset.py\n"
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="download_dataset",
        description="ParkiSense — Gait Dataset Downloader",
    )
    p.add_argument("--data-dir", "-d", default=DEFAULT_DATA_DIR,
                   help="Root download directory (default: data/raw)")
    p.add_argument("--list", "-l", action="store_true",
                   help="List available datasets and exit")
    p.add_argument("--dataset", "-n", default=None,
                   help="Download a specific dataset by name")
    return p


def main() -> None:
    args       = _build_arg_parser().parse_args()
    downloader = DatasetDownloader(data_dir=args.data_dir)

    if args.list:
        downloader.list_datasets()
        return

    if args.dataset:
        downloader.download_one(args.dataset)
    else:
        downloader.download_all()


if __name__ == "__main__":
    main()
