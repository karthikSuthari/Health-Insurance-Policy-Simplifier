"""
Phase 1: Google Drive PDF Downloader
=====================================
Authenticates with Google Drive via service account,
lists all PDFs in a target folder, and downloads them locally.
Skips files that already exist on disk (by name + size match).

Requirements:
    pip install google-api-python-client google-auth google-auth-httplib2

Usage:
    1. Place your service account credentials.json in the project root.
    2. Set FOLDER_ID below (or pass via env var DRIVE_FOLDER_ID).
    3. Run:  python drive_downloader.py
"""

import io
import os
import re
import sys
import json
import logging
from pathlib import Path
from datetime import datetime, timezone

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.errors import HttpError

# ─── Configuration ────────────────────────────────────────────────────────────

FOLDER_ID = os.getenv("DRIVE_FOLDER_ID", "YOUR_FOLDER_ID_HERE")
CREDENTIALS_FILE = os.getenv("GOOGLE_CREDENTIALS", "credentials.json")
DOWNLOAD_DIR = Path("./data/policies")
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
CHUNK_SIZE = 50 * 1024 * 1024  # 50 MB per chunk (good for large PDFs)

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("drive_downloader")

MAX_FILENAME_LENGTH = 120  # keep well under Windows 260-char path limit

# ─── Filename Helpers ─────────────────────────────────────────────────────────


def normalize_filename(raw_name: str) -> str:
    """
    Normalise a Drive filename for safe local storage.

    1. Ensure the file ends with .pdf (Drive says mimeType is PDF).
    2. Replace characters illegal on Windows: \ / : * ? " < > |
    3. Collapse multiple spaces / dots.
    4. Truncate to MAX_FILENAME_LENGTH (preserving .pdf).
    """
    name = raw_name.strip()

    # 1. Guarantee .pdf extension
    #    .cdr files that Drive reports as application/pdf are really PDFs
    stem = name
    # Strip known misleading extensions
    for ext in (".cdr", ".pdf"):
        if stem.lower().endswith(ext):
            stem = stem[: -len(ext)]
            break
    name = stem + ".pdf"

    # 2. Remove / replace illegal path characters
    name = re.sub(r'[\\/:*?"<>|]', '_', name)

    # 3. Collapse whitespace & dots
    name = re.sub(r'\s+', ' ', name).strip()
    name = re.sub(r'\.{2,}', '.', name)

    # 4. Truncate if too long (keep .pdf)
    if len(name) > MAX_FILENAME_LENGTH:
        name = name[: MAX_FILENAME_LENGTH - 4].rstrip('. ') + ".pdf"

    return name


def deduplicate_filename(name: str, seen: dict[str, int]) -> str:
    """
    If *name* was already used, append _1, _2, … to make it unique.
    Mutates *seen* in-place to track counts.

    Args:
        name: normalised filename.
        seen: dict mapping lowercase name → count of times seen so far.

    Returns:
        A unique filename (with suffix if necessary).
    """
    key = name.lower()
    if key in seen:
        seen[key] += 1
        stem, ext = os.path.splitext(name)
        unique = f"{stem}_{seen[key]}{ext}"
        log.info("Dedup: '%s' → '%s'", name, unique)
        return unique
    else:
        seen[key] = 0  # first occurrence, no suffix needed
        return name


# ─── Core Functions ───────────────────────────────────────────────────────────


def authenticate(credentials_path: str = CREDENTIALS_FILE):
    """
    Authenticate with Google Drive using a service account JSON key file.

    Returns:
        googleapiclient.discovery.Resource – authorized Drive v3 service object.

    Raises:
        FileNotFoundError: if credentials file is missing.
        ValueError: if the credentials file is malformed.
    """
    creds_path = Path(credentials_path)
    if not creds_path.exists():
        raise FileNotFoundError(
            f"Service account credentials not found at '{creds_path.resolve()}'. "
            "Download the JSON key from Google Cloud Console and place it there."
        )

    log.info("Authenticating with service account: %s", creds_path.name)
    try:
        credentials = service_account.Credentials.from_service_account_file(
            str(creds_path), scopes=SCOPES
        )
        service = build("drive", "v3", credentials=credentials, cache_discovery=False)
        log.info("Authentication successful.")
        return service
    except Exception as e:
        raise ValueError(f"Failed to authenticate: {e}") from e


def list_pdfs(service, folder_id: str = FOLDER_ID) -> list[dict]:
    """
    List every PDF file inside the given Drive folder (non-recursive).

    Returns:
        List of dicts with keys: id, name, size, modifiedTime
    """
    log.info("Listing PDFs in folder: %s", folder_id)

    query = (
        f"'{folder_id}' in parents "
        "and mimeType='application/pdf' "
        "and trashed=false"
    )
    fields = "nextPageToken, files(id, name, size, modifiedTime)"

    all_files: list[dict] = []
    page_token = None

    while True:
        try:
            response = (
                service.files()
                .list(
                    q=query,
                    fields=fields,
                    pageSize=100,
                    pageToken=page_token,
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                )
                .execute()
            )
        except HttpError as e:
            log.error("Drive API error while listing files: %s", e)
            raise

        files = response.get("files", [])
        all_files.extend(files)
        log.info("  … fetched %d files (total so far: %d)", len(files), len(all_files))

        page_token = response.get("nextPageToken")
        if not page_token:
            break

    log.info("Found %d PDF(s) in folder.", len(all_files))
    return all_files


def _should_skip(file_meta: dict, dest: Path) -> bool:
    """Return True if the local file already exists and matches the remote size."""
    if not dest.exists():
        return False
    local_size = dest.stat().st_size
    remote_size = int(file_meta.get("size", -1))
    if local_size == remote_size:
        return True
    return False


def download_file(
    service,
    file_meta: dict,
    dest_dir: Path = DOWNLOAD_DIR,
    local_name: str | None = None,
) -> dict:
    """
    Download a single file from Drive to *dest_dir*.

    Args:
        local_name: If provided, use this as the on-disk filename instead of
                    the raw Drive name. Callers should pass the normalised &
                    deduplicated name here.

    Returns:
        Result dict: {file, drive_name, status, path, size_bytes, error}
    """
    file_id = file_meta["id"]
    drive_name = file_meta["name"]
    file_name = local_name or drive_name
    dest_path = dest_dir / file_name

    # --- skip check -----------------------------------------------------------
    if _should_skip(file_meta, dest_path):
        log.info("SKIP (already exists, size matches): %s", file_name)
        return {
            "file": file_name,
            "drive_name": drive_name,
            "status": "skipped",
            "path": str(dest_path),
            "size_bytes": dest_path.stat().st_size,
            "error": None,
        }

    # --- download -------------------------------------------------------------
    log.info("Downloading: %s  (%s bytes)", file_name, file_meta.get("size", "?"))
    if drive_name != file_name:
        log.info("  (Drive name: %s)", drive_name)
    try:
        request = service.files().get_media(fileId=file_id)
        buffer = io.BytesIO()
        downloader = MediaIoBaseDownload(buffer, request, chunksize=CHUNK_SIZE)

        done = False
        while not done:
            status, done = downloader.next_chunk()
            if status:
                pct = int(status.progress() * 100)
                log.info("  %s — %d%% complete", file_name, pct)

        # Write to disk
        dest_path.write_bytes(buffer.getvalue())
        final_size = dest_path.stat().st_size
        log.info("Saved: %s (%d bytes)", dest_path, final_size)

        return {
            "file": file_name,
            "drive_name": drive_name,
            "status": "downloaded",
            "path": str(dest_path),
            "size_bytes": final_size,
            "error": None,
        }

    except HttpError as e:
        log.error("Drive API error downloading %s: %s", file_name, e)
        return {
            "file": file_name,
            "drive_name": drive_name,
            "status": "error",
            "path": None,
            "size_bytes": 0,
            "error": str(e),
        }
    except Exception as e:
        log.error("Unexpected error downloading %s: %s", file_name, e)
        return {
            "file": file_name,
            "drive_name": drive_name,
            "status": "error",
            "path": None,
            "size_bytes": 0,
            "error": str(e),
        }


def download_all_pdfs(
    folder_id: str = FOLDER_ID,
    credentials_path: str = CREDENTIALS_FILE,
    dest_dir: Path = DOWNLOAD_DIR,
) -> dict:
    """
    End-to-end: authenticate → list → download all PDFs.

    Returns:
        Summary dict with totals and per-file results.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    log.info("Download directory: %s", dest_dir.resolve())

    # 1. Auth
    service = authenticate(credentials_path)

    # 2. List
    pdf_files = list_pdfs(service, folder_id)
    if not pdf_files:
        log.warning("No PDFs found — nothing to download.")
        return {
            "status": "success",
            "folder_id": folder_id,
            "total_files": 0,
            "downloaded": 0,
            "skipped": 0,
            "errors": 0,
            "results": [],
        }

    # 3. Normalise & deduplicate filenames
    seen_names: dict[str, int] = {}
    file_plan: list[tuple[dict, str]] = []  # (meta, local_name)
    for f in pdf_files:
        norm = normalize_filename(f["name"])
        local = deduplicate_filename(norm, seen_names)
        file_plan.append((f, local))

    log.info("Filename plan ready (%d unique names).", len(file_plan))

    # 4. Download each
    results: list[dict] = []
    for idx, (f, local_name) in enumerate(file_plan, 1):
        log.info("--- [%d/%d] %s ---", idx, len(file_plan), local_name)
        result = download_file(service, f, dest_dir, local_name=local_name)
        results.append(result)

    # 5. Summarise
    downloaded = sum(1 for r in results if r["status"] == "downloaded")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    errors = sum(1 for r in results if r["status"] == "error")
    total_bytes = sum(r["size_bytes"] for r in results)

    summary = {
        "status": "success" if errors == 0 else "partial",
        "folder_id": folder_id,
        "total_files": len(pdf_files),
        "downloaded": downloaded,
        "skipped": skipped,
        "errors": errors,
        "total_bytes": total_bytes,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }

    log.info("=" * 60)
    log.info(
        "DONE  |  Downloaded: %d  |  Skipped: %d  |  Errors: %d  |  Total: %.2f MB",
        downloaded,
        skipped,
        errors,
        total_bytes / (1024 * 1024),
    )
    log.info("=" * 60)

    # Persist manifest for downstream phases
    manifest_path = dest_dir / "_download_manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2, default=str))
    log.info("Manifest saved to %s", manifest_path)

    return summary


# ─── Quick Test (offline, no credentials needed) ─────────────────────────────


def _quick_self_test():
    """Smoke-test helper functions without hitting the API."""
    import tempfile  # noqa: E401

    log.info("Running offline self-test …")

    # ── Test _should_skip logic ───────────────────────────────────────────
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(b"fake-pdf-content")
        tmp_path = Path(tmp.name)

    meta_match = {"id": "x", "name": tmp_path.name, "size": str(tmp_path.stat().st_size)}
    meta_mismatch = {"id": "x", "name": tmp_path.name, "size": "999999"}

    assert _should_skip(meta_match, tmp_path) is True, "Should skip when sizes match"
    assert _should_skip(meta_mismatch, tmp_path) is False, "Should NOT skip when sizes differ"
    assert _should_skip(meta_match, Path("nonexistent.pdf")) is False, "Should NOT skip missing file"
    tmp_path.unlink()
    log.info("  _should_skip … OK")

    # ── Test normalize_filename ───────────────────────────────────────────
    assert normalize_filename("Policy.cdr") == "Policy.pdf"
    assert normalize_filename("MyFile.pdf") == "MyFile.pdf"
    assert normalize_filename("No Extension") == "No Extension.pdf"
    assert normalize_filename("Has:Colons/Slashes*Stars") == "Has_Colons_Slashes_Stars.pdf"
    assert normalize_filename("a" * 200 + ".cdr").endswith(".pdf")
    assert len(normalize_filename("a" * 200 + ".cdr")) <= MAX_FILENAME_LENGTH
    long_name = "Some of the definitions are not yet incorporated in the documents- i.e. Day Care, Dependent Child, ICU, Inpatient Care, Medically necessary. You may also define Family in prospectus"
    norm = normalize_filename(long_name)
    assert norm.endswith(".pdf") and len(norm) <= MAX_FILENAME_LENGTH
    log.info("  normalize_filename … OK")

    # ── Test deduplicate_filename ─────────────────────────────────────────
    seen: dict[str, int] = {}
    assert deduplicate_filename("Policy.pdf", seen) == "Policy.pdf"
    assert deduplicate_filename("Policy.pdf", seen) == "Policy_1.pdf"
    assert deduplicate_filename("Policy.pdf", seen) == "Policy_2.pdf"
    assert deduplicate_filename("Other.pdf", seen) == "Other.pdf"
    # Case-insensitive dedup
    assert deduplicate_filename("policy.pdf", seen) == "policy_3.pdf"
    log.info("  deduplicate_filename … OK")

    log.info("Self-test PASSED.")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--test" in sys.argv:
        _quick_self_test()
        sys.exit(0)

    if FOLDER_ID == "YOUR_FOLDER_ID_HERE":
        log.error(
            "Set DRIVE_FOLDER_ID env var or edit FOLDER_ID in this script before running."
        )
        sys.exit(1)

    summary = download_all_pdfs()
    print(json.dumps(summary, indent=2, default=str))
