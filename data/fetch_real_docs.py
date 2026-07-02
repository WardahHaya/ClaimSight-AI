import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.ingest import fetch_real_documents


if __name__ == "__main__":
    downloaded = fetch_real_documents(force=False)
    print(f"Downloaded or reused {len(downloaded)} real source documents in data/raw.")
