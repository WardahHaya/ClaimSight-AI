import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import httpx
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import DATA_DIR, RAW_DATA_DIR
from backend.vectorstore import get_vectorstore

MANIFEST_PATH = os.path.join(DATA_DIR, "sources.json")


def load_source_manifest() -> List[Dict[str, Any]]:
    with open(MANIFEST_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_filename(title: str, source_type: str, url: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    extension = ".pdf" if source_type == "pdf" else ".html"
    if url.lower().endswith(".pdf"):
        extension = ".pdf"
    return f"{slug}{extension}"


def fetch_real_documents(force: bool = False) -> List[Path]:
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    manifest = load_source_manifest()
    downloaded_paths: List[Path] = []
    failed_sources: List[str] = []

    with httpx.Client(
        timeout=60.0,
        follow_redirects=True,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36"
            ),
            "Accept": "*/*",
        },
    ) as client:
        for item in manifest:
            target_path = Path(RAW_DATA_DIR) / _safe_filename(item["title"], item["type"], item["url"])
            if target_path.exists() and not force:
                downloaded_paths.append(target_path)
                continue

            try:
                response = client.get(item["url"])
                response.raise_for_status()
                target_path.write_bytes(response.content)
                downloaded_paths.append(target_path)
            except Exception as exc:
                if target_path.exists():
                    target_path.unlink(missing_ok=True)
                curl_result = subprocess.run(
                    [
                        "curl.exe",
                        "--fail",
                        "--silent",
                        "--show-error",
                        "--location",
                        "--user-agent",
                        "Mozilla/5.0",
                        "--output",
                        str(target_path),
                        item["url"],
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if curl_result.returncode == 0 and target_path.exists() and target_path.stat().st_size > 0:
                    downloaded_paths.append(target_path)
                else:
                    if target_path.exists():
                        target_path.unlink(missing_ok=True)
                    failure_detail = curl_result.stderr.strip() or str(exc)
                    failed_sources.append(f"{item['title']}: {failure_detail}")

    if failed_sources:
        print("Warning: some source documents could not be downloaded:")
        for failure in failed_sources:
            print(f"- {failure}")

    if not downloaded_paths:
        raise RuntimeError("No source documents were downloaded successfully.")
    return downloaded_paths


def _clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_pdf_documents(path: Path, source: Dict[str, Any]) -> List[Document]:
    reader = PdfReader(str(path))
    documents: List[Document] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = _clean_text(page.extract_text() or "")
        if not text:
            continue
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": path.name,
                    "source_title": source["title"],
                    "source_url": source["url"],
                    "source_type": source["type"],
                    "page_number": page_number,
                },
            )
        )
    return documents


def _extract_html_documents(path: Path, source: Dict[str, Any]) -> List[Document]:
    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    main_content = soup.find("main") or soup.find("article") or soup.body or soup
    title = source["title"]
    page_text = _clean_text(main_content.get_text("\n"))
    combined = f"{title}\nSource URL: {source['url']}\n\n{page_text}"

    return [
        Document(
            page_content=combined,
            metadata={
                "source": path.name,
                "source_title": source["title"],
                "source_url": source["url"],
                "source_type": source["type"],
            },
        )
    ]


def load_documents_from_raw() -> List[Document]:
    manifest = load_source_manifest()
    manifest_by_filename = {
        _safe_filename(item["title"], item["type"], item["url"]): item for item in manifest
    }

    documents: List[Document] = []
    raw_dir = Path(RAW_DATA_DIR)
    for path in sorted(raw_dir.iterdir()):
        if not path.is_file():
            continue
        source = manifest_by_filename.get(path.name)
        if not source:
            continue
        if path.suffix.lower() == ".pdf":
            documents.extend(_extract_pdf_documents(path, source))
        elif path.suffix.lower() in {".html", ".htm"}:
            documents.extend(_extract_html_documents(path, source))
    return documents


def _chunk_documents(documents: Iterable[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunked = splitter.split_documents(list(documents))
    for index, chunk in enumerate(chunked):
        chunk.metadata["chunk_index"] = index
    return chunked


def ingest_documents(fetch: bool = False, force_fetch: bool = False) -> int:
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    if fetch or not any(Path(RAW_DATA_DIR).iterdir()):
        fetch_real_documents(force=force_fetch)

    documents = load_documents_from_raw()
    if not documents:
        raise RuntimeError(
            "No real source documents were found. Run `python data/fetch_real_docs.py` or `python backend/ingest.py --fetch`."
        )

    chunks = _chunk_documents(documents)
    vectorstore = get_vectorstore(create_if_missing=True)

    try:
        existing_ids = vectorstore._collection.get().get("ids", [])
        if existing_ids:
            vectorstore._collection.delete(ids=existing_ids)
    except Exception:
        pass

    ids: List[str] = []
    for chunk in chunks:
        fingerprint = hashlib.sha256(
            f"{chunk.metadata.get('source')}::{chunk.metadata.get('page_number', '')}::{chunk.metadata.get('chunk_index')}::{chunk.page_content[:200]}".encode(
                "utf-8"
            )
        ).hexdigest()
        ids.append(fingerprint)

    vectorstore.add_documents(chunks, ids=ids)
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and ingest real insurance documents into ChromaDB.")
    parser.add_argument("--fetch", action="store_true", help="Fetch the real source corpus before ingesting.")
    parser.add_argument("--force-fetch", action="store_true", help="Re-download the source corpus even if files exist.")
    args = parser.parse_args()

    total_chunks = ingest_documents(fetch=args.fetch, force_fetch=args.force_fetch)
    print(f"Ingestion complete. Indexed {total_chunks} chunks into the local Chroma collection.")


if __name__ == "__main__":
    main()
