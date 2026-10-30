"""One-off job: load the documents, split them, embed them, and save the index.

    python scripts/ingest.py              # reads data/raw, writes data/processed
    python scripts/ingest.py --check      # only checks the saved index is up to date (exit code 1 if not)

Run it again whenever a document in data/raw changes or EMBED_MODEL changes. The app uses the saved index when it
matches the documents; if it does not, the app embeds them itself at start-up (slower, but never wrong).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from voice_agent.config import get_settings  # noqa: E402
from voice_agent.rag import load_documents, split_into_chunks  # noqa: E402
from voice_agent.rag.embedder import build_embedder  # noqa: E402
from voice_agent.rag.store import build_store, chunks_fingerprint, read_meta, save_index  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--docs", type=Path, help="documents folder (default: DOCS_DIR, data/raw)")
    parser.add_argument("--out", type=Path, help="where to save the index (default: PROCESSED_DIR, data/processed)")
    parser.add_argument("--check", action="store_true", help="do not write anything; exit 1 if the index is out of date")
    args = parser.parse_args()

    settings = get_settings()
    docs_dir, out_dir = args.docs or settings.docs_dir, args.out or settings.processed_dir
    documents = load_documents(docs_dir)
    chunks = split_into_chunks(documents)
    print(f"Read {len(documents)} document(s) from {docs_dir}: {len(chunks)} chunks")
    for chunk in chunks:
        print(f"  - {chunk.metadata['source']}: {chunk.metadata['title']}")

    if args.check:
        try:
            meta = read_meta(out_dir)
        except FileNotFoundError:
            print("No saved index. Run: python scripts/ingest.py")
            return 1
        current = meta.get("fingerprint") == chunks_fingerprint(chunks) and meta.get("embed_model") == settings.embed_model
        print("The saved index is up to date." if current else "The saved index is OUT OF DATE. Run: python scripts/ingest.py")
        return 0 if current else 1

    print(f"Embedding with {settings.embed_model} ...")
    start = time.perf_counter()
    store = build_store(chunks, build_embedder(settings.embed_model))
    meta = save_index(store, out_dir, settings.embed_model, chunks)
    print(f"Saved {meta['chunks']} vectors to {out_dir} in {time.perf_counter() - start:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
