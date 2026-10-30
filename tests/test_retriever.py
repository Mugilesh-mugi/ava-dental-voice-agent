"""The knowledge search (RAG): loading documents, chunking, the vector store, the similarity limit, reranking, and
the saved index made by scripts/ingest.py."""
import subprocess
import sys

import pytest
from langchain_core.documents import Document

from fakes import KeywordEmbedder
from voice_agent.config import Settings
from voice_agent.rag import load_chunks, load_documents, split_into_chunks
from voice_agent.rag.retriever import Retriever, build_retriever
from voice_agent.rag.store import build_store, chunks_fingerprint, load_index, read_meta, save_index


def chunk(title, text, source="faq.md"):
    """A chunk as the chunker makes it."""
    return Document(page_content=f"{title}: {text}", metadata={"title": title, "source": source})


def titles(result):
    return [doc.metadata["title"] for doc, _ in result.matches]


# ---------- Loading and chunking ----------
def test_the_clinic_documents_load_into_chunks():
    chunks = load_chunks(Settings(_env_file=None).docs_dir)
    found = [c.metadata["title"] for c in chunks]
    for topic in ["Opening hours", "Prices", "Insurance and payment", "Cancellations", "Location and parking",
                  "Children", "Emergency appointments"]:
        assert topic in found
    assert all(c.metadata["source"] == "faq.md" and c.page_content.startswith(c.metadata["title"] + ": ") for c in chunks)


def test_each_heading_starts_a_chunk(tmp_path):
    (tmp_path / "a.md").write_text("# Title\nintro that is ignored\n\n## First\nOne.\nTwo.\n\n## Second\nThree.\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("## Third\nFour.\n", encoding="utf-8")
    (tmp_path / "ignored.pdf").write_text("## Not read\nNo.", encoding="utf-8")
    chunks = load_chunks(tmp_path)
    assert [(c.page_content, c.metadata["title"], c.metadata["source"]) for c in chunks] == [
        ("First: One. Two.", "First", "a.md"), ("Second: Three.", "Second", "a.md"), ("Third: Four.", "Third", "b.txt")]


def test_a_single_file_can_be_loaded(tmp_path):
    path = tmp_path / "faq.md"
    path.write_text("## Parking\nFree.\n", encoding="utf-8")
    assert [d.metadata["source"] for d in load_documents(path)] == ["faq.md"]


def test_missing_or_empty_documents_are_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="No documents found"):
        load_documents(tmp_path / "nope")
    with pytest.raises(ValueError, match="No .md or .txt"):
        load_documents(tmp_path)
    with pytest.raises(ValueError, match="No '## ' sections"):
        split_into_chunks([Document(page_content="just text, no headings", metadata={"source": "notes.md"})])


# ---------- The similarity limit ----------
def test_threshold_separates_answerable_from_unanswerable():
    retriever = Retriever([chunk("Parking", "There is free parking behind the building.")], KeywordEmbedder(),
                          threshold=0.25, top_k=2)
    assert retriever.search("Is there parking?").answerable
    unanswerable = retriever.search("Do you sell train tickets?")
    assert not unanswerable.answerable and unanswerable.best_score < 0.25 and unanswerable.matches == []


def test_only_chunks_above_the_threshold_are_returned():
    chunks = [chunk("Parking", "Free parking behind the building."), chunk("Hours", "Open Monday to Friday."),
              chunk("Prices", "A check-up costs 45 pounds.")]
    result = Retriever(chunks, KeywordEmbedder(), threshold=0.25, top_k=3).search("is there free parking")
    assert titles(result) == ["Parking"] and result.matches[0][1] > 0.25


def test_an_optional_reranker_can_reorder_the_matches():
    chunks = [chunk("Parking", "Free parking behind the building."), chunk("Disabled parking", "Free parking for blue badge holders.")]
    calls = []

    def reverse(question, matches):
        calls.append(question)
        return list(reversed(matches))

    plain = Retriever(chunks, KeywordEmbedder(), 0.1, 2).search("free parking")
    reranked = Retriever(chunks, KeywordEmbedder(), 0.1, 2, reranker=reverse).search("free parking")
    assert titles(reranked) == list(reversed(titles(plain))) and calls == ["free parking"]
    assert Retriever(chunks, KeywordEmbedder(), 0.99, 2, reranker=reverse).search("free parking").matches == [] and calls == ["free parking"]


# ---------- The saved index (scripts/ingest.py) ----------
def test_the_fingerprint_changes_when_any_text_changes():
    a, b = [chunk("T", "one")], [chunk("T", "two")]
    assert chunks_fingerprint(a) == chunks_fingerprint([chunk("T", "one")]) != chunks_fingerprint(b)


def test_the_index_can_be_saved_and_loaded(tmp_path):
    chunks = [chunk("Parking", "Free parking behind the building."), chunk("Hours", "Open 8 to 6.")]
    embedder = KeywordEmbedder()
    meta = save_index(build_store(chunks, embedder), tmp_path / "out", "fake-model", chunks)
    assert meta["chunks"] == 2 and meta["embed_model"] == "fake-model" and meta["fingerprint"] == chunks_fingerprint(chunks)
    assert read_meta(tmp_path / "out") == meta

    loaded, loaded_meta = load_index(tmp_path / "out", embedder)
    assert loaded_meta == meta
    best, score = loaded.similarity_search_with_score("is there parking", k=1)[0]
    assert best.metadata["title"] == "Parking" and score > 0.25


def test_loading_a_missing_or_incomplete_index_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_index(tmp_path / "nothing", KeywordEmbedder())
    chunks = [chunk("Parking", "Free parking.")]
    save_index(build_store(chunks, KeywordEmbedder()), tmp_path / "idx", "m", chunks)
    meta_file = tmp_path / "idx" / "meta.json"
    meta_file.write_text(meta_file.read_text(encoding="utf-8").replace('"chunks": 1', '"chunks": 5'), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete"):
        load_index(tmp_path / "idx", KeywordEmbedder())


class NamedEmbedder(KeywordEmbedder):
    model_name = "keyword-test-model"


@pytest.fixture
def index_settings(tmp_path):
    return Settings(_env_file=None, processed_dir=tmp_path / "processed", docs_dir=tmp_path / "raw")


def save_test_index(settings, chunks, model="keyword-test-model"):
    save_index(build_store(chunks, NamedEmbedder()), settings.processed_dir, model, chunks)


def test_the_app_uses_the_saved_index_when_it_matches(index_settings, caplog):
    chunks = [chunk("Parking", "Free parking behind the building.")]
    save_test_index(index_settings, chunks)

    class ExplodingEmbedder(NamedEmbedder):
        def embed_documents(self, texts):               # the chunks must NOT be embedded again
            raise AssertionError("the documents were embedded again")

    with caplog.at_level("INFO"):
        retriever = build_retriever(index_settings, ExplodingEmbedder(), chunks)
    assert retriever.search("is there parking").answerable
    assert any("rag_index_loaded" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("what", ["documents changed", "model changed", "no index"])
def test_the_app_rebuilds_the_index_when_it_does_not_match(index_settings, caplog, what):
    chunks = [chunk("Parking", "Free parking behind the building.")]
    if what != "no index":
        save_test_index(index_settings, [chunk("Parking", "OLD text about parking.")] if what == "documents changed" else chunks,
                        model="keyword-test-model" if what == "documents changed" else "another-model")
    with caplog.at_level("INFO"):
        retriever = build_retriever(index_settings, NamedEmbedder(), chunks)
    assert retriever.search("is there parking").answerable                    # still works: it embedded them itself
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert ("rag_index_missing" if what == "no index" else "rag_index_out_of_date") in messages


def test_a_damaged_index_does_not_stop_the_app(index_settings):
    chunks = [chunk("Parking", "Free parking behind the building.")]
    save_test_index(index_settings, chunks)
    (index_settings.processed_dir / "index.json").write_text("not json", encoding="utf-8")
    assert build_retriever(index_settings, NamedEmbedder(), chunks).search("is there parking").answerable


def test_the_saved_index_can_be_switched_off(index_settings):
    chunks = [chunk("Parking", "Free parking behind the building.")]
    save_test_index(index_settings, [chunk("Parking", "OLD text.")])
    settings = index_settings.model_copy(update={"use_prebuilt_index": False})
    assert build_retriever(settings, NamedEmbedder(), chunks).search("is there parking").answerable


def test_the_ingest_script_checks_a_missing_index(tmp_path):
    result = subprocess.run([sys.executable, "scripts/ingest.py", "--check", "--out", str(tmp_path / "nothing")],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 1 and "No saved index" in result.stdout


# ---------- With the real embedding model (skipped if it cannot be loaded) ----------
@pytest.fixture(scope="module")
def real_retriever():
    try:
        from voice_agent.rag.embedder import build_embedder

        settings = Settings(_env_file=None)
        embedder = build_embedder(settings.embed_model)
    except Exception as e:  # no internet on first run, or model missing
        pytest.skip(f"embedding model not available: {e}")
    return Retriever(load_chunks(settings.docs_dir), embedder, settings.match_threshold, settings.top_k)


@pytest.mark.parametrize("question,topic", [
    ("What time do you open on Saturday?", "Opening hours"),
    ("How much is a check-up?", "Prices"),
    ("Do you take dental insurance?", "Insurance and payment"),
    ("Where is the clinic?", "Location and parking"),
    ("Can I bring my kids?", "Children"),
])
def test_real_model_answers_clinic_questions(real_retriever, question, topic):
    result = real_retriever.search(question)
    assert result.answerable, f"best score {result.best_score:.3f}"
    assert titles(result)[0] == topic


@pytest.mark.parametrize("question", ["Who won the football last night?", "What's the weather like tomorrow?",
                                      "Can you recommend a good pizza place?"])
def test_real_model_refuses_out_of_scope(real_retriever, question):
    result = real_retriever.search(question)
    assert not result.answerable, f"best score {result.best_score:.3f} is above the threshold"
