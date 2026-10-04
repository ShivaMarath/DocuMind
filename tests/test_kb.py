from langchain_core.embeddings import DeterministicFakeEmbedding

from documind import kb
from documind.ingestion import chunk_id
from documind.vectorstore import get_vectorstore

from conftest import make_pdf

LONG = "The refund window for annual plans is thirty days from purchase. " * 30


def _vs():
    return get_vectorstore(DeterministicFakeEmbedding(size=32))


def test_upload_pdf_list_and_remove(env):
    vs = _vs()
    pdf = make_pdf(["Annual plans can be refunded within 30 days of purchase.",
                    "Monthly plans are not refundable."])
    summary = kb.add_files([("policy.pdf", pdf), ("virus.exe", b"x")], vectorstore=vs)
    assert summary["files"] == 1 and summary["rejected"] == ["virus.exe"]
    assert summary["details"][0].get("error") is None

    rows = kb.list_sources()
    assert [r["source"] for r in rows] == ["policy.pdf"] and rows[0]["pages"] == 1
    chunks = kb.get_chunks()
    assert "30 days" in chunks[0].page_content
    assert (kb.upload_dir() / "policy.pdf").exists()

    kb.remove_source("policy.pdf")
    assert kb.list_sources() == [] and not (kb.upload_dir() / "policy.pdf").exists()


def test_reupload_replaces_stale_chunks_and_get_chunks_ids_match(env):
    vs = _vs()
    kb.add_files([("a.txt", LONG.encode())], vectorstore=vs)
    before = {c.metadata["chunk_id"] for c in kb.get_chunks()}
    # chunk ids stored in Qdrant must equal ids recomputed from a retrieved document (used for eval scoring)
    hit = vs.similarity_search("refund window", k=1)[0]
    assert chunk_id(hit) in before

    kb.add_files([("a.txt", b"Completely different content about vacation days. " * 40)], vectorstore=vs)
    after = {c.metadata["chunk_id"] for c in kb.get_chunks()}
    assert before.isdisjoint(after) and len(kb.list_sources()) == 1


def test_scanned_pdf_reports_no_text_and_bad_file_does_not_abort(env):
    vs = _vs()
    blank = make_pdf([""])
    s = kb.add_files([("scan.pdf", blank), ("ok.txt", LONG.encode()), ("broken.pdf", b"not a pdf")], vectorstore=vs)
    notes = {d["file"]: d for d in s["details"]}
    assert "no extractable text" in notes["scan.pdf"]["note"]
    assert "error" in notes["broken.pdf"]
    assert notes["ok.txt"]["chunks"] > 0 and s["files"] == 1


def test_get_chunks_filters_by_source(env):
    vs = _vs()
    kb.add_files([("a.txt", LONG.encode()), ("b.txt", (LONG + " extra").encode())], vectorstore=vs)
    assert {c.metadata["source_name"] for c in kb.get_chunks(["b.txt"])} == {"b.txt"}
    assert kb.safe_name("../../etc/passwd") == "passwd"
