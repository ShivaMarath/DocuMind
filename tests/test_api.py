import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import DeterministicFakeEmbedding

from documind import api
from documind.vectorstore import get_vectorstore

from conftest import make_pdf


class StubRag:
    def __init__(self):
        self.vs = get_vectorstore(DeterministicFakeEmbedding(size=32))


@pytest.fixture
def client(env, monkeypatch):
    monkeypatch.setattr(api, "_rag", StubRag())
    return TestClient(api.app)


def test_upload_pdf_list_delete(client):
    pdf = make_pdf(["Annual plans can be refunded within 30 days of purchase by any customer."])
    r = client.post("/sources", files=[("files", ("../policy.pdf", pdf, "application/pdf")),
                                       ("files", ("notes.exe", b"x", "application/octet-stream"))])
    assert r.status_code == 200 and r.json()["files"] == 1 and r.json()["rejected"] == ["notes.exe"]
    assert [s["source"] for s in client.get("/sources").json()] == ["policy.pdf"]
    assert client.delete("/sources/policy.pdf").status_code == 200
    assert client.get("/sources").json() == []


def test_import_and_list_eval_sets(client):
    csv = b"question,reference,expected_source\nHow long is the refund window?,30 days,policy.pdf\n"
    r = client.post("/eval/import", params={"name": "mine"}, files={"file": ("qa.csv", csv, "text/csv")})
    assert r.json() == {"name": "mine", "questions": 1}
    assert client.get("/eval/sets").json()[0]["name"] == "mine"
    assert client.get("/eval/sets/mine").json()[0]["expected_source"] == "policy.pdf"
    assert client.get("/eval/sets/nope").status_code == 404
    bad = client.post("/eval/import", params={"name": "x"}, files={"file": ("qa.csv", b"a\n1\n", "text/csv")})
    assert bad.status_code == 400


def test_generate_on_empty_kb_returns_400(client):
    r = client.post("/eval/generate", json={"name": "g"})
    assert r.status_code == 400 and "empty" in r.json()["detail"].lower()
