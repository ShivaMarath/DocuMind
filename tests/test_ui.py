from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def test_app_renders_without_api_keys_and_with_empty_kb(env, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, at.exception
    assert len(at.tabs) == 4
    assert any("knowledge base is empty" in i.value.lower() for i in at.info)


def test_app_renders_with_populated_kb_and_eval_set(env, monkeypatch):
    from langchain_core.embeddings import DeterministicFakeEmbedding

    from documind import evalsets, kb
    from documind.vectorstore import get_vectorstore

    from conftest import make_pdf

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    vs = get_vectorstore(DeterministicFakeEmbedding(size=32))
    kb.add_files([("policy.pdf", make_pdf(["Annual plans can be refunded within 30 days of purchase."]))], vectorstore=vs)
    evalsets.save_set("demo", [
        {"question": "How long is the refund window?", "reference": "30 days", "expected_source": "policy.pdf",
         "must_include": ["30 days"]},
        {"question": "Who is the CEO?", "answerable": False}])

    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, at.exception
    assert not any("knowledge base is empty" in i.value.lower() for i in at.info)
    assert at.selectbox[0].value == "demo"            # eval set picker shows the saved set
