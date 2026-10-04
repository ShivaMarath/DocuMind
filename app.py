"""DocuMind AI web app.   Run:  streamlit run app.py   (or: python -m documind.cli ui)"""
from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from documind import evalsets, kb
from documind.config import get_settings
from documind.metrics import load_interactions, record_feedback, summarize

st.set_page_config(page_title="DocuMind AI", page_icon="📄", layout="wide")
settings = get_settings()


@st.cache_resource
def get_rag():
    from documind.graph import RAGPipeline

    return RAGPipeline()


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("📄 DocuMind AI")
    st.caption(f"Provider: **{settings.provider}**  \nChat model: `{settings.chat_model}`  \n"
               f"Embeddings: `{settings.embedding_model}`")
    key_name = "OPENAI_API_KEY" if settings.provider == "openai" else "GOOGLE_API_KEY"
    if not os.getenv(key_name):
        entered = st.text_input(f"{key_name}", type="password", help="Kept in memory for this session only.")
        if entered:
            os.environ[key_name] = entered
            st.rerun()
        st.warning(f"Set {key_name} in .env or paste it above.")
    st.caption("Qdrant: " + (settings.qdrant_url or f"embedded ({settings.qdrant_path})"))

tab_kb, tab_chat, tab_eval, tab_stats = st.tabs(["📚 Knowledge base", "💬 Chat", "🧪 Evaluation", "📈 Analytics"])

# ------------------------------------------------------------------ knowledge base
with tab_kb:
    st.subheader("Add documents")
    st.caption("Upload PDFs (also DOCX, MD, TXT). Re-uploading a file with the same name replaces the old version. "
               "Scanned/image-only PDFs have no text layer and can't be indexed (no OCR).")
    if flash := st.session_state.pop("kb_flash", None):
        for level, msg in flash:
            getattr(st, level)(msg)

    n = st.session_state.get("uploader_n", 0)
    uploads = st.file_uploader("Drop files here", type=["pdf", "docx", "md", "txt"],
                               accept_multiple_files=True, key=f"uploader_{n}")
    if st.button("Index uploaded files", type="primary", disabled=not uploads):
        try:
            with st.spinner("Extracting, chunking and embedding..."):
                summary = kb.add_files([(f.name, f.getvalue()) for f in uploads], vectorstore=get_rag().vs)
            flash = []
            for d in summary["details"]:
                if d.get("error"):
                    flash.append(("error", f"{d['file']}: {d['error']}"))
                elif d.get("note"):
                    flash.append(("warning", f"{d['file']}: {d['note']}"))
                else:
                    flash.append(("success", f"{d['file']}: indexed {d['chunks']} chunks"))
            flash += [("warning", f"{r}: unsupported file type") for r in summary["rejected"]]
            st.session_state["kb_flash"] = flash
            st.session_state["uploader_n"] = n + 1
            st.rerun()
        except Exception as e:  # noqa: BLE001
            st.error(f"Indexing failed: {e}")

    st.subheader("Indexed documents")
    sources = kb.list_sources()
    if sources:
        st.dataframe(pd.DataFrame(sources), use_container_width=True, hide_index=True)
        to_remove = st.multiselect("Remove documents", [s["source"] for s in sources])
        if st.button("Delete selected", disabled=not to_remove):
            for name in to_remove:
                kb.remove_source(name)
            st.rerun()
    else:
        st.info("Your knowledge base is empty. Upload a document above to get started.")

# ------------------------------------------------------------------ chat
with tab_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    top = st.columns([6, 1])
    top[0].subheader("Ask your documents")
    if top[1].button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("sources"):
                with st.expander(f"Sources ({len(m['sources'])})"):
                    for s in m["sources"]:
                        page = f", p.{s['page']}" if s["page"] else ""
                        st.markdown(f"**[{s['n']}] {s['source']}{page}**  \n> {s['snippet']}...")
            if m.get("interaction_id"):
                fb = st.feedback("thumbs", key=f"fb_{m['interaction_id']}")
                if fb is not None:
                    record_feedback(m["interaction_id"], bool(fb))

    if prompt := st.chat_input("Ask a question about your documents"):
        if not kb.list_sources():
            st.warning("Add at least one document in the Knowledge base tab first.")
        else:
            history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
            st.session_state.messages.append({"role": "user", "content": prompt})
            try:
                with st.spinner("Retrieving, checking relevance, answering, verifying..."):
                    res = get_rag().ask(prompt, history=history)
                caption = (f"\n\n*{res['outcome']} · {res['latency_ms'] / 1000:.1f}s · "
                           f"{' → '.join(res['trace'])}*")
                st.session_state.messages.append({
                    "role": "assistant", "content": res["answer"] + caption,
                    "sources": res["sources"], "interaction_id": res.get("interaction_id")})
            except Exception as e:  # noqa: BLE001
                st.session_state.messages.append({"role": "assistant", "content": f"⚠️ Error: {e}"})
            st.rerun()

# ------------------------------------------------------------------ evaluation
with tab_eval:
    st.subheader("1 · Build ground truth from your documents")
    mode = st.radio("How do you want to create the ground truth?",
                    ["Generate from my uploaded documents", "Upload my own Q&A file"], horizontal=True)
    set_name = st.text_input("Name for this eval set", value="my-eval")

    if mode.startswith("Generate"):
        all_sources = [s["source"] for s in kb.list_sources()]
        chosen = st.multiselect("Documents to draw questions from", all_sources, default=all_sources)
        c1, c2, c3 = st.columns(3)
        n_ans = c1.slider("Answerable questions", 5, 100, 20)
        n_unans = c2.slider("Unanswerable (hallucination traps)", 0, 20, 5)
        seed = c3.number_input("Random seed", value=42, step=1)
        st.caption("Each question is written by the LLM from a real passage of your documents; that passage is stored "
                   "as the ground-truth evidence, so retrieval can be scored exactly. Unanswerable questions are "
                   "verified against your index. **Review them below before trusting the scores.**")
        if st.button("Generate ground truth", type="primary", disabled=not chosen):
            from documind.evalgen import generate_eval_set

            bar = st.progress(0.0, text="Generating questions...")
            total = n_ans + n_unans

            def _cb(stage, done, tot):
                offset = 0 if stage == "answerable" else n_ans
                bar.progress(min((offset + done) / max(total, 1), 1.0), text=f"{stage}: {done}/{tot}")

            try:
                items = generate_eval_set(get_rag(), kb.get_chunks(chosen), n_ans, n_unans, int(seed), _cb)
                evalsets.save_set(set_name, items)
                st.session_state["eval_selected"] = evalsets.slug(set_name)
                st.success(f"Saved {len(items)} questions as '{evalsets.slug(set_name)}'.")
            except Exception as e:  # noqa: BLE001
                st.error(f"Generation failed: {e}")
            finally:
                bar.empty()
    else:
        st.caption("CSV or JSONL with a `question` column (required) and optional `reference`, `expected_source`, "
                   "`answerable` (true/false), `must_include` (a;b;c).")
        up = st.file_uploader("Q&A file", type=["csv", "jsonl", "json"], key="qa_upload")
        if st.button("Import", type="primary", disabled=not up):
            try:
                items = evalsets.parse_upload(up.name, up.getvalue())
                evalsets.save_set(set_name, items)
                st.session_state["eval_selected"] = evalsets.slug(set_name)
                st.success(f"Imported {len(items)} questions as '{evalsets.slug(set_name)}'.")
            except Exception as e:  # noqa: BLE001
                st.error(f"Import failed: {e}")

    st.divider()
    st.subheader("2 · Review and edit")
    saved = evalsets.list_sets()
    if not saved:
        st.info("No eval sets yet. Create one above.")
    else:
        names = [s["name"] for s in saved]
        default_idx = names.index(st.session_state["eval_selected"]) if st.session_state.get("eval_selected") in names else 0
        selected = st.selectbox("Eval set", names, index=default_idx)
        items = evalsets.load_set(selected)
        edited = st.data_editor(
            evalsets.to_frame(items), num_rows="dynamic", use_container_width=True, hide_index=True,
            key=f"editor_{selected}_{len(items)}",
            disabled=["id", "origin", "expected_chunk_id", "evidence"],
            column_config={
                "answerable": st.column_config.CheckboxColumn("answerable"),
                "must_include": st.column_config.TextColumn("must_include (a; b; c)"),
                "evidence": st.column_config.TextColumn("evidence (source passage)", width="large"),
            })
        b1, b2, _ = st.columns([1, 1, 4])
        if b1.button("Save edits"):
            try:
                evalsets.save_set(selected, evalsets.from_frame(edited))
                st.success("Saved.")
                st.rerun()
            except Exception as e:  # noqa: BLE001
                st.error(str(e))
        if b2.button("Delete set"):
            evalsets.delete_set(selected)
            st.rerun()

        st.divider()
        st.subheader("3 · Run evaluation")
        use_judge = st.checkbox("Score answers with an LLM judge (extra cost)", value=True)
        if st.button("Run evaluation", type="primary"):
            from documind.evaluate import run_eval

            bar = st.progress(0.0, text="Running...")
            try:
                out = run_eval(evalsets.load_set(selected), name=selected, pipeline=get_rag(), use_judge=use_judge,
                               progress_cb=lambda i, n, r: bar.progress(i / n, text=f"{i}/{n}: {r['question'][:60]}"))
                st.session_state["last_eval"] = out
            except Exception as e:  # noqa: BLE001
                st.error(f"Evaluation failed: {e}")
            finally:
                bar.empty()

        if out := st.session_state.get("last_eval"):
            sm = out["summary"]
            st.markdown("#### Results")
            labels = [("source_hit_rate", "Retrieval hit (doc)"), ("chunk_hit_rate", "Retrieval hit (chunk)"),
                      ("chunk_mrr", "Chunk MRR"), ("avg_judge_score", "Judge score /5"),
                      ("abstention_accuracy_on_unanswerable", "Abstains when unanswerable"),
                      ("grounded_rate", "Grounded rate"), ("latency_p50_ms", "Latency p50 (ms)")]
            cols = st.columns(len(labels))
            for col, (k, label) in zip(cols, labels):
                v = sm.get(k)
                col.metric(label, "n/a" if v is None else v)
            if sm.get("n_errors"):
                st.warning(f"{sm['n_errors']} question(s) errored - see the 'error' column.")
            st.dataframe(out["results"], use_container_width=True, hide_index=True)
            st.download_button("Download results CSV", out["results"].to_csv(index=False), file_name="eval_results.csv")

# ------------------------------------------------------------------ analytics
with tab_stats:
    st.subheader("Live usage and quality")
    df = load_interactions()
    if df.empty:
        st.info("No questions asked yet. Metrics appear here as you use the Chat tab.")
    else:
        sm = summarize(df)
        cols = st.columns(5)
        cols[0].metric("Questions", sm["total_queries"])
        cols[1].metric("Answered", f"{sm['answered_rate']:.0%}")
        cols[2].metric("Self-corrected", f"{sm['self_correction_rate']:.0%}")
        cols[3].metric("Latency p50 (s)", f"{sm['latency_p50_ms'] / 1000:.1f}")
        cols[4].metric("Helpful (thumbs)", "n/a" if sm["helpful_rate"] is None else f"{sm['helpful_rate']:.0%}")
        left, right = st.columns(2)
        left.markdown("**Outcomes**")
        left.bar_chart(df["outcome"].value_counts())
        right.markdown("**Latency per question (ms)**")
        right.line_chart(df["latency_ms"])
        st.dataframe(df.sort_values("ts", ascending=False), use_container_width=True, hide_index=True)
