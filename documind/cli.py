"""Command line interface:  python -m documind.cli <command>"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _print_answer(res: dict) -> None:
    print(f"\n{res['answer']}\n")
    for s in res["sources"]:
        page = f" p.{s['page']}" if s["page"] else ""
        print(f"  [{s['n']}] {s['source']}{page}  (score {s['score']})")
    print(f"\n  outcome={res['outcome']}  latency={res['latency_ms']}ms  trace={' > '.join(res['trace'])}\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="documind")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ingest", help="Add PDFs/MD/TXT/DOCX (files or folders) to the knowledge base")
    s.add_argument("paths", nargs="+")
    sub.add_parser("sources", help="List documents in the knowledge base")
    s = sub.add_parser("remove", help="Remove a document from the knowledge base")
    s.add_argument("name")

    s = sub.add_parser("ask", help="Ask a single question")
    s.add_argument("question")
    sub.add_parser("chat", help="Interactive chat with memory")

    s = sub.add_parser("eval-gen", help="Generate a ground-truth eval set from your indexed documents")
    s.add_argument("--name", required=True)
    s.add_argument("--sources", nargs="*", help="Limit to these documents (default: all)")
    s.add_argument("--n", type=int, default=20, help="Number of answerable questions")
    s.add_argument("--unanswerable", type=int, default=5, help="Number of unanswerable questions")
    s.add_argument("--seed", type=int, default=42)
    s = sub.add_parser("eval-import", help="Import your own Q&A ground truth (.csv or .jsonl)")
    s.add_argument("file")
    s.add_argument("--name", required=True)
    sub.add_parser("eval-sets", help="List saved eval sets")
    s = sub.add_parser("eval", help="Run an eval set through the pipeline")
    s.add_argument("--set", dest="set_name", required=True)
    s.add_argument("--no-judge", action="store_true", help="Skip LLM-judge scoring (cheaper)")

    sub.add_parser("stats", help="Usage and quality metrics from logged interactions")
    s = sub.add_parser("serve", help="Run the FastAPI server")
    s.add_argument("--port", type=int, default=8000)
    sub.add_parser("ui", help="Launch the Streamlit web app")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        _run(args)
    except (ValueError, FileNotFoundError) as e:  # user-fixable problems: show a message, not a traceback
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


def _run(args) -> None:

    if args.cmd == "ingest":
        from .ingestion import ingest_paths
        print(json.dumps(ingest_paths(args.paths), indent=2))
    elif args.cmd == "sources":
        from .kb import list_sources
        rows = list_sources()
        for r in rows:
            print(f"{r['source']:<40} chunks={r['chunks']:<5} pages={r['pages']}")
        if not rows:
            print("Knowledge base is empty. Add files with: python -m documind.cli ingest <file-or-folder>")
    elif args.cmd == "remove":
        from .kb import remove_source
        remove_source(args.name)
        print(f"removed {args.name}")
    elif args.cmd == "ask":
        from .graph import RAGPipeline
        _print_answer(RAGPipeline().ask(args.question))
    elif args.cmd == "chat":
        from .graph import RAGPipeline
        rag, history = RAGPipeline(), []
        print("DocuMind chat. Type 'exit' to quit.")
        while (q := input("\nyou> ").strip()) not in {"exit", "quit"}:
            if not q:
                continue
            res = rag.ask(q, history=history)
            _print_answer(res)
            history += [{"role": "user", "content": q}, {"role": "assistant", "content": res["answer"]}]
    elif args.cmd == "eval-gen":
        from .evalgen import generate_eval_set
        from .evalsets import save_set
        from .graph import RAGPipeline
        from .kb import get_chunks
        chunks = get_chunks(args.sources)
        items = generate_eval_set(RAGPipeline(), chunks, args.n, args.unanswerable, args.seed,
                                  progress_cb=lambda stage, d, t: print(f"  {stage}: {d}/{t}"))
        print(f"saved {len(items)} questions -> {save_set(args.name, items)}")
        print("Review/edit them in the web app (python -m documind.cli ui) before trusting the scores.")
    elif args.cmd == "eval-import":
        from .evalsets import parse_upload, save_set
        f = Path(args.file)
        items = parse_upload(f.name, f.read_bytes())
        print(f"saved {len(items)} questions -> {save_set(args.name, items)}")
    elif args.cmd == "eval-sets":
        from .evalsets import list_sets
        for s in list_sets():
            print(f"{s['name']:<30} {s['questions']:>4} questions   {s['modified']}")
    elif args.cmd == "eval":
        from .evalsets import load_set
        from .evaluate import run_eval
        out = run_eval(load_set(args.set_name), name=args.set_name, use_judge=not args.no_judge,
                       progress_cb=lambda i, n, r: print(f"[{i}/{n}] {r['outcome']:<10} judge={r['judge_score']} {r['question'][:70]}"))
        print(json.dumps(out["summary"], indent=2))
        print(f"\nper-question results: {out['csv_path']}")
    elif args.cmd == "stats":
        from .metrics import summarize
        print(json.dumps(summarize(), indent=2))
    elif args.cmd == "serve":
        import uvicorn
        uvicorn.run("documind.api:app", host="0.0.0.0", port=args.port)
    elif args.cmd == "ui":
        app = Path(__file__).resolve().parent.parent / "app.py"
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(app)], check=False)


if __name__ == "__main__":
    main()
