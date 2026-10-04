import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def make_pdf(lines: list[str]) -> bytes:
    """Build a minimal but valid one-page PDF containing the given text lines (no extra deps)."""
    content = "BT /F1 12 Tf 72 720 Td 14 TL\n" + "\n".join(f"({l}) Tj T*" for l in lines) + "\nET"
    objs = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        "/Resources << /Font << /F1 5 0 R >> >> >>",
        f"<< /Length {len(content)} >>\nstream\n{content}\nendstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out, offsets = b"%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{o}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    return out


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Isolated Qdrant / uploads / eval sets / metrics DB for each test."""
    monkeypatch.setenv("QDRANT_PATH", str(tmp_path / "qdrant"))
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("EVAL_SETS_DIR", str(tmp_path / "eval_sets"))
    monkeypatch.setenv("METRICS_DB", str(tmp_path / "m.db"))
    monkeypatch.delenv("QDRANT_URL", raising=False)
    from documind.config import get_settings
    from documind.vectorstore import get_client

    get_settings.cache_clear()
    get_client.cache_clear()
    yield tmp_path
    get_client.cache_clear()
    get_settings.cache_clear()
