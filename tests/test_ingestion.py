from documind.ingestion import chunk_id, load_file, split_documents


def test_load_and_split_markdown(tmp_path):
    f = tmp_path / "notes.md"
    f.write_text("# Title\n\n" + ("Sentence about refunds. " * 200))
    docs = load_file(f)
    chunks = split_documents(docs, chunk_size=300, chunk_overlap=30)
    assert len(chunks) > 3
    assert all(c.metadata["source_name"] == "notes.md" for c in chunks)
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))


def test_chunk_ids_are_deterministic_and_unique(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("alpha beta gamma. " * 120)
    chunks = split_documents(load_file(f), chunk_size=200, chunk_overlap=20)
    ids1 = [chunk_id(c) for c in chunks]
    ids2 = [chunk_id(c) for c in split_documents(load_file(f), chunk_size=200, chunk_overlap=20)]
    assert ids1 == ids2
    assert len(set(ids1)) == len(ids1)


def test_unsupported_extension(tmp_path):
    f = tmp_path / "x.exe"
    f.write_text("nope")
    try:
        load_file(f)
        assert False, "should have raised"
    except ValueError:
        pass
