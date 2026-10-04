import pytest

from documind import evalsets


def test_import_csv_and_jsonl(env):
    csv = b"question,reference,expected_source,answerable,must_include\n" \
          b"What is the refund window?,30 days,policy.pdf,true,30 days;annual\n" \
          b"Who is the CEO?,,,false,\n"
    items = evalsets.parse_upload("qa.csv", csv)
    assert items[0]["must_include"] == ["30 days", "annual"] and items[0]["answerable"] is True
    assert items[1]["answerable"] is False and items[1]["expected_source"] is None

    jl = b'{"question": "Q1", "reference": "A1"}\n{"question": "Q2", "answerable": "no"}\n'
    items = evalsets.parse_upload("qa.jsonl", jl)
    assert [i["answerable"] for i in items] == [True, False]

    with pytest.raises(ValueError):
        evalsets.parse_upload("qa.csv", b"reference\nfoo\n")  # missing question column
    with pytest.raises(ValueError):
        evalsets.parse_upload("qa.pdf", b"")


def test_save_load_list_delete_and_frame_roundtrip(env):
    items = evalsets.parse_upload("qa.jsonl", b'{"question": "Q1", "reference": "A1", "must_include": ["x","y"]}\n')
    evalsets.save_set("My Set!", items)
    assert evalsets.list_sets()[0]["name"] == "My-Set"
    loaded = evalsets.load_set("My Set!")
    assert loaded == items

    df = evalsets.to_frame(loaded)
    df.loc[len(df)] = {"question": ""}   # blank row from the editor is ignored
    df.loc[len(df)] = {"question": "Added by hand", "answerable": False}
    back = evalsets.from_frame(df)
    assert [i["question"] for i in back] == ["Q1", "Added by hand"]
    assert back[0]["must_include"] == ["x", "y"] and back[1]["id"]

    evalsets.delete_set("My Set!")
    assert evalsets.list_sets() == []
    with pytest.raises(FileNotFoundError):
        evalsets.load_set("My Set!")
