"""Tests for app/core/ingest.py.

SCOPE BOUNDARY under test: no canonical schema, no renaming, no coercion, no
row-dropping-without-reporting. See ingest.py's module docstring.
"""

from __future__ import annotations

import io

import pandas as pd
import pytest

from app.core import ingest, storage


# ---------------------------------------------------------------------------
# Encoding detection
# ---------------------------------------------------------------------------


def test_detect_encoding_plain_utf8():
    encoding, text = ingest.detect_encoding("hello,world".encode("utf-8"))
    assert encoding == "utf-8"
    assert text == "hello,world"


def test_detect_encoding_bom_stripped():
    raw = b"\xef\xbb\xbfid,name\n1,Alice\n"
    encoding, text = ingest.detect_encoding(raw)
    assert encoding == "utf-8-sig"
    assert text.startswith("id,name")  # no stray ﻿ glued to the header
    assert "﻿" not in text


def test_detect_encoding_latin1_fallback():
    # a byte sequence that is invalid utf-8 but decodes fine as latin-1
    raw = "café".encode("latin-1")
    encoding, text = ingest.detect_encoding(raw)
    assert encoding in ("utf-8", "latin-1")  # "café" happens to also be valid utf-8; just must not raise
    assert isinstance(text, str)

    genuinely_invalid_utf8 = b"\xff\xfe\x00invalid"
    encoding2, text2 = ingest.detect_encoding(genuinely_invalid_utf8)
    assert encoding2 == "latin-1"
    assert isinstance(text2, str)


# ---------------------------------------------------------------------------
# Delimiter detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("delim,label", [(",", "comma"), ("|", "pipe"), ("\t", "tab"), (";", "semicolon")])
def test_detect_delimiter_each_candidate(delim, label):
    text = delim.join(["a", "b", "c"]) + "\n" + delim.join(["1", "2", "3"]) + "\n" + delim.join(["4", "5", "6"]) + "\n"
    detection = ingest.detect_delimiter(text)
    assert detection.delimiter == delim
    assert detection.label == label
    assert detection.consistency == 1.0


def test_detect_delimiter_avoids_degenerate_single_column_trap():
    # comma-delimited: pipe/tab/semicolon would all "consistently" produce 1 column
    text = "a,b,c\n1,2,3\n4,5,6\n7,8,9\n"
    detection = ingest.detect_delimiter(text)
    assert detection.delimiter == ","
    assert detection.most_common_field_count == 3


def test_detect_delimiter_picks_most_consistent():
    # pipe is consistent at 3 fields; commas appear inconsistently inside values too
    text = "a|b|c\n1|2,x|3\n4|5,y|6\n7|8,z|9\n"
    detection = ingest.detect_delimiter(text)
    assert detection.delimiter == "|"


def test_detect_delimiter_empty_text_raises():
    with pytest.raises(ValueError):
        ingest.detect_delimiter("")


# ---------------------------------------------------------------------------
# Reading as strings, malformed row counting
# ---------------------------------------------------------------------------


def test_load_raw_file_never_coerces_values():
    text = "id,code,score\n007,00A,1.50\n042,00B,2.00\n"
    result = ingest.load_raw_file(text.encode("utf-8"), "x.csv")
    assert result.df["id"].tolist() == ["007", "042"]  # leading zero preserved
    assert result.df["score"].tolist() == ["1.50", "2.00"]  # trailing zero preserved
    assert not any(pd.api.types.is_numeric_dtype(result.df[c]) for c in result.df.columns)  # nothing coerced


def test_load_raw_file_counts_malformed_rows_without_dropping_silently():
    text = "a,b,c\n1,2,3\n4,5,6,extra\n7,8\n9,10,11\n"
    result = ingest.load_raw_file(text.encode("utf-8"), "messy.csv")
    assert result.malformed_row_count == 2
    assert any("extra" in ex for ex in result.malformed_row_examples)
    assert any(ex == "7,8" for ex in result.malformed_row_examples)
    # the over-long row is skipped (can't be represented rectangularly); the
    # short row is kept and NaN-padded — both counted above, neither silent
    assert len(result.df) == 3
    assert pd.isna(result.df.loc[result.df["a"] == "7", "c"]).all()


def test_load_raw_file_reports_delimiter_and_encoding():
    raw = b"\xef\xbb\xbf" + "a|b\n1|2\n3|4\n".encode("utf-8")
    result = ingest.load_raw_file(raw, "x.csv")
    assert result.delimiter == "|"
    assert result.encoding == "utf-8-sig"
    assert result.row_count == 2
    assert result.col_count == 2


def test_load_raw_file_unsupported_extension_raises():
    with pytest.raises(ValueError):
        ingest.load_raw_file(b"whatever", "x.pdf")


# ---------------------------------------------------------------------------
# Excel
# ---------------------------------------------------------------------------


def _make_excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)
    return buf.getvalue()


def test_list_excel_sheets():
    data = _make_excel_bytes({"Leads": pd.DataFrame({"id": [1]}), "Notes": pd.DataFrame({"x": [1]})})
    assert ingest.list_excel_sheets(data) == ["Leads", "Notes"]


def test_load_raw_file_excel_defaults_to_first_sheet():
    data = _make_excel_bytes({"Leads": pd.DataFrame({"id": [1, 2]}), "Notes": pd.DataFrame({"x": [1]})})
    result = ingest.load_raw_file(data, "book.xlsx")
    assert result.sheet_name == "Leads"
    assert result.available_sheets == ["Leads", "Notes"]
    assert result.row_count == 2
    assert result.malformed_row_count == 0
    assert result.delimiter is None


def test_load_raw_file_excel_explicit_sheet_choice():
    data = _make_excel_bytes({"Leads": pd.DataFrame({"id": [1, 2]}), "Notes": pd.DataFrame({"x": [1, 2, 3]})})
    result = ingest.load_raw_file(data, "book.xlsx", sheet_name="Notes")
    assert result.sheet_name == "Notes"
    assert result.row_count == 3


def test_load_raw_file_excel_reads_as_strings():
    data = _make_excel_bytes({"Sheet1": pd.DataFrame({"id": [7, 42], "score": [1.5, 2.0]})})
    result = ingest.load_raw_file(data, "book.xlsx")
    assert not any(pd.api.types.is_numeric_dtype(result.df[c]) for c in result.df.columns)


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


def test_content_hash_of_is_deterministic():
    assert ingest.content_hash_of(b"hello") == ingest.content_hash_of(b"hello")
    assert ingest.content_hash_of(b"hello") != ingest.content_hash_of(b"world")


def test_save_raw_dataset_is_byte_exact_and_appends_on_duplicate(tmp_path):
    store = storage.MetadataStore(tmp_path / "meta.db")
    original_bytes = b"a,b\n1,2\n3,4\n"
    result = ingest.load_raw_file(original_bytes, "leads.csv")

    save1 = ingest.save_raw_dataset(store=store, root=tmp_path, merchant="Evoke", purpose="train", result=result, uploaded_by="tester")
    assert save1.duplicate_of == []
    assert save1.stored_path.read_bytes() == original_bytes
    assert save1.parquet_path.exists()

    row = store.get_raw_dataset(save1.raw_dataset_id)
    assert row["merchant"] == "Evoke"
    assert row["original_filename"] == "leads.csv"
    assert row["content_hash"] == save1.content_hash

    save2 = ingest.save_raw_dataset(store=store, root=tmp_path, merchant="Evoke", purpose="train", result=result, uploaded_by="tester")
    assert len(save2.duplicate_of) == 1
    assert save2.duplicate_of[0]["raw_dataset_id"] == save1.raw_dataset_id
    assert save1.raw_dataset_id != save2.raw_dataset_id  # append-only: a new row either way
    assert len(store.list_raw_datasets(merchant="Evoke")) == 2


def test_save_raw_dataset_preserves_original_filename_on_disk(tmp_path):
    store = storage.MetadataStore(tmp_path / "meta.db")
    result = ingest.load_raw_file(b"a,b\n1,2\n", "quarterly leads (v2).csv")
    save = ingest.save_raw_dataset(store=store, root=tmp_path, merchant="Evoke", purpose="train", result=result, uploaded_by="tester")
    assert save.stored_path.name.endswith("_quarterly_leads__v2_.csv")


# ---------------------------------------------------------------------------
# Cleaned-file storage (Tab 3's second upload point)
# ---------------------------------------------------------------------------


def test_save_cleaned_dataset_is_byte_exact_and_appends_on_duplicate(tmp_path):
    store = storage.MetadataStore(tmp_path / "meta.db")
    original_bytes = b"lead_id,phone\n1,9000000001\n2,9000000002\n"
    result = ingest.load_raw_file(original_bytes, "leads_cleaned.csv")

    save1 = ingest.save_cleaned_dataset(store=store, root=tmp_path, merchant="Evoke", purpose="train", result=result, uploaded_by="tester")
    assert save1.duplicate_of == []
    assert save1.stored_path.read_bytes() == original_bytes
    assert save1.parquet_path.exists()

    row = store.get_cleaned_dataset(save1.cleaned_dataset_id)
    assert row["merchant"] == "Evoke"
    assert row["source_raw_dataset_id"] is None

    save2 = ingest.save_cleaned_dataset(store=store, root=tmp_path, merchant="Evoke", purpose="train", result=result, uploaded_by="tester")
    assert len(save2.duplicate_of) == 1
    assert save1.cleaned_dataset_id != save2.cleaned_dataset_id
    assert len(store.list_cleaned_datasets(merchant="Evoke")) == 2


def test_save_cleaned_dataset_records_lineage_link(tmp_path):
    store = storage.MetadataStore(tmp_path / "meta.db")
    raw_result = ingest.load_raw_file(b"a,b\n1,2\n", "raw.csv")
    raw_save = ingest.save_raw_dataset(store=store, root=tmp_path, merchant="Evoke", purpose="train", result=raw_result, uploaded_by="tester")

    cleaned_result = ingest.load_raw_file(b"a,b\n1,2\n", "cleaned.csv")
    save = ingest.save_cleaned_dataset(
        store=store, root=tmp_path, merchant="Evoke", purpose="train", result=cleaned_result,
        uploaded_by="tester", source_raw_dataset_id=raw_save.raw_dataset_id,
    )
    row = store.get_cleaned_dataset(save.cleaned_dataset_id)
    assert row["source_raw_dataset_id"] == raw_save.raw_dataset_id
