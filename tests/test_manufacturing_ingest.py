from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

_MODULE_PATH = ROOT / "lightrag" / "manufacturing_ingest.py"
spec = importlib.util.spec_from_file_location("manufacturing_ingest", _MODULE_PATH)
assert spec and spec.loader
manufacturing_ingest = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = manufacturing_ingest
spec.loader.exec_module(manufacturing_ingest)

IngestionError = manufacturing_ingest.IngestionError
ingest_csv = manufacturing_ingest.ingest_csv


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    lines = [",".join(header), *(",".join(row) for row in rows)]
    path.write_text("\n".join(lines), encoding="utf-8")


def test_ingest_csv_builds_graph_and_vector_payload(tmp_path: Path) -> None:
    header = [
        "Process Step",
        "Characteristic",
        "Specification",
        "Lower Spec",
        "Target",
        "Upper Spec",
        "Measurement Method",
        "Sample Size",
        "Sample Frequency",
        "Reaction Plan",
        "Responsibility",
    ]
    rows = [
        [
            "Drill Hole #3",
            "Diameter",
            "Diameter within tolerance",
            "10 mm",
            "10.5 mm",
            "11 mm",
            "CMM",
            "5",
            "Every Lot",
            "Notify supervisor",
            "Quality Tech",
        ],
        [
            "Deburr",
            "Surface Finish",
            "Smooth",
            "",
            "",
            "",
            "Visual",
            "2",
            "Per Shift",
            "Rework",
            "Operator",
        ],
    ]

    csv_path = tmp_path / "control_plan_sample.csv"
    _write_csv(csv_path, header, rows)

    result = ingest_csv(csv_path, doc_type="control_plan", import_batch="batch-1")

    expected_doc_id = hashlib.sha1("control_plan||control_plan_sample.csv".encode("utf-8")).hexdigest()
    assert result.document.doc_id == expected_doc_id
    assert len(result.cypher_parameters) == 2

    first_row = result.cypher_parameters[0]
    expected_step_id = hashlib.sha1(f"{expected_doc_id}||1".encode("utf-8")).hexdigest()
    assert first_row["step_id"] == expected_step_id
    assert first_row["char_id"] == hashlib.sha1("diameter".encode("utf-8")).hexdigest()
    assert first_row["lower_num"] == 10.0
    assert first_row["lower_unit"] == "mm"

    # Two data rows + one heading chunk
    assert len(result.vector_chunks) == 3
    heading_chunk = result.vector_chunks[0]
    assert heading_chunk.metadata["chunk_type"] == "heading"
    row_chunk = result.vector_chunks[1]
    assert "Process Step: Drill Hole #3" in row_chunk.text
    assert row_chunk.metadata["row_index"] == 1
    assert row_chunk.metadata["column_map"]["Process Step"] == "Drill Hole #3"


def test_ingest_csv_missing_required_column(tmp_path: Path) -> None:
    header = [
        "Characteristic",
        "Specification",
    ]
    rows = [["Diameter", "10 +/- 0.5"]]
    csv_path = tmp_path / "invalid.csv"
    _write_csv(csv_path, header, rows)

    with pytest.raises(IngestionError):
        ingest_csv(csv_path, doc_type="control_plan", import_batch="batch-1")


def test_ingest_csv_parses_spec_parse_error_flag(tmp_path: Path) -> None:
    header = [
        "Process Step",
        "Characteristic",
        "Lower Spec",
    ]
    rows = [["Measure", "Length", "N/A"]]
    csv_path = tmp_path / "spec_parse.csv"
    _write_csv(csv_path, header, rows)

    result = ingest_csv(csv_path, doc_type="process_flow", import_batch="batch-x")
    params = result.cypher_parameters[0]
    assert params["lower"] == "N/A"
    assert params["lower_parse_error"] is True


def test_ingest_csv_allows_idempotent_reimport(tmp_path: Path) -> None:
    header = ["Process Step", "Characteristic"]
    rows = [["Assemble", "Torque"], ["Inspect", "Gap"]]
    csv_path = tmp_path / "reimport.csv"
    _write_csv(csv_path, header, rows)

    first = ingest_csv(csv_path, doc_type="ppap", import_batch="batch-a")
    second = ingest_csv(
        csv_path,
        doc_type="ppap",
        import_batch="batch-b",
        allow_reimport=True,
        existing_row_checksums=first.row_checksums,
    )

    # All rows skipped, only heading chunk remains
    assert second.normalized_rows == []
    assert second.skipped_rows == [1, 2]
    assert len(second.vector_chunks) == 1
    assert second.vector_chunks[0].metadata["chunk_type"] == "heading"

