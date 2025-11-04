from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

ROOT = Path(__file__).resolve().parents[1]
_MODULE_PATH = ROOT / "lightrag" / "manufacturing_ingest.py"
_SPEC = importlib.util.spec_from_file_location("manufacturing_ingest", _MODULE_PATH)
assert _SPEC and _SPEC.loader
manufacturing_ingest = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = manufacturing_ingest
_SPEC.loader.exec_module(manufacturing_ingest)

ingest_csv = manufacturing_ingest.ingest_csv
build_manufacturing_custom_kg = manufacturing_ingest.build_manufacturing_custom_kg


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    lines = [",".join(header), *(",".join(row) for row in rows)]
    path.write_text("\n".join(lines), encoding="utf-8")


def test_build_manufacturing_custom_kg_creates_entities_and_relationships(tmp_path: Path) -> None:
    header = [
        "Process Step",
        "Characteristic",
        "Specification",
        "Lower Spec",
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
            "10 +/- 0.5",
            "9.5 mm",
            "10.5 mm",
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
            "Visual",
            "2",
            "Per Shift",
            "Rework",
            "Operator",
        ],
    ]

    csv_path = tmp_path / "control_plan.csv"
    _write_csv(csv_path, header, rows)

    result = ingest_csv(csv_path, doc_type="control_plan", import_batch="batch-kg")
    custom_kg, row_chunk_map = build_manufacturing_custom_kg(
        result, "manufacturing/control_plan.csv"
    )

    # Expect one chunk per vector chunk produced by ingestion
    assert len(custom_kg["chunks"]) == len(result.vector_chunks)
    assert set(row_chunk_map.keys()) == {
        row.row.row_index for row in result.normalized_rows
    }

    # Entities should include hashed identifiers for step and characteristic
    entity_ids = {entity["entity_name"] for entity in custom_kg["entities"]}
    first_row = result.normalized_rows[0]
    assert first_row.step_id in entity_ids
    assert first_row.char_id in entity_ids

    # Relationships should connect the step to the characteristic and measurement method
    relationships = {(rel["src_id"], rel["tgt_id"]) for rel in custom_kg["relationships"]}
    assert (first_row.step_id, first_row.char_id) in relationships
    if first_row.mm_id:
        assert (first_row.step_id, first_row.mm_id) in relationships
