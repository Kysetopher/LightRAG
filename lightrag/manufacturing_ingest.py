"""Structured ingestion utilities for manufacturing control documents.

This module normalizes CSV exports for manufacturing quality artifacts
(Control Plan, Process Flow, FMEA, PPAP) into deterministic graph and
vector representations.  The implementation follows the ingestion
contract described in the manufacturing ingestion specification:

* Normalize noisy CSV input and compute stable identifiers using SHA-1
  hashes over normalized strings.
* Validate required columns per document type and fail fast on missing
  data that is mandatory for identifier materialization.
* Emit Cypher parameter dictionaries that can be fed into a batched
  ``MERGE`` pipeline for Neo4j (or any compatible labelled property
  graph engine).
* Produce compact, citation-friendly vector chunks for LightRAG's
  vector storage with metadata that preserves provenance.
* Provide helper utilities for enforcing Neo4j constraints that keep the
  graph consistent.

The main entry point is :func:`ingest_csv`, which reads a CSV file,
validates the schema, normalizes rows, and returns a
``ManufacturingIngestionResult`` containing graph parameters, vector
chunks, and bookkeeping metadata suitable for downstream storage layers.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple


class IngestionError(ValueError):
    """Raised when CSV input violates ingestion invariants."""


def _normalize_whitespace(value: str) -> str:
    """Collapse repeated whitespace and trim the input string."""

    return re.sub(r"\s+", " ", value.strip())


def _normalize_for_id(value: str) -> str:
    """Normalize text for hash materialization."""

    return _normalize_whitespace(value).casefold()


def _normalize_header_key(name: str) -> str:
    """Normalize column headers for alias matching."""

    collapsed = re.sub(r"[\s/]+", " ", name.strip().casefold())
    return collapsed


def _stable_hash(*values: str) -> str:
    """Create a deterministic SHA-1 hash for the provided values."""

    joined = "||".join(values)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


_NUMERIC_WITH_UNIT = re.compile(
    r"^\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([^\d]*)$"
)


@dataclass(frozen=True)
class Row:
    """Raw row extracted from a CSV file."""

    doc_type: str
    filename: str
    row_index: int
    columns: Dict[str, str]
    header_lookup: Mapping[str, str]


@dataclass(frozen=True)
class FieldValue:
    """Represents a resolved column value with provenance information."""

    header: Optional[str]
    raw: Optional[str]
    normalized: Optional[str]


@dataclass(frozen=True)
class SpecBound:
    """A parsed specification bound (e.g., lower/upper limit)."""

    raw: Optional[str]
    normalized: Optional[str]
    numeric: Optional[float]
    unit: Optional[str]
    parse_error: bool = False


@dataclass(frozen=True)
class DocumentMetadata:
    """Metadata shared by all rows originating from a single CSV file."""

    doc_type: str
    filename: str
    import_batch: str
    doc_id: str
    created_at: str


@dataclass(frozen=True)
class VectorChunk:
    """Vector chunk with provenance metadata for LightRAG storage."""

    chunk_id: str
    text: str
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class DocumentSchema:
    """Defines column aliases and required fields for a document type."""

    doc_type: str
    field_aliases: Mapping[str, Tuple[str, ...]]
    required_fields: Tuple[str, ...]
    column_order: Tuple[str, ...]

    def resolve_field(self, row: Row, field_key: str) -> FieldValue:
        """Resolve a field value for a row, returning empty FieldValue if missing."""

        aliases = self.field_aliases.get(field_key, ())
        for alias in aliases:
            normalized_alias = _normalize_header_key(alias)
            header = row.header_lookup.get(normalized_alias)
            if header is None:
                continue
            raw_value = row.columns.get(header, "")
            if raw_value == "":
                return FieldValue(header=header, raw=None, normalized=None)
            return FieldValue(
                header=header,
                raw=raw_value,
                normalized=_normalize_whitespace(raw_value),
            )
        return FieldValue(header=None, raw=None, normalized=None)


@dataclass(frozen=True)
class NormalizedRow:
    """Normalized representation of a CSV row ready for graph/vector storage."""

    row: Row
    metadata: DocumentMetadata
    row_checksum: str
    field_values: Mapping[str, FieldValue]
    spec_bounds: Mapping[str, SpecBound]
    step_id: str
    char_id: Optional[str]
    spec_id: Optional[str]
    mm_id: Optional[str]
    sample_plan_id: Optional[str]
    reaction_plan_id: Optional[str]
    responsibility_id: Optional[str]

    def get_field(self, key: str) -> FieldValue:
        return self.field_values.get(key, FieldValue(None, None, None))

    def to_cypher_params(self) -> Dict[str, Any]:
        """Return Cypher parameters for MERGE-based upserts."""

        step = self.get_field("step_name")
        if step.raw is None:
            raise IngestionError(
                f"Row {self.row.row_index} is missing Process Step; cannot build Cypher params."
            )

        char = self.get_field("characteristic")
        spec_text = self.get_field("spec_text")
        mm = self.get_field("measurement_method")
        sample_size = self.get_field("sample_size")
        sample_freq = self.get_field("sample_frequency")
        reaction = self.get_field("reaction_plan")
        responsibility = self.get_field("responsibility")

        lower = self.spec_bounds.get("lower_spec")
        target = self.spec_bounds.get("target_spec")
        upper = self.spec_bounds.get("upper_spec")

        params: Dict[str, Any] = {
            "doc_id": self.metadata.doc_id,
            "doc_type": self.metadata.doc_type,
            "filename": self.metadata.filename,
            "batch": self.metadata.import_batch,
            "created_at": self.metadata.created_at,
            "row_index": self.row.row_index,
            "row_checksum": self.row_checksum,
            "step_id": self.step_id,
            "step_name": step.raw,
            "step_name_norm": step.normalized,
            "step_num": self.get_field("step_number").raw,
            "step_desc": self.get_field("step_description").raw,
            "char_id": self.char_id,
            "char_name": char.raw,
            "char_name_norm": char.normalized,
            "spec_id": self.spec_id,
            "spec_text": spec_text.raw,
            "spec_text_norm": spec_text.normalized,
            "lower": lower.raw if lower else None,
            "lower_normalized": lower.normalized if lower else None,
            "lower_num": lower.numeric if lower else None,
            "lower_unit": lower.unit if lower else None,
            "lower_parse_error": lower.parse_error if lower else False,
            "target": target.raw if target else None,
            "target_normalized": target.normalized if target else None,
            "target_num": target.numeric if target else None,
            "target_unit": target.unit if target else None,
            "target_parse_error": target.parse_error if target else False,
            "upper": upper.raw if upper else None,
            "upper_normalized": upper.normalized if upper else None,
            "upper_num": upper.numeric if upper else None,
            "upper_unit": upper.unit if upper else None,
            "upper_parse_error": upper.parse_error if upper else False,
            "mm_id": self.mm_id,
            "mm_name": mm.raw,
            "mm_name_norm": mm.normalized,
            "sample_size": sample_size.raw,
            "sample_size_norm": sample_size.normalized,
            "sample_frequency": sample_freq.raw,
            "sample_frequency_norm": sample_freq.normalized,
            "sp_id": self.sample_plan_id,
            "reaction_text": reaction.raw,
            "reaction_text_norm": reaction.normalized,
            "rp_id": self.reaction_plan_id,
            "resp_text": responsibility.raw,
            "resp_text_norm": responsibility.normalized,
            "resp_id": self.responsibility_id,
        }
        return params

    def to_vector_chunk(self, column_order: Sequence[str]) -> VectorChunk:
        """Return a vector chunk representing this row."""

        lines: List[str] = []
        used_headers: set[str] = set()

        for key in column_order:
            field_value = self.field_values.get(key)
            if not field_value or field_value.raw in (None, ""):
                continue
            label = field_value.header or key
            used_headers.add(label)
            lines.append(f"{label}: {field_value.raw}")

        for header, value in self.row.columns.items():
            if header in used_headers:
                continue
            if value is None or value == "":
                continue
            lines.append(f"{header}: {value}")

        text = "\n".join(lines)
        metadata = {
            "doc_type": self.metadata.doc_type,
            "doc_id": self.metadata.doc_id,
            "filename": self.metadata.filename,
            "import_batch": self.metadata.import_batch,
            "row_index": self.row.row_index,
            "step_id": self.step_id,
            "row_checksum": self.row_checksum,
            "column_map": dict(self.row.columns),
            "chunk_type": "row",
        }
        chunk_id = _stable_hash(self.metadata.doc_id, "row", str(self.row.row_index))
        return VectorChunk(chunk_id=chunk_id, text=text, metadata=metadata)


@dataclass(frozen=True)
class ManufacturingIngestionResult:
    """Final payload from ingesting a manufacturing CSV document."""

    document: DocumentMetadata
    normalized_rows: List[NormalizedRow]
    cypher_parameters: List[Dict[str, Any]]
    vector_chunks: List[VectorChunk]
    skipped_rows: List[int] = field(default_factory=list)
    row_checksums: Dict[int, str] = field(default_factory=dict)


_BASE_FIELD_ALIASES: Dict[str, Tuple[str, ...]] = {
    "step_name": (
        "Process Step",
        "Process Step/Function",
        "Operation",
        "Step",
        "Process Name",
    ),
    "step_number": (
        "Step #",
        "Step Number",
        "Operation #",
        "Process Step Number",
    ),
    "step_description": (
        "Description",
        "Process Description",
        "Step Description",
        "Function Description",
    ),
    "characteristic": (
        "Characteristic",
        "Product Characteristic",
        "Process Characteristic",
        "Characteristic / Requirement",
        "Key Characteristic",
        "Process Input",
    ),
    "spec_text": (
        "Specification",
        "Specification / Tolerance",
        "Spec",
        "Requirements",
    ),
    "lower_spec": (
        "Lower Spec",
        "LSL",
        "Lower Limit",
        "Min",
    ),
    "target_spec": (
        "Target",
        "Nominal",
        "Target Spec",
    ),
    "upper_spec": (
        "Upper Spec",
        "USL",
        "Upper Limit",
        "Max",
    ),
    "measurement_method": (
        "Measurement Method",
        "Measurement Technique",
        "Evaluation Method",
        "Gage/Measurement Equipment",
        "Control Method",
    ),
    "sample_size": (
        "Sample Size",
        "Sample Qty",
        "Sample Size (n)",
        "Sample Amount",
    ),
    "sample_frequency": (
        "Sampling Frequency",
        "Sample Frequency",
        "Control Frequency",
        "Sampling Plan",
    ),
    "reaction_plan": (
        "Reaction Plan",
        "Reaction Plan/Corrective Action",
        "Reaction",
        "Corrective Action",
        "Control Reaction Plan",
    ),
    "responsibility": (
        "Responsibility",
        "Responsible",
        "Who",
        "Responsibility (Name/Title)",
    ),
}


_DOC_TYPE_SCHEMAS: Dict[str, DocumentSchema] = {}


def _register_schema(
    doc_type: str,
    *,
    required_fields: Sequence[str],
    overrides: Optional[Mapping[str, Sequence[str]]] = None,
) -> None:
    field_aliases: Dict[str, Tuple[str, ...]] = {}
    for key, aliases in _BASE_FIELD_ALIASES.items():
        override_values = overrides.get(key) if overrides else None
        if override_values:
            combined = tuple(dict.fromkeys([*override_values, *aliases]))
        else:
            combined = aliases
        field_aliases[key] = combined

    column_order = tuple(_BASE_FIELD_ALIASES.keys())
    _DOC_TYPE_SCHEMAS[doc_type] = DocumentSchema(
        doc_type=doc_type,
        field_aliases=field_aliases,
        required_fields=tuple(required_fields),
        column_order=column_order,
    )


_register_schema(
    "control_plan",
    required_fields=("step_name", "characteristic"),
)
_register_schema(
    "process_flow",
    required_fields=("step_name",),
    overrides={
        "step_description": ("Process Step Description",),
    },
)
_register_schema(
    "fmea",
    required_fields=("step_name",),
    overrides={
        "characteristic": (
            "Process Input",
            "Requirement",
            "Function",
            "Potential Failure Mode",
        )
    },
)
_register_schema(
    "ppap",
    required_fields=("step_name", "characteristic"),
)


NEO4J_CONSTRAINTS: Tuple[str, ...] = (
    "CREATE CONSTRAINT doc_docid IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE",
    "CREATE CONSTRAINT step_stepid IF NOT EXISTS FOR (s:ProcessStep) REQUIRE s.step_id IS UNIQUE",
    "CREATE CONSTRAINT char_charid IF NOT EXISTS FOR (c:Characteristic) REQUIRE c.char_id IS UNIQUE",
    "CREATE CONSTRAINT spec_specid IF NOT EXISTS FOR (sp:Spec) REQUIRE sp.spec_id IS UNIQUE",
    "CREATE CONSTRAINT mm_mmid IF NOT EXISTS FOR (m:MeasurementMethod) REQUIRE m.mm_id IS UNIQUE",
    "CREATE CONSTRAINT sp_spid IF NOT EXISTS FOR (p:SamplePlan) REQUIRE p.sp_id IS UNIQUE",
    "CREATE CONSTRAINT rp_rpid IF NOT EXISTS FOR (r:ReactionPlan) REQUIRE r.rp_id IS UNIQUE",
    "CREATE CONSTRAINT resp_respid IF NOT EXISTS FOR (a:Responsibility) REQUIRE a.resp_id IS UNIQUE",
)


def supported_doc_types() -> Tuple[str, ...]:
    """Return the supported document types."""

    return tuple(sorted(_DOC_TYPE_SCHEMAS.keys()))


def get_document_schema(doc_type: str) -> DocumentSchema:
    """Fetch the document schema for the given document type."""

    normalized = doc_type.strip().casefold()
    schema = _DOC_TYPE_SCHEMAS.get(normalized)
    if not schema:
        raise IngestionError(
            f"Unsupported document type '{doc_type}'. Supported types: {', '.join(supported_doc_types())}"
        )
    return schema


def _parse_numeric_bound(value: Optional[str]) -> SpecBound:
    """Parse a specification bound string, extracting numeric value and unit."""

    if value is None:
        return SpecBound(raw=None, normalized=None, numeric=None, unit=None, parse_error=False)

    cleaned = value.strip()
    if cleaned == "":
        return SpecBound(raw=None, normalized=None, numeric=None, unit=None, parse_error=False)

    normalized = _normalize_whitespace(cleaned)
    match = _NUMERIC_WITH_UNIT.match(cleaned)
    if match:
        number_str, unit = match.groups()
        try:
            numeric = float(number_str)
        except ValueError:
            numeric = None
            parse_error = True
        else:
            parse_error = False
        unit_normalized = _normalize_whitespace(unit) if unit.strip() else None
        return SpecBound(
            raw=cleaned,
            normalized=normalized,
            numeric=numeric,
            unit=unit_normalized,
            parse_error=parse_error,
        )

    return SpecBound(
        raw=cleaned,
        normalized=normalized,
        numeric=None,
        unit=None,
        parse_error=True,
    )


def _compute_doc_id(doc_type: str, filename: str) -> str:
    return _stable_hash(_normalize_for_id(doc_type), _normalize_for_id(filename))


def _compute_spec_id(spec_text: FieldValue, bounds: Mapping[str, SpecBound]) -> Optional[str]:
    parts: List[str] = []
    if spec_text.raw:
        parts.append(_normalize_for_id(spec_text.raw))
    for key in ("lower_spec", "target_spec", "upper_spec"):
        bound = bounds.get(key)
        if not bound:
            continue
        if bound.numeric is not None:
            unit_component = bound.unit or ""
            parts.append(f"{bound.numeric}::{unit_component.casefold() if unit_component else ''}")
        elif bound.normalized:
            parts.append(_normalize_for_id(bound.normalized))
    if not parts:
        return None
    return _stable_hash(*parts)


def _compute_sample_plan_id(sample_size: FieldValue, sample_freq: FieldValue) -> Optional[str]:
    parts: List[str] = []
    if sample_size.raw:
        parts.append(_normalize_for_id(sample_size.raw))
    if sample_freq.raw:
        parts.append(_normalize_for_id(sample_freq.raw))
    if not parts:
        return None
    return _stable_hash(*parts)


def _row_checksum(columns: Mapping[str, str]) -> str:
    serialised = json.dumps(columns, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(serialised.encode("utf-8")).hexdigest()


def _build_header_lookup(fieldnames: Sequence[str]) -> Dict[str, str]:
    return {_normalize_header_key(name): name for name in fieldnames}


def _detect_required_columns(schema: DocumentSchema, header_lookup: Mapping[str, str]) -> None:
    missing: List[str] = []
    for field_key in schema.required_fields:
        aliases = schema.field_aliases.get(field_key, ())
        found = False
        for alias in aliases:
            if _normalize_header_key(alias) in header_lookup:
                found = True
                break
        if not found:
            missing.append(field_key)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise IngestionError(
            f"Missing required columns for document type '{schema.doc_type}': {missing_list}"
        )


def _iter_csv_rows(path: Path) -> Tuple[Sequence[str], List[Mapping[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise IngestionError(f"CSV file '{path}' is missing a header row.")
        rows = list(reader)
    return reader.fieldnames, rows


def _prepare_rows(
    *,
    path: Path,
    doc_type: str,
    schema: DocumentSchema,
) -> Tuple[List[Row], List[str]]:
    fieldnames, raw_rows = _iter_csv_rows(path)
    header_lookup = _build_header_lookup(fieldnames)
    _detect_required_columns(schema, header_lookup)

    rows: List[Row] = []
    for row_index, raw_row in enumerate(raw_rows, start=1):
        columns = {name: (value if value is not None else "") for name, value in raw_row.items()}
        if all((value or "").strip() == "" for value in columns.values()):
            continue
        rows.append(
            Row(
                doc_type=doc_type,
                filename=path.name,
                row_index=row_index,
                columns=columns,
                header_lookup=header_lookup,
            )
        )

    return rows, list(fieldnames)


def _normalize_row(row: Row, schema: DocumentSchema, metadata: DocumentMetadata) -> NormalizedRow:
    field_values: Dict[str, FieldValue] = {}
    for key in schema.column_order:
        field_values[key] = schema.resolve_field(row, key)

    step = field_values["step_name"]
    if step.raw is None:
        raise IngestionError(
            f"Row {row.row_index} in '{row.filename}' is missing a Process Step value."
        )

    step_id = _stable_hash(metadata.doc_id, str(row.row_index))

    characteristic = field_values["characteristic"]
    char_id: Optional[str]
    if characteristic.raw:
        char_id = _stable_hash(_normalize_for_id(characteristic.raw))
    else:
        char_id = None

    spec_bounds = {
        "lower_spec": _parse_numeric_bound(field_values["lower_spec"].raw),
        "target_spec": _parse_numeric_bound(field_values["target_spec"].raw),
        "upper_spec": _parse_numeric_bound(field_values["upper_spec"].raw),
    }

    spec_id = _compute_spec_id(field_values["spec_text"], spec_bounds)

    mm = field_values["measurement_method"]
    mm_id = _stable_hash(_normalize_for_id(mm.raw)) if mm.raw else None

    sample_size = field_values["sample_size"]
    sample_freq = field_values["sample_frequency"]
    sample_plan_id = _compute_sample_plan_id(sample_size, sample_freq)

    reaction = field_values["reaction_plan"]
    reaction_plan_id = _stable_hash(_normalize_for_id(reaction.raw)) if reaction.raw else None

    responsibility = field_values["responsibility"]
    responsibility_id = (
        _stable_hash(_normalize_for_id(responsibility.raw)) if responsibility.raw else None
    )

    checksum = _row_checksum(row.columns)

    return NormalizedRow(
        row=row,
        metadata=metadata,
        row_checksum=checksum,
        field_values=field_values,
        spec_bounds=spec_bounds,
        step_id=step_id,
        char_id=char_id,
        spec_id=spec_id,
        mm_id=mm_id,
        sample_plan_id=sample_plan_id,
        reaction_plan_id=reaction_plan_id,
        responsibility_id=responsibility_id,
    )


def _build_heading_chunk(
    metadata: DocumentMetadata, fieldnames: Sequence[str]
) -> VectorChunk:
    text = (
        f"{metadata.doc_type} columns for {metadata.filename}: "
        + ", ".join(fieldnames)
    )
    chunk_id = _stable_hash(metadata.doc_id, "heading")
    metadata_dict = {
        "doc_type": metadata.doc_type,
        "doc_id": metadata.doc_id,
        "filename": metadata.filename,
        "import_batch": metadata.import_batch,
        "row_index": None,
        "step_id": None,
        "column_map": {name: None for name in fieldnames},
        "chunk_type": "heading",
    }
    return VectorChunk(chunk_id=chunk_id, text=text, metadata=metadata_dict)


def ingest_csv(
    path: str | Path,
    *,
    doc_type: str,
    import_batch: str,
    allow_reimport: bool = False,
    existing_row_checksums: Optional[Mapping[int, str]] = None,
) -> ManufacturingIngestionResult:
    """Ingest a manufacturing CSV file.

    Args:
        path: Filesystem path to the CSV file.
        doc_type: One of ``control_plan``, ``process_flow``, ``fmea``, ``ppap``.
        import_batch: Arbitrary string identifying the import batch.
        allow_reimport: If ``True``, rows with matching checksums may be skipped
            instead of causing duplicate errors.
        existing_row_checksums: Optional mapping from ``row_index`` to checksum
            gathered from a previous import. Used to enforce deterministic
            re-import semantics.

    Returns:
        ManufacturingIngestionResult containing normalized rows, Cypher
        parameters, and vector chunks.
    """

    csv_path = Path(path)
    if not csv_path.exists():
        raise IngestionError(f"CSV file '{csv_path}' does not exist.")

    schema = get_document_schema(doc_type)
    rows, fieldnames = _prepare_rows(path=csv_path, doc_type=schema.doc_type, schema=schema)

    doc_id = _compute_doc_id(schema.doc_type, csv_path.name)
    metadata = DocumentMetadata(
        doc_type=schema.doc_type,
        filename=csv_path.name,
        import_batch=import_batch,
        doc_id=doc_id,
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    normalized_rows: List[NormalizedRow] = []
    cypher_params: List[Dict[str, Any]] = []
    vector_chunks: List[VectorChunk] = []
    skipped_rows: List[int] = []
    row_checksums: Dict[int, str] = {}

    existing_row_checksums = existing_row_checksums or {}

    heading_chunk = _build_heading_chunk(metadata, fieldnames)
    vector_chunks.append(heading_chunk)

    for row in rows:
        normalized_row = _normalize_row(row, schema, metadata)
        row_checksum = normalized_row.row_checksum
        row_checksums[row.row_index] = row_checksum

        existing_checksum = existing_row_checksums.get(row.row_index)
        if existing_checksum is not None:
            if not allow_reimport:
                raise IngestionError(
                    f"Row {row.row_index} in document {doc_id} already exists; "
                    "set allow_reimport=True to ignore identical rows."
                )
            if existing_checksum != row_checksum:
                raise IngestionError(
                    f"Row {row.row_index} in document {doc_id} changed since last import."
                )
            skipped_rows.append(row.row_index)
            continue

        normalized_rows.append(normalized_row)
        cypher_params.append(normalized_row.to_cypher_params())
        vector_chunks.append(normalized_row.to_vector_chunk(schema.column_order))

    return ManufacturingIngestionResult(
        document=metadata,
        normalized_rows=normalized_rows,
        cypher_parameters=cypher_params,
        vector_chunks=vector_chunks,
        skipped_rows=skipped_rows,
        row_checksums=row_checksums,
    )


__all__ = [
    "FieldValue",
    "SpecBound",
    "VectorChunk",
    "DocumentMetadata",
    "DocumentSchema",
    "NormalizedRow",
    "ManufacturingIngestionResult",
    "NEO4J_CONSTRAINTS",
    "IngestionError",
    "ingest_csv",
    "get_document_schema",
    "supported_doc_types",
]

