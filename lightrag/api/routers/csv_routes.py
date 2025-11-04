"""CSV export routes for LightRAG (strict structured outputs, Option A)."""

from __future__ import annotations

import csv
import io
import json
import re
from collections.abc import AsyncIterator
from typing import Any, Dict, List, Optional, Sequence, Type, TypeVar

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, ValidationError, ConfigDict, create_model

from ..utils_api import get_combined_auth_dependency
from ...base import QueryParam
from ...utils import logger

router = APIRouter(prefix="/csv", tags=["csv"])

SchemaModelT = TypeVar("SchemaModelT", bound=BaseModel)


# ---------------------------
# Templates (unchanged)
# ---------------------------
TEMPLATES: Dict[str, List[str]] = {
    "fmea": [
        "Process Step",
        "Function / Requirements",
        "Potential Failure Mode",
        "Potential Effects of Failure",
        "Severity (S)",
        "Classification (Special Characteristic)",
        "Potential Causes / Mechanisms",
        "Occurrence (O)",
        "Current Process Controls – Prevention",
        "Current Process Controls – Detection",
        "Detection (D)",
        "Risk Priority Number (RPN)",
        "Recommended Actions",
        "Action Owner",
        "Target Completion Date",
        "Actions Taken & Effective Date",
        "Revised Severity (S)",
        "Revised Occurrence (O)",
        "Revised Detection (D)",
        "Revised RPN",
    ],
    "control_plan": [
        "Process Step",
        "Characteristic",
        "Specification/Tolerance",
        "Measurement Method",
        "Sample Size/Frequency",
        "Reaction Plan",
        "Responsibility",
    ],
    "process_flow": [
        "Step #",
        "Process Step",
        "Input",
        "Output",
        "Equipment",
        "Notes",
    ],
    "ppap": [
        "Part Number",
        "Part Name",
        "Customer",
        "Supplier",
        "Submission Level",
        "Requirement",
        "Status",
        "Comments",
    ],
}


_PLANNING_KEYWORDS: Dict[str, Sequence[str]] = {
    "control_plan": (
        "control plan",
        "process step",
        "measurement method",
        "reaction plan",
        "sample frequency",
    ),
    "process_flow": (
        "process flow",
        "input",
        "output",
        "equipment",
    ),
    "fmea": (
        "FMEA",
        "failure mode",
        "occurrence",
        "severity",
        "detection",
        "RPN",
        "recommended actions",
    ),
    "ppap": (
        "PPAP",
        "submission level",
        "requirement",
        "status",
        "comments",
    ),
}


def _dedupe_preserve(items: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for item in items:
        normalized = item.strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(normalized)
    return ordered


def _truncate_text(value: str, max_length: int = 280) -> str:
    cleaned = value.replace("\r", " ").replace("\n", " ").strip()
    if len(cleaned) <= max_length:
        return cleaned
    return cleaned[: max_length - 1].rstrip() + "…"


def _extract_sections(payload: Optional[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not payload:
        return [], [], []
    data = payload.get("data") or {}
    entities = list(data.get("entities") or [])
    relationships = list(data.get("relationships") or [])
    chunks = list(data.get("chunks") or [])
    return entities, relationships, chunks


def _result_is_empty(payload: Optional[Dict[str, Any]]) -> bool:
    if not payload:
        return True
    status = payload.get("status")
    if status and status.lower() != "success":
        return True
    entities, relationships, chunks = _extract_sections(payload)
    return not entities and not relationships and not chunks


def _format_planning_context(payload: Dict[str, Any], *, limit_entities: int = 12, limit_relationships: int = 30, limit_chunks: int = 4) -> str:
    entities, relationships, chunks = _extract_sections(payload)
    metadata = payload.get("metadata") or {}
    keywords = metadata.get("keywords") or {}

    lines: List[str] = []
    if keywords:
        high_level = ", ".join(keywords.get("high_level") or [])
        low_level = ", ".join(keywords.get("low_level") or [])
        if high_level:
            lines.append(f"High-level keywords: {high_level}")
        if low_level:
            lines.append(f"Low-level keywords: {low_level}")

    if relationships:
        lines.append(
            f"Relationships (sample {min(len(relationships), limit_relationships)} of {len(relationships)}):"
        )
        for rel in relationships[:limit_relationships]:
            src = rel.get("src_id") or rel.get("src") or "?"
            tgt = rel.get("tgt_id") or rel.get("tgt") or "?"
            description = _truncate_text(rel.get("description", ""), 220)
            keywords_text = rel.get("keywords")
            annotation = f" [{_truncate_text(keywords_text, 120)}]" if keywords_text else ""
            lines.append(f"- {src} → {tgt}: {description}{annotation}")

    if entities:
        lines.append(f"Entities (sample {min(len(entities), limit_entities)} of {len(entities)}):")
        for entity in entities[:limit_entities]:
            name = entity.get("entity_name") or entity.get("name") or "?"
            entity_type = entity.get("entity_type") or entity.get("type")
            description = _truncate_text(entity.get("description", ""), 220)
            qualifier = f" ({entity_type})" if entity_type else ""
            lines.append(f"- {name}{qualifier}: {description}")

    if chunks:
        lines.append(f"Representative row snippets (showing {min(len(chunks), limit_chunks)} of {len(chunks)}):")
        for chunk in chunks[:limit_chunks]:
            content = _truncate_text(chunk.get("content", ""), 260)
            source = chunk.get("file_path") or chunk.get("reference_id")
            suffix = f" [{source}]" if source else ""
            lines.append(f"- {content}{suffix}")

    if not lines:
        return "No retrieved planning context."
    return "\n".join(lines)


def _format_row_context(payload: Dict[str, Any], plan_label: str, *, limit_entities: int = 10, limit_relationships: int = 18, limit_chunks: int = 6) -> str:
    entities, relationships, chunks = _extract_sections(payload)
    metadata = payload.get("metadata") or {}
    keywords = metadata.get("keywords") or {}

    lines: List[str] = [f"Plan label: {plan_label}"]
    high_level = ", ".join(keywords.get("high_level") or [])
    low_level = ", ".join(keywords.get("low_level") or [])
    if high_level:
        lines.append(f"High-level keywords: {high_level}")
    if low_level:
        lines.append(f"Low-level keywords: {low_level}")

    if relationships:
        lines.append(
            f"Key relationships (sample {min(len(relationships), limit_relationships)} of {len(relationships)}):"
        )
        for rel in relationships[:limit_relationships]:
            src = rel.get("src_id") or rel.get("src") or "?"
            tgt = rel.get("tgt_id") or rel.get("tgt") or "?"
            description = _truncate_text(rel.get("description", ""), 200)
            lines.append(f"- {src} → {tgt}: {description}")

    if entities:
        lines.append(f"Referenced entities (sample {min(len(entities), limit_entities)} of {len(entities)}):")
        for entity in entities[:limit_entities]:
            name = entity.get("entity_name") or entity.get("name") or "?"
            description = _truncate_text(entity.get("description", ""), 200)
            lines.append(f"- {name}: {description}")

    if chunks:
        lines.append(f"Source excerpts (showing {min(len(chunks), limit_chunks)} of {len(chunks)}):")
        for chunk in chunks[:limit_chunks]:
            content = _truncate_text(chunk.get("content", ""), 320)
            source = chunk.get("file_path") or chunk.get("reference_id")
            suffix = f" [{source}]" if source else ""
            lines.append(f"- {content}{suffix}")

    if not relationships and not entities and chunks:
        lines.append("(Graph context empty; using nearby text chunks only.)")

    return "\n".join(lines)


# ---------------------------
# Request/Response models (strict)
# ---------------------------
class CsvRequest(BaseModel):
    """Request payload for generating CSV exports."""
    model_config = ConfigDict(extra="forbid")

    workspace: Optional[str] = Field(
        default=None,
        description="Optional workspace identifier when running multi-workspace setups.",
    )
    template: str = Field(..., min_length=1, description="The preset template identifier.")
    prompt: Optional[str] = Field(
        default=None,
        description="Optional natural-language instructions describing the desired document.",
    )
    columns: Optional[List[str]] = Field(
        default=None,
        description="Custom column list when using the 'custom' template.",
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional filters to scope exported rows. Reserved for future use.",
    )
    limit: Optional[int] = Field(
        default=1000,
        ge=1,
        description="Maximum number of rows to emit.",
    )


class PlannedRow(BaseModel):
    """Structured description for a row that should be generated."""
    model_config = ConfigDict(extra="forbid")

    label: str = Field(
        ..., description="Short identifier for the row, e.g., a process step or part name."
    )
    description: str = Field(
        ..., description="Detailed natural-language summary of the intended row content."
    )


class RowPlanResponse(BaseModel):
    """Collection of row plans returned from the planning LLM call."""
    model_config = ConfigDict(extra="forbid")

    rows: List[PlannedRow] = Field(default_factory=list)


# ---------------------------
# Helpers for strict dynamic schema
# ---------------------------
def _safe_field_name(alias: string) -> str:  # type: ignore[name-defined]
    """Type-narrowed alias for editors. Replaced below by the real function."""
    ...


def _safe_field_name(alias: str) -> str:
    """
    Convert arbitrary column labels into valid python identifiers,
    preserving the original label via Field(alias=...).
    """
    name = re.sub(r"\W+", "_", alias.strip())
    if not name or name[0].isdigit():
        name = f"f_{name or 'field'}"
    return name.lower()


def _build_row_schema_model(columns: List[str]) -> Type[BaseModel]:
    """
    Build a strict Pydantic model where:
      - document_schema is a strict object with one field per column
      - each field is str with default "" (so the model is easy to satisfy)
      - extra fields are forbidden (=> additionalProperties: false)
    """
    # DocumentSchema with one aliased field per column
    doc_fields: Dict[str, tuple[type, Field]] = {}
    for col in columns:
        safe = _safe_field_name(col)
        # store as string; allow empty string
        doc_fields[safe] = (str, Field("", alias=col))

    DocumentSchema = create_model("DocumentSchema", **doc_fields)
    DocumentSchema.model_config = ConfigDict(extra="forbid")

    RowSchemaDynamic = create_model(
        "RowSchemaDynamic",
        document_schema=(DocumentSchema, ...),
    )
    RowSchemaDynamic.model_config = ConfigDict(extra="forbid")
    return RowSchemaDynamic


# ---------------------------
# Route factory
# ---------------------------
def create_csv_routes(_rag, api_key: Optional[str] = None):  # pragma: no cover - wiring only
    """Attach CSV generation endpoints to the FastAPI app."""
    combined_auth = get_combined_auth_dependency(api_key)

    async def _retrieve_planning_payload(req: CsvRequest, columns: List[str]) -> Dict[str, Any]:
        planning_keywords = _dedupe_preserve(
            list(_PLANNING_KEYWORDS.get(req.template, ())) + columns
        )
        query_segments = [
            f"Template: {req.template}",
            f"Columns: {', '.join(columns)}",
        ]
        if req.prompt:
            query_segments.append(f"User instructions: {req.prompt.strip()}")
        planning_query = " | ".join(query_segments)

        planning_param = QueryParam(
            mode="global",
            top_k=60,
            chunk_top_k=0,
            max_relation_tokens=1500,
            max_entity_tokens=800,
            max_total_tokens=2200,
            hl_keywords=planning_keywords,
            ll_keywords=planning_keywords,
        )

        payload = await _rag.aquery_data(planning_query, param=planning_param)
        if _result_is_empty(payload):
            logger.info(
                "[csv_routes] Planning retrieval empty for template '%s', retrying in naive mode.",
                req.template,
            )
            fallback_param = QueryParam(
                mode="naive",
                top_k=12,
                chunk_top_k=12,
                max_total_tokens=2200,
                max_entity_tokens=800,
                max_relation_tokens=1500,
                ll_keywords=planning_keywords,
            )
            payload = await _rag.aquery_data(planning_query, param=fallback_param)
            if _result_is_empty(payload):
                raise HTTPException(
                    status_code=502,
                    detail="Unable to retrieve planning context for CSV generation.",
                )
        return payload

    async def _retrieve_row_payload(
        req: CsvRequest,
        columns: List[str],
        plan: PlannedRow,
    ) -> Dict[str, Any]:
        template_keywords = list(_PLANNING_KEYWORDS.get(req.template, ()))
        row_keywords = _dedupe_preserve(
            [plan.label, req.template] + columns + template_keywords
        )
        row_query_parts = [
            f"Template: {req.template}",
            f"Target row: {plan.label}",
            f"Columns: {', '.join(columns)}",
            f"Plan description: {plan.description}",
        ]
        if req.prompt:
            row_query_parts.append(f"Global instruction: {req.prompt.strip()}")
        row_query = " | ".join(row_query_parts)

        max_total_tokens = 2600 if req.template == "fmea" else 2200
        row_param = QueryParam(
            mode="mix",
            top_k=12,
            chunk_top_k=10,
            max_entity_tokens=900,
            max_relation_tokens=900,
            max_total_tokens=max_total_tokens,
            hl_keywords=row_keywords,
            ll_keywords=row_keywords,
        )

        payload = await _rag.aquery_data(row_query, param=row_param)
        entities, relationships, chunks = _extract_sections(payload)

        if not entities and not relationships and not chunks:
            logger.info(
                "[csv_routes] Row retrieval empty for '%s', retrying in naive mode.",
                plan.label,
            )
            fallback_param = QueryParam(
                mode="naive",
                top_k=12,
                chunk_top_k=12,
                max_total_tokens=max_total_tokens,
                max_entity_tokens=900,
                max_relation_tokens=900,
                ll_keywords=row_keywords,
            )
            payload = await _rag.aquery_data(row_query, param=fallback_param)
            entities, relationships, chunks = _extract_sections(payload)
            if not entities and not relationships and not chunks:
                raise HTTPException(
                    status_code=502,
                    detail=f"Unable to retrieve context for row '{plan.label}'.",
                )

        if not entities and not relationships and chunks:
            logger.warning(
                "[csv_routes] Row '%s' has chunk-only context after retrieval.",
                plan.label,
            )

        return payload

    @router.get("/templates", dependencies=[Depends(combined_auth)])
    async def list_templates() -> Dict[str, Any]:
        """Return the available CSV templates."""
        templates = [{"id": key, "columns": value} for key, value in TEMPLATES.items()]
        templates.append({"id": "custom", "columns": []})
        return {"templates": templates}

    async def _call_llm(
        prompt: str,
        *,
        system_prompt: str,
        schema_model: Type[SchemaModelT],
    ) -> SchemaModelT:
        """Execute an LLM completion with structured response enforcement."""
        if not hasattr(_rag, "llm_model_func") or _rag.llm_model_func is None:
            raise HTTPException(status_code=500, detail="LLM model function is not configured")

        response = await _rag.llm_model_func(  # type: ignore[func-returns-value]
            prompt,
            system_prompt=system_prompt,
            response_format=schema_model,
        )

        if response is None:
            raise HTTPException(status_code=502, detail="LLM returned no data")

        # Handle OpenAI-style parsed completions
        if hasattr(response, "choices"):
            choices = getattr(response, "choices", [])
            if not choices:
                raise HTTPException(status_code=502, detail="LLM returned no choices")
            message = getattr(choices[0], "message", None)
            if message is None:
                raise HTTPException(status_code=502, detail="LLM response missing message")
            parsed = getattr(message, "parsed", None)
            if parsed is None:
                content = getattr(message, "content", None)
                if content is None:
                    raise HTTPException(status_code=502, detail="LLM response missing parsed content")
                try:
                    return schema_model.model_validate_json(content)
                except ValidationError as exc:
                    raise HTTPException(
                        status_code=502,
                        detail=f"Structured response validation failed: {exc.errors()}",
                    ) from exc
            if isinstance(parsed, schema_model):
                return parsed
            try:
                return schema_model.model_validate(parsed)
            except ValidationError as exc:
                raise HTTPException(
                    status_code=502,
                    detail=f"Structured response validation failed: {exc.errors()}",
                ) from exc

        if isinstance(response, schema_model):
            return response

        if isinstance(response, BaseModel):
            try:
                return schema_model.model_validate(response.model_dump())
            except ValidationError as exc:
                raise HTTPException(
                    status_code=502,
                    detail=f"Structured response validation failed: {exc.errors()}",
                ) from exc

        if isinstance(response, str):
            try:
                return schema_model.model_validate_json(response)
            except ValidationError as exc:
                raise HTTPException(
                    status_code=502,
                    detail=f"Structured response validation failed: {exc.errors()}",
                ) from exc

        if isinstance(response, dict):
            try:
                return schema_model.model_validate(response)
            except ValidationError as exc:
                raise HTTPException(
                    status_code=502,
                    detail=f"Structured response validation failed: {exc.errors()}",
                ) from exc

        raise HTTPException(status_code=502, detail="Unexpected structured LLM response type")

    async def _plan_rows(req: CsvRequest, columns: List[str]) -> List[PlannedRow]:
        """Ask the LLM to outline each row that should be generated."""
        target_rows = max(1, min(req.limit or 1, 20))
        user_instruction = (req.prompt or "").strip()

        planning_payload = await _retrieve_planning_payload(req, columns)
        planning_context = _format_planning_context(planning_payload)

        planning_system_prompt = (
            "You are an expert manufacturing documentation planner. "
            "Return strictly valid JSON for the provided schema."
        )

        planning_prompt = (
            "Plan the rows that should appear in a CSV export.\n"
            f"Template identifier: {req.template}.\n"
            f"Columns: {json.dumps(columns, ensure_ascii=False)}.\n"
            f"Maximum rows to plan: {target_rows}.\n"
            "If the user provided instructions, incorporate them.\n"
            "Instructions from user: "
            f"{user_instruction or 'None provided.'}\n"
            "Use the retrieved manufacturing context below to anchor the plan.\n"
            "Retrieved context:\n"
            f"{planning_context}\n"
            "Provide between 1 and the requested maximum number of rows."
        )

        plan = await _call_llm(
            planning_prompt,
            system_prompt=planning_system_prompt,
            schema_model=RowPlanResponse,
        )

        if not plan.rows:
            raise HTTPException(status_code=502, detail="LLM did not return any row plans")

        return plan.rows[:target_rows]

    async def _materialize_row(
        req: CsvRequest,
        columns: List[str],
        plan: PlannedRow,
    ) -> Dict[str, Any]:
        """Generate a concrete row following the provided schema description."""
        row_payload = await _retrieve_row_payload(req, columns, plan)
        row_context = _format_row_context(row_payload, plan.label)

        completion_system_prompt = (
            "You create detailed manufacturing documents. "
            "Only respond with JSON that matches the required schema."
        )

        completion_prompt = (
            "Generate a single row for a CSV export.\n"
            f"Template identifier: {req.template}.\n"
            f"Columns (schema keys): {json.dumps(columns, ensure_ascii=False)}.\n"
            "Row plan label: "
            f"{plan.label}.\n"
            "Row plan description: "
            f"{plan.description}.\n"
            "Ground your response in the retrieved manufacturing context.\n"
            "Retrieved context:\n"
            f"{row_context}\n"
            "Fill every column with a contextually appropriate value using your domain knowledge.\n"
            "If a value is not applicable, return an empty string.\n"
            "Respond with JSON shaped exactly like:\n"
            "{\n  \"document_schema\": {\n    \"Column Name\": \"value\"\n  }\n}.\n"
            "Use the column names EXACTLY as provided (including spaces, punctuation, and case). "
            "Do not add, remove, or rename keys."
        )

        # Build strict row schema for *this* set of columns
        RowSchemaDynamic = _build_row_schema_model(columns)

        # Call LLM with strict schema
        row_model = await _call_llm(
            completion_prompt,
            system_prompt=completion_system_prompt,
            schema_model=RowSchemaDynamic,
        )

        # RowSchemaDynamic.document_schema is a Pydantic model with aliases per column.
        doc = row_model.document_schema
        data_by_alias = doc.model_dump(by_alias=True)  # exact column labels

        # Normalize into simple dict[str, str]
        normalized: Dict[str, Any] = {}
        for column in columns:
            val = data_by_alias.get(column, "")
            if val is None:
                normalized[column] = ""
            elif isinstance(val, (dict, list)):
                normalized[column] = json.dumps(val, ensure_ascii=False)
            else:
                normalized[column] = str(val)
        return normalized

    async def _csv_stream(req: CsvRequest, columns: List[str]) -> AsyncIterator[str]:
        """Stream CSV content row by row as it is generated."""
        plans = await _plan_rows(req, columns)

        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        yield buffer.getvalue()
        buffer.seek(0)
        buffer.truncate(0)

        for plan in plans:
            row = await _materialize_row(req, columns, plan)
            writer.writerow(row)
            yield buffer.getvalue()
            buffer.seek(0)
            buffer.truncate(0)

    @router.post("/generate", dependencies=[Depends(combined_auth)])
    async def generate_csv(req: CsvRequest):
        """Stream a CSV export for the requested template."""
        if req.template != "custom":
            columns = TEMPLATES.get(req.template)
            if not columns:
                raise HTTPException(status_code=400, detail="Unknown template")
        else:
            if not req.columns:
                raise HTTPException(
                    status_code=400,
                    detail="Custom template requires 'columns'",
                )
            columns = req.columns

        stream = _csv_stream(req, columns)
        return StreamingResponse(
            stream,
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{req.template}.csv"'},
        )

    return router


__all__ = ["create_csv_routes", "router"]
