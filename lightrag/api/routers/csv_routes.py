"""CSV export routes for LightRAG."""

from __future__ import annotations

import csv
import io
import json
from collections.abc import AsyncIterator
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, ValidationError

from ..utils_api import get_combined_auth_dependency

router = APIRouter(prefix="/csv", tags=["csv"])


class CsvRequest(BaseModel):
    """Request payload for generating CSV exports."""

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

    label: str = Field(
        ..., description="Short identifier for the row, e.g., a process step or part name."
    )
    description: str = Field(
        ..., description="Detailed natural-language summary of the intended row content."
    )


class RowPlanResponse(BaseModel):
    """Collection of row plans returned from the planning LLM call."""

    rows: List[PlannedRow] = Field(default_factory=list)


class RowSchema(BaseModel):
    """Structured row output matching the template schema."""

    document_schema: Dict[str, Any] = Field(
        ..., description="Mapping of template columns to populated values."
    )


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
    "Revised RPN"
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


def create_csv_routes(_rag, api_key: Optional[str] = None):  # pragma: no cover - simple wiring
    """Attach CSV generation endpoints to the FastAPI app."""

    combined_auth = get_combined_auth_dependency(api_key)

    @router.get("/templates", dependencies=[Depends(combined_auth)])
    async def list_templates() -> Dict[str, Any]:
        """Return the available CSV templates."""

        templates = [
            {"id": key, "columns": value}
            for key, value in TEMPLATES.items()
        ]
        templates.append({"id": "custom", "columns": []})
        return {"templates": templates}

    async def _call_llm(prompt: str, *, system_prompt: str) -> str:
        """Execute an LLM completion ensuring a plain string response."""

        if not hasattr(_rag, "llm_model_func") or _rag.llm_model_func is None:
            raise HTTPException(status_code=500, detail="LLM model function is not configured")

        response = await _rag.llm_model_func(  # type: ignore[func-returns-value]
            prompt,
            system_prompt=system_prompt,
            stream=False,
        )

        if isinstance(response, str):
            return response

        if hasattr(response, "__aiter__"):
            chunks: List[str] = []
            async for chunk in response:  # type: ignore[assignment]
                chunks.append(str(chunk))
            return "".join(chunks)

        raise HTTPException(status_code=502, detail="Unexpected LLM response type")

    async def _plan_rows(req: CsvRequest, columns: List[str]) -> List[PlannedRow]:
        """Ask the LLM to outline each row that should be generated."""

        target_rows = max(1, min(req.limit or 1, 20))
        user_instruction = req.prompt.strip() if req.prompt else ""
        planning_system_prompt = (
            "You are an expert manufacturing documentation planner. "
            "Always respond with valid JSON that conforms to the provided schema."
        )

        planning_prompt = (
            "Plan the rows that should appear in a CSV export.\n"
            f"Template identifier: {req.template}.\n"
            f"Columns: {json.dumps(columns, ensure_ascii=False)}.\n"
            f"Maximum rows to plan: {target_rows}.\n"
            "If the user provided instructions, incorporate them."
            "\nInstructions from user: "
            f"{user_instruction or 'None provided.'}\n"
            "Return JSON with this structure:\n"
            "{\n  \"rows\": [\n    {\n      \"label\": \"short identifier\",\n"
            "      \"description\": \"detailed description of what the row should contain\"\n"
            "    }\n  ]\n}.\n"
            "Provide between 1 and the requested maximum number of rows."
        )

        raw = await _call_llm(planning_prompt, system_prompt=planning_system_prompt)

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Failed to parse planning response as JSON: {exc}",
            ) from exc

        try:
            plan = RowPlanResponse.model_validate(parsed)
        except ValidationError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Planning response did not match expected schema: {exc.errors()}",
            ) from exc

        if not plan.rows:
            raise HTTPException(status_code=502, detail="LLM did not return any row plans")

        return plan.rows[:target_rows]

    async def _materialize_row(
        req: CsvRequest,
        columns: List[str],
        plan: PlannedRow,
    ) -> Dict[str, Any]:
        """Generate a concrete row following the provided schema description."""

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
            "Fill every column with a contextually appropriate value using your domain knowledge."
            "\nIf a value is not applicable, return an empty string."
            "\nRespond with JSON shaped exactly like:\n"
            "{\n  \"document_schema\": {\n    \"Column Name\": \"value\"\n  }\n}."
        )

        raw = await _call_llm(completion_prompt, system_prompt=completion_system_prompt)

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Failed to parse row generation response as JSON: {exc}",
            ) from exc

        try:
            row = RowSchema.model_validate(parsed)
        except ValidationError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Row generation response did not match schema: {exc.errors()}",
            ) from exc

        normalized: Dict[str, Any] = {}
        for column in columns:
            value = row.document_schema.get(column, "")
            if value is None:
                normalized[column] = ""
            elif isinstance(value, (dict, list)):
                normalized[column] = json.dumps(value, ensure_ascii=False)
            else:
                normalized[column] = str(value)

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
            headers={
                "Content-Disposition": f'attachment; filename="{req.template}.csv"'
            },
        )

    return router


__all__ = ["create_csv_routes", "router"]
