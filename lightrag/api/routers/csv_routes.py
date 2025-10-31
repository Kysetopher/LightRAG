"""CSV export routes for LightRAG."""

from __future__ import annotations

import csv
import io
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

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


TEMPLATES: Dict[str, List[str]] = {
    "fmea": [
        "Process Step",
        "Function",
        "Potential Failure Mode",
        "Potential Effects",
        "S",
        "Potential Causes",
        "O",
        "Current Controls",
        "D",
        "RPN",
        "Action Owner",
        "Target Date",
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

    def _fetch_rows(req: CsvRequest) -> List[Dict[str, Any]]:
        """Map the requested template to row dictionaries.

        This placeholder implementation returns synthetic data so the
        end-to-end flow works before a knowledge-graph mapping is wired in.
        Replace with queries against ``rag`` once the data model is ready.
        """

        rows: List[Dict[str, Any]] = []
        raw_prompt = req.prompt.strip() if req.prompt and req.prompt.strip() else None
        prompt_note = (
            f"Generated based on instructions: {raw_prompt}"
            if raw_prompt
            else None
        )

        if req.template == "fmea":
            for index in range(1, 4):
                rows.append(
                    {
                        "Process Step": f"Op {index}",
                        "Function": "Weld",
                        "Potential Failure Mode": "Porosity",
                        "Potential Effects": "Leak",
                        "S": 8,
                        "Potential Causes": "Contamination",
                        "O": 5,
                        "Current Controls": "Visual, Helium Test",
                        "D": 6,
                        "RPN": 8 * 5 * 6,
                        "Action Owner": "Chris" if not raw_prompt else f"Chris — {raw_prompt}",
                        "Target Date": "2025-11-15",
                    }
                )
        elif req.template == "control_plan":
            rows.append(
                {
                    "Process Step": "Press-Fit",
                    "Characteristic": "Diameter",
                    "Specification/Tolerance": "Ø10.00 ±0.05 mm",
                    "Measurement Method": "Go/No-Go",
                    "Sample Size/Frequency": "1/Hour",
                    "Reaction Plan": (
                        "Stop line if fail"
                        if not raw_prompt
                        else f"Stop line if fail — {raw_prompt}"
                    ),
                    "Responsibility": "Operator",
                }
            )
        elif req.template == "process_flow":
            rows = [
                {
                    "Step #": 10,
                    "Process Step": "Cut",
                    "Input": "Bar",
                    "Output": "Blank",
                    "Equipment": "Saw",
                    "Notes": prompt_note or "",
                },
                {
                    "Step #": 20,
                    "Process Step": "Weld",
                    "Input": "Blank",
                    "Output": "Assembly",
                    "Equipment": "MIG",
                    "Notes": "Jig A",
                },
            ]
        elif req.template == "ppap":
            rows.append(
                {
                    "Part Number": "PN-001",
                    "Part Name": "Bracket",
                    "Customer": "ACME OEM",
                    "Supplier": "LightRAG Plant 1",
                    "Submission Level": "3",
                    "Requirement": "Dimensional Report",
                    "Status": "Submitted",
                    "Comments": (
                        "Awaiting approval"
                        if not raw_prompt
                        else f"Awaiting approval — {raw_prompt}"
                    ),
                }
            )
        else:
            columns = req.columns or []
            rows = [{column: "" for column in columns}]
            if prompt_note and columns:
                rows[0][columns[0]] = prompt_note

        return rows

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

        rows = _fetch_rows(req)

        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()

        max_rows = req.limit or 1000
        for row in rows[:max_rows]:
            writer.writerow(row)
        buffer.seek(0)

        return StreamingResponse(
            iter([buffer.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="{req.template}.csv"'
            },
        )

    return router


__all__ = ["create_csv_routes", "router"]
