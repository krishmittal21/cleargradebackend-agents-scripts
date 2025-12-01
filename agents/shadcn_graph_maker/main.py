import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any

from agent import build_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Shadcn Graph Maker",
    description="Generates ChartInput JSON from analytics text",
    version="1.0",
)


class GenerateRequest(BaseModel):
    session_id: str = Field(..., description="Session ID")
    user_id: str = Field(..., description="User ID")
    message: str = Field(..., min_length=1, description="Analytics text to convert")


class GenerateResponse(BaseModel):
    success: bool
    output: str
    session_id: str
    user_id: str
    validation: str | None = None
    json_outputs: list[Dict[str, Any]] | None = None
    primary_json: Dict[str, Any] | None = None
    validations: list[str] | None = None


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> GenerateResponse:
    try:
        agent = build_agent(req.session_id, req.user_id)
        response = await agent.ainvoke({"input": req.message})

        return GenerateResponse(
            success=response.get("success", True),
            output=response.get("output", "{}"),
            session_id=req.session_id,
            user_id=req.user_id,
            validation=response.get("validation"),
            json_outputs=response.get("json_outputs"),
            primary_json=response.get("primary_json"),
            validations=response.get("validations"),
        )
    except Exception as e:
        logger.error(f"Generate error: {e}")
        raise HTTPException(status_code=500, detail="Generation failed")


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "healthy"}
