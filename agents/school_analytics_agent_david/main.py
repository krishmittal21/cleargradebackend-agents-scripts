import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any

from agent import build_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="School Business Agent",
    description="AI-powered school management and analytics",
    version="2.0",
)


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Session ID")
    user_id: str = Field(..., description="User ID")
    message: str = Field(..., min_length=1, description="User message")


class ChatResponse(BaseModel):
    success: bool
    output: str
    session_id: str
    user_id: str


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """Process chat request."""
    try:
        agent = build_agent(req.session_id, req.user_id)
        response = await agent.ainvoke({"input": req.message})

        return ChatResponse(
            success=response.get("success", True),
            output=response.get("output", "No response"),
            session_id=req.session_id,
            user_id=req.user_id,
        )
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail="Chat failed")


@app.get("/sessions/{session_id}/history")
async def get_history(
    session_id: str,
    user_id: str
) -> Dict[str, Any]:
    """Get conversation history."""
    try:
        from memory import get_chat_history

        async with get_chat_history(session_id, user_id=user_id) as history:
            messages = [
                {
                    "type": m.__class__.__name__,
                    "content": m.content,
                }
                for m in history.messages
            ]
            return {
                "session_id": session_id,
                "user_id": user_id,
                "count": len(messages),
                "messages": messages,
            }
    except Exception as e:
        logger.error(f"History error: {e}")
        raise HTTPException(status_code=500, detail="History failed")


@app.post("/sessions/{session_id}/clear")
async def clear_session(
    session_id: str,
    user_id: str
) -> Dict[str, str]:
    """Clear session history."""
    try:
        from memory import get_chat_history

        async with get_chat_history(session_id, user_id=user_id) as history:
            await history.aclear()
            return {
                "status": "cleared",
                "session_id": session_id,
                "user_id": user_id
            }
    except Exception as e:
        logger.error(f"Clear error: {e}")
        raise HTTPException(status_code=500, detail="Clear failed")


@app.get("/health")
async def health() -> Dict[str, str]:
    """Health check."""
    return {"status": "healthy"}