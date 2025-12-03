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


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """Stream chat responses in real-time."""
    from fastapi.responses import StreamingResponse
    import json
    
    async def event_generator():
        try:
            agent = build_agent(req.session_id, req.user_id)
            async for event in agent.astream({"input": req.message}):
                # Format as Server-Sent Event
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            logger.error(f"Stream error: {e}")
            error_event = {"type": "error", "content": "Stream failed"}
            yield f"data: {json.dumps(error_event)}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@app.get("/sessions/{session_id}/history")
async def get_history(
    session_id: str,
    user_id: str
) -> Dict[str, Any]:
    """Get conversation history."""
    try:
        from memory import get_chat_history

        async with get_chat_history(session_id, user_id=user_id) as history:
            # Also fetch graph components and raw message data directly from the document
            # since they are not part of the standard message history
            graph_components = []
            raw_messages = []
            if history._db:
                doc = await history._doc_ref().get()
                if doc.exists:
                    data = doc.to_dict() or {}
                    graph_components = data.get("graphComponents", [])
                    raw_messages = data.get("messages", [])

            # Use raw messages if available to preserve timestamps
            if raw_messages:
                messages = [
                    {
                        "type": "HumanMessage" if msg.get("role") == "human" else "AIMessage",
                        "content": msg.get("content", ""),
                        "timestamp": msg.get("timestamp"),
                    }
                    for msg in raw_messages
                ]
            else:
                # Fallback to history.messages if raw messages not available
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
                "graphComponents": graph_components,
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