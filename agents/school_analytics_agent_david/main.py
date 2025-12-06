import logging
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query
from pydantic import BaseModel, Field
from typing import Dict, Any

from agent import build_agent
from voice_handler import websocket_voice_handler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="School Business Agent",
    description="AI-powered school management and analytics with voice support",
    version="2.1",
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


class FeedbackRequest(BaseModel):
    rating: str = Field(..., description="Feedback rating: 'good' or 'bad'")
    reason: str | None = Field(None, description="Optional reason for negative feedback")


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


@app.post("/sessions/{session_id}/messages/{message_index}/feedback")
async def submit_feedback(
    session_id: str,
    message_index: int,
    feedback: FeedbackRequest,
    user_id: str
) -> Dict[str, str]:
    """Submit feedback for a specific message."""
    try:
        from memory import get_chat_history

        if feedback.rating not in ["good", "bad"]:
            raise HTTPException(status_code=400, detail="Invalid rating")

        async with get_chat_history(session_id, user_id=user_id) as history:
            await history.add_feedback(
                message_index=message_index,
                rating=feedback.rating,
                reason=feedback.reason
            )
            return {
                "status": "feedback_submitted",
                "session_id": session_id,
                "message_index": str(message_index),
                "rating": feedback.rating
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Feedback error: {e}")
        raise HTTPException(status_code=500, detail="Feedback submission failed")


@app.get("/health")
async def health() -> Dict[str, str]:
    """Health check."""
    return {"status": "healthy"}


@app.websocket("/voice")
async def voice_endpoint(
    websocket: WebSocket,
    session_id: str = Query(..., description="Session ID"),
    user_id: str = Query(..., description="User ID")
):
    """
    Real-time voice conversation endpoint.
    
    Connect via WebSocket with query params: ?session_id=xxx&user_id=yyy
    
    Protocol:
    - Send: {"type": "audio", "data": "<base64 PCM audio>"}
    - Send: {"type": "end_audio"} when done speaking
    - Receive: {"type": "status", "state": "transcribing|thinking|speaking|idle"}
    - Receive: {"type": "transcript", "text": "<user's speech>"}
    - Receive: {"type": "response", "text": "<David's response>"}
    - Receive: {"type": "audio", "data": "<base64 PCM audio>", "sampleRate": 24000, "isLast": bool}
    """
    await websocket.accept()
    logger.info(f"Voice connection opened: session={session_id}, user={user_id}")
    
    try:
        await websocket_voice_handler(
            websocket=websocket,
            session_id=session_id,
            user_id=user_id,
            agent_builder=build_agent
        )
    except WebSocketDisconnect:
        logger.info(f"Voice connection closed: session={session_id}")
    except Exception as e:
        logger.error(f"Voice endpoint error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except:
            pass