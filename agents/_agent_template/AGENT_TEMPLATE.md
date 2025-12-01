# Agent Template

This is a template for creating new AI agents in the ClearGrade system. Use the `school_analytics_agent_david` as a reference implementation.

## 🏗️ Agent Structure

Each agent should be a self-contained directory with the following files:

```
agent_name/
├── agent.py                 # Core agent logic and tool building
├── main.py                  # FastAPI server and endpoints
├── tools.py                 # Tool definitions (optional)
├── memory.py                # Chat history management (shared)
├── {api_client}.py          # External API client (if needed)
├── requirements.txt         # Python dependencies
├── dockerfile               # Container configuration
├── .dockerignore           # Docker build exclusions
├── .env                    # Environment variables (not in git)
└── {agent_name}.json       # GCP service account key (not in git)
```

## 📋 Step-by-Step Guide

### 1. Create Agent Directory

```bash
mkdir backend/agents/your_agent_name
cd backend/agents/your_agent_name
```

### 2. Core Files to Create

#### **agent.py** - Agent Logic

This is the main agent implementation. It should contain:

- **Input Validation Classes**: Define validation utilities for your domain
- **Tool Building Function**: `build_tools(client)` returns list of LangChain tools
- **LLM Builder**: `build_llm()` configures the language model
- **System Prompt**: Define agent personality and capabilities
- **Agent Wrapper Class**: Wraps agent executor with session management
- **Build Agent Function**: Factory function to create agent instances

**Key Components:**

```python
# 1. Input Validation
class ValidatedInputs:
    @staticmethod
    def validate_your_input(input_data: str) -> bool:
        # Add validation logic
        pass

# 2. Build Tools
def build_tools(client: YourApiClient) -> List[StructuredTool]:
    # Define async tool functions
    async def your_tool(param: str) -> str:
        # Tool implementation
        pass
    
    return [
        StructuredTool.from_function(
            coroutine=your_tool,
            name="tool_name",
            description="Tool description"
        ),
    ]

# 3. Build LLM
def build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        model=os.environ.get("OPENROUTER_MODEL", "google/gemini-2.0-flash-exp:free"),
        temperature=0,
        max_tokens=2000,
    )

# 4. System Prompt
SYSTEM_PROMPT_TEXT = """You are [Agent Name], [description].

YOUR GUIDELINES:
1. [Guideline 1]
2. [Guideline 2]
3. Current Date: {current_date}
"""

# 5. Agent Wrapper
class YourAgentWrapper:
    def __init__(self, session_id: str, user_id: Optional[str] = None):
        self.session_id = session_id
        self.user_id = user_id
        self.client = YourApiClient()
        self.tools = build_tools(self.client)
        self.llm = build_llm()
        
        # Build prompt and agent
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT_TEXT.format(current_date=datetime.now().strftime("%Y-%m-%d"))),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        self.agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            max_iterations=50,
            handle_parsing_errors=True,
        )
    
    async def ainvoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        user_input = inputs.get("input", "").strip()
        if not user_input:
            return {"output": "Please provide a question."}
        
        async with get_chat_history(self.session_id, user_id=self.user_id) as history:
            try:
                await history.aadd_user_message(user_input)
                result = await self.agent_executor.ainvoke({
                    "input": user_input,
                    "chat_history": history.messages
                })
                output = result.get("output", "Unable to process request")
                await history.aadd_ai_message(output)
                return {"output": output, "success": True}
            except Exception as e:
                logger.error(f"Agent error: {e}", exc_info=True)
                return {
                    "output": "I encountered an internal error.",
                    "success": False,
                    "error": str(e)
                }

# 6. Factory Function
def build_agent(session_id: str, user_id: Optional[str] = None) -> YourAgentWrapper:
    return YourAgentWrapper(session_id, user_id)
```

#### **main.py** - FastAPI Server

Standard FastAPI server with these endpoints:

```python
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any
from agent import build_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Your Agent Name",
    description="Agent description",
    version="1.0",
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
async def get_history(session_id: str, user_id: str) -> Dict[str, Any]:
    """Get conversation history."""
    try:
        from memory import get_chat_history
        async with get_chat_history(session_id, user_id=user_id) as history:
            messages = [
                {"type": m.__class__.__name__, "content": m.content}
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
async def clear_session(session_id: str, user_id: str) -> Dict[str, str]:
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
```

#### **memory.py** - Chat History

Copy the `memory.py` file from the David agent or use it as a shared module. It provides:
- Async Firestore-backed chat history
- Message persistence with TTL
- Automatic cleanup of old messages
- Context manager for easy usage

#### **{api_client}.py** - External API Client (Optional)

If your agent needs to call external APIs, create a client:

```python
import httpx
from typing import Any, Dict

class YourApiClient:
    def __init__(self, timeout: int = 30):
        self.base_url = "https://your-api.com/api"
        self.api_key = os.environ.get("YOUR_API_KEY")
        self.timeout = timeout
    
    async def _request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                resp = await client.get(f"{self.base_url}/{endpoint}", params=params)
                resp.raise_for_status()
                return {'success': True, 'data': resp.json()}
            except httpx.HTTPStatusError as e:
                return {'success': False, 'error': f'HTTP Error: {e.response.status_code}'}
            except httpx.RequestError as e:
                return {'success': False, 'error': f'Connection Error: {str(e)}'}
    
    async def your_method(self, param: str) -> Dict[str, Any]:
        return await self._request('endpoint', {'param': param})
```

#### **requirements.txt** - Dependencies

```txt
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
langchain==0.3.7
langchain-community==0.3.7
langchain-core==0.3.19
langchain-openai==0.2.8
google-cloud-firestore>=2.19.0
httpx>=0.26.0
requests>=2.31.0
pydantic>=2.5.0
python-dotenv>=1.0.0
# Add any additional dependencies your agent needs
```

#### **dockerfile** - Container Configuration

```dockerfile
# Multi-stage build for better caching and smaller image
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Final stage
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PORT=8000

WORKDIR /app

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

# Copy application code
COPY --chown=appuser:appuser . .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:${PORT}/health', timeout=5)"

# Expose port
EXPOSE ${PORT}

# Run application
CMD exec uvicorn main:app \
    --host 0.0.0.0 \
    --port ${PORT} \
    --workers 4 \
    --access-log
```

#### **.dockerignore** - Build Exclusions

```
__pycache__
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.venv
.git
.gitignore
.dockerignore
.env
.env.local
*.md
.pytest_cache/
.coverage
htmlcov/
dist/
build/
*.egg-info/
.DS_Store
node_modules/
.idea/
.vscode/
*.log
.mypy_cache/
.ruff_cache/
```

#### **.env** - Environment Variables

```bash
# OpenRouter Configuration
OPENROUTER_API_KEY=your_api_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=google/gemini-2.0-flash-exp:free
OPENROUTER_SITE_URL=https://your-site.com

# Firestore Configuration
FIRESTORE_PROJECT_ID=your-project-id
GCP_PROJECT_ID=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=./your-service-account.json

# API Keys (if needed)
YOUR_API_KEY=your_external_api_key

# Application Configuration
PORT=8000
LOG_LEVEL=INFO
```

## 🔧 Additional Optional Files

### **tools.py** - Separate Tool Definitions

If you have many tools or want to organize them separately:

```python
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class YourInputSchema(BaseModel):
    param: str = Field(description="Parameter description")

@tool("tool_name", args_schema=YourInputSchema)
async def your_tool(param: str) -> str:
    """Tool description."""
    # Implementation
    return "result"

def get_tools():
    return [your_tool]
```

### **test_agent.py** - Testing

```python
import asyncio
from agent import build_agent

async def test_agent():
    agent = build_agent(session_id="test-session", user_id="test-user")
    result = await agent.ainvoke({"input": "Your test query"})
    print(result)

if __name__ == "__main__":
    asyncio.run(test_agent())
```

## 🚀 Deployment

### Local Development

1. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your values
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the server:
   ```bash
   uvicorn main:app --reload --port 8000
   ```

### Docker Deployment

1. Build the image:
   ```bash
   docker build -t your-agent-name .
   ```

2. Run the container:
   ```bash
   docker run -p 8000:8000 --env-file .env your-agent-name
   ```

### GCP Cloud Run Deployment

See the `deploy.sh` script in the David agent for reference.

## 📝 Best Practices

1. **Input Validation**: Always validate inputs before processing
2. **Error Handling**: Use try-catch blocks and return meaningful error messages
3. **Logging**: Log errors and important events
4. **Async/Await**: Use async functions for all I/O operations
5. **Type Hints**: Use type hints for better code readability
6. **Documentation**: Document your tools and functions clearly
7. **Environment Variables**: Never hardcode secrets, use environment variables
8. **Testing**: Test your agent thoroughly before deployment

## 🎯 Common Patterns

### Pattern 1: Date Range Tools

```python
async def report_tool(date_range: str) -> str:
    """Get report. Input format: 'YYYY-MM-DD to YYYY-MM-DD'."""
    try:
        parts = date_range.split(" to ")
        if len(parts) != 2:
            return "Error: Use format 'YYYY-MM-DD to YYYY-MM-DD'"
        
        from_date = parts[0].strip()
        to_date = parts[1].strip()
        
        # Validate dates
        # Fetch report
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return f"Error generating report: {str(e)}"
```

### Pattern 2: List/View Pattern

```python
async def list_items_tool(category: str) -> str:
    """Get list of items in a category."""
    result = await client.list_items(category)
    if not result.get("success"):
        return f"Failed to fetch items: {result.get('error')}"
    return str(result)

async def view_item_tool(item_id: int) -> str:
    """Get detailed information about an item."""
    result = await client.view_item(item_id)
    if not result.get("success"):
        return f"Item not found: {result.get('error')}"
    return str(result)
```

## 📚 Reference Implementation

See `backend/agents/school_analytics_agent_david` for a complete working example.

## 🔗 Integration

### API Endpoints

All agents expose these standard endpoints:

- `POST /chat` - Send a message to the agent
- `GET /sessions/{session_id}/history` - Get conversation history
- `POST /sessions/{session_id}/clear` - Clear conversation history
- `GET /health` - Health check

### Request/Response Format

**Chat Request:**
```json
{
  "session_id": "unique-session-id",
  "user_id": "user-id",
  "message": "Your question here"
}
```

**Chat Response:**
```json
{
  "success": true,
  "output": "Agent response",
  "session_id": "unique-session-id",
  "user_id": "user-id"
}
```

## ⚠️ Important Notes

1. **Memory Management**: The chat history has a default TTL of 30 days and max 500 messages
2. **Concurrent Requests**: The agent uses async/await for handling concurrent requests
3. **Model Selection**: Default model is `google/gemini-2.0-flash-exp:free`, but can be configured via env vars
4. **Security**: Always use `.gitignore` to exclude `.env` and service account JSON files
5. **Firestore**: Ensure Firestore is properly configured with the service account

## 🎨 Customization Checklist

When creating a new agent, customize these elements:

- [ ] Agent name and description
- [ ] System prompt and personality
- [ ] Tool definitions and names
- [ ] Input validation logic
- [ ] API client (if needed)
- [ ] Environment variables
- [ ] Dependencies in requirements.txt
- [ ] FastAPI title and description
- [ ] Docker image name
- [ ] Health check configuration
- [ ] Testing scenarios

## 🆘 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Firestore Connection**: Check credentials and project ID
3. **OpenRouter API**: Verify API key is set correctly
4. **Tool Errors**: Check tool descriptions and input schemas
5. **Memory Issues**: Monitor Firestore usage and message counts

### Debug Mode

Enable verbose logging:
```python
logging.basicConfig(level=logging.DEBUG)
```

---

**Version**: 1.0  
**Last Updated**: 2024-12-01  
**Based On**: school_analytics_agent_david  
**Maintainer**: ClearGrade Engineering Team
