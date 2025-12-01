# Agent Template - Quick Reference

## 📁 Template Location

`/Users/krishmittal/Developer/cleargrade/backend/agents/_agent_template`

## 🚀 Quick Start

### Option 1: Use the creation script (Recommended)

```bash
cd backend/agents/_agent_template
./create_agent.sh my_new_agent "My Agent Description"
```

### Option 2: Manual copy

```bash
cp -r backend/agents/_agent_template backend/agents/my_new_agent
cd backend/agents/my_new_agent
cp .env.example .env
# Edit files as needed
```

## 📦 Template Contents

| File | Purpose | Action Required |
|------|---------|----------------|
| `agent.py` | Core agent logic and tools | ⚠️ **CUSTOMIZE** - Replace example tools |
| `main.py` | FastAPI server | ⚠️ Update title/description |
| `memory.py` | Chat history (shared) | ✅ Use as-is |
| `api_client_template.py` | External API client | ⚠️ Rename and customize |
| `requirements.txt` | Python dependencies | ⚠️ Add custom dependencies |
| `dockerfile` | Container config | ✅ Use as-is |
| `.dockerignore` | Build exclusions | ✅ Use as-is |
| `.env.example` | Environment template | ⚠️ Copy to `.env` and fill |
| `test_agent.py` | Testing script | ⚠️ Update test queries |
| `README.md` | Documentation | ⚠️ Update with agent details |
| `.gitignore` | Git exclusions | ✅ Use as-is |
| `create_agent.sh` | Creation script | ✅ Helper tool |

Legend: ✅ = Use as-is, ⚠️ = Requires customization

## ✏️ Customization Checklist

### 1. Initial Setup
- [ ] Copy template to new directory
- [ ] Copy `.env.example` to `.env`
- [ ] Add GCP service account JSON
- [ ] Fill in environment variables

### 2. Core Files
- [ ] **agent.py**
  - [ ] Rename `AgentWrapper` class
  - [ ] Update `SYSTEM_PROMPT_TEXT`
  - [ ] Add custom validation in `ValidatedInputs`
  - [ ] Replace example tools with real tools
  - [ ] Import and initialize API client (if needed)
- [ ] **main.py**
  - [ ] Update FastAPI `title` and `description`
  - [ ] Add custom endpoints (if needed)
- [ ] **api_client_template.py**
  - [ ] Rename file to match your API
  - [ ] Update class name
  - [ ] Implement API methods
  - [ ] Configure authentication

### 3. Configuration
- [ ] **requirements.txt** - Add dependencies
- [ ] **.env** - Configure all variables
- [ ] **README.md** - Document your agent

### 4. Testing
- [ ] Update `test_agent.py` with relevant queries
- [ ] Run local tests
- [ ] Test API endpoints

## 🔑 Key Patterns

### Tool Definition Pattern
```python
async def your_tool_name(param: str) -> str:
    """Tool description for LLM."""
    # Validation
    if not param:
        return "Error: Parameter required"
    
    try:
        # API call or logic
        result = await client.your_method(param)
        if not result.get("success"):
            return f"Error: {result.get('error')}"
        return str(result)
    except Exception as e:
        logger.error(f"Error: {e}")
        return f"Error: {str(e)}"

# Register in build_tools()
return [
    StructuredTool.from_function(
        coroutine=your_tool_name,
        name="tool_name",
        description="What this tool does"
    ),
]
```

### Date Range Pattern
```python
async def report_tool(date_range: str) -> str:
    """Input: 'YYYY-MM-DD to YYYY-MM-DD'"""
    parts = date_range.split(" to ")
    if len(parts) != 2:
        return "Error: Use format 'YYYY-MM-DD to YYYY-MM-DD'"
    
    from_date, to_date = parts[0].strip(), parts[1].strip()
    # Validate and process
```

### List/View Pattern
```python
# List items
async def list_items(category: str) -> str:
    result = await client.list(category)
    return str(result)

# View single item
async def view_item(item_id: int) -> str:
    result = await client.get(item_id)
    return str(result)
```

## 🔧 Environment Variables

Required for all agents:
```bash
# OpenRouter (LLM)
OPENROUTER_API_KEY=YOUR_OPENROUTER_API_KEY...
OPENROUTER_MODEL=google/gemini-2.0-flash-exp:free

# Firestore (Memory)
FIRESTORE_PROJECT_ID=your-project
GOOGLE_APPLICATION_CREDENTIALS=./credentials.json

# App Config
PORT=8000
```

Add your custom variables in `.env`

## 🧪 Testing

```bash
# Local testing
python test_agent.py

# Run server
uvicorn main:app --reload --port 8000

# Test API
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test",
    "user_id": "user1",
    "message": "Hello"
  }'
```

## 🐳 Docker

```bash
# Build
docker build -t my-agent .

# Run
docker run -p 8000:8000 --env-file .env my-agent
```

## 📚 Reference

- **Full Documentation**: See `AGENT_TEMPLATE.md`
- **Working Example**: `backend/agents/school_analytics_agent_david`
- **API Client Example**: `school_analytics_agent_david/tiaf_api_client.py`

## ⚠️ Common Mistakes

1. ❌ Forgetting to update `SYSTEM_PROMPT_TEXT`
2. ❌ Not removing example tools
3. ❌ Missing `.env` configuration
4. ❌ Not updating FastAPI title/description
5. ❌ Hardcoding API keys instead of using env vars
6. ❌ Not testing before deployment

## 💡 Tips

- Start with the working `school_analytics_agent_david` example
- Test each tool individually before integration
- Use meaningful tool names and descriptions (LLM sees these)
- Add comprehensive error handling
- Log errors for debugging
- Validate all user inputs
- Keep tools focused and single-purpose

---

**Version**: 1.0  
**Created**: 2024-12-01  
**Location**: `/backend/agents/_agent_template`
