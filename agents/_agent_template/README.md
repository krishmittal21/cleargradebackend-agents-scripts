# [AGENT_NAME] Agent Template

> **TODO**: Update this README with your agent's specific information

## Overview

[AGENT_NAME] is an AI-powered assistant that helps with [PURPOSE/DESCRIPTION].

## Features

- 🤖 Powered by LangChain and OpenRouter
- 💬 Persistent conversation history with Firestore
- 🔧 Customizable tools and capabilities
- 🚀 FastAPI-based REST API
- 🐳 Docker containerized

## Setup

### Prerequisites

- Python 3.11+
- Docker (optional)
- GCP Project with Firestore enabled
- OpenRouter API key

### Installation

1. **Copy this template directory**:
   ```bash
   cp -r backend/agents/_agent_template backend/agents/your_agent_name
   cd backend/agents/your_agent_name
   ```

2. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your actual values
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Add your GCP service account key**:
   - Download your service account JSON from GCP
   - Place it in this directory
   - Update `GOOGLE_APPLICATION_CREDENTIALS` in `.env`

### Customization Steps

Follow these steps to create your custom agent:

#### 1. Update `agent.py`

- [ ] Rename `AgentWrapper` class to match your agent (e.g., `YourAgentWrapper`)
- [ ] Update `SYSTEM_PROMPT_TEXT` with your agent's personality and guidelines
- [ ] Implement your validation methods in `ValidatedInputs`
- [ ] Create your tools in `build_tools()` function
- [ ] If needed, import and initialize your API client
- [ ] Remove example tools and add your actual tools

#### 2. Update `main.py`

- [ ] Update FastAPI `title` and `description`
- [ ] Add any custom endpoints if needed

#### 3. Create API Client (if needed)

- [ ] Rename `api_client_template.py` to your client name (e.g., `your_api_client.py`)
- [ ] Update class name and methods to match your API
- [ ] Update API authentication and configuration
- [ ] Implement your API methods

#### 4. Update Configuration Files

- [ ] Update `requirements.txt` with any additional dependencies
- [ ] Update `.env.example` with your specific environment variables
- [ ] Update this README with your agent's documentation

#### 5. Update Test Script

- [ ] Update `test_agent.py` with relevant test queries
- [ ] Test your agent locally

## Usage

### Local Development

```bash
# Run the server
uvicorn main:app --reload --port 8000
```

### Test the Agent

```bash
# Run the test script
python test_agent.py
```

### Docker

```bash
# Build the image
docker build -t your-agent-name .

# Run the container
docker run -p 8000:8000 --env-file .env your-agent-name
```

## API Endpoints

### POST `/chat`

Send a message to the agent.

**Request**:
```json
{
  "session_id": "unique-session-id",
  "user_id": "user-id",
  "message": "Your question here"
}
```

**Response**:
```json
{
  "success": true,
  "output": "Agent response",
  "session_id": "unique-session-id",
  "user_id": "user-id"
}
```

### GET `/sessions/{session_id}/history`

Get conversation history for a session.

**Query Parameters**:
- `user_id`: User ID

### POST `/sessions/{session_id}/clear`

Clear conversation history for a session.

**Query Parameters**:
- `user_id`: User ID

### GET `/health`

Health check endpoint.

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENROUTER_API_KEY` | OpenRouter API key | Yes |
| `OPENROUTER_BASE_URL` | OpenRouter base URL | No |
| `OPENROUTER_MODEL` | Model to use | No |
| `FIRESTORE_PROJECT_ID` | GCP project ID | Yes |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to service account key | Yes |
| `PORT` | Server port | No |

**TODO**: Add your custom environment variables here

## Tools

**TODO**: Document your agent's tools and capabilities

### Example Tool 1: list_items

Description of what this tool does.

**Input**: Category name

**Output**: List of items

### Example Tool 2: view_item

Description of what this tool does.

**Input**: Item ID

**Output**: Item details

## Development

### Adding New Tools

1. Define the async tool function in `agent.py`
2. Add input validation
3. Register the tool in the `build_tools()` function
4. Update this README with tool documentation

### Testing

```bash
# Run the test script
python test_agent.py
```

## Deployment

**TODO**: Add deployment instructions specific to your infrastructure

### Cloud Run Example

```bash
# Build and push to GCR
docker build -t gcr.io/your-project/your-agent .
docker push gcr.io/your-project/your-agent

# Deploy to Cloud Run
gcloud run deploy your-agent \
  --image gcr.io/your-project/your-agent \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Firestore Connection**: Check credentials and project ID
3. **OpenRouter API**: Verify API key is set correctly
4. **Tool Errors**: Check tool descriptions and input schemas

### Debug Mode

Enable verbose logging:
```python
logging.basicConfig(level=logging.DEBUG)
```

## License

**TODO**: Add license information

## Support

**TODO**: Add support contact information

---

**Version**: 1.0  
**Last Updated**: 2024-12-01  
**Based On**: Agent Template v1.0
