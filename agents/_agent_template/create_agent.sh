#!/bin/bash

# Agent Template Quick Start Script
# Usage: ./create_agent.sh your_agent_name "Agent Description"

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if agent name is provided
if [ -z "$1" ]; then
    echo -e "${RED}Error: Agent name is required${NC}"
    echo "Usage: ./create_agent.sh your_agent_name \"Agent Description\""
    exit 1
fi

AGENT_NAME=$1
AGENT_DESC=${2:-"AI Agent"}
TEMPLATE_DIR="backend/agents/_agent_template"
TARGET_DIR="backend/agents/$AGENT_NAME"

# Check if template exists
if [ ! -d "$TEMPLATE_DIR" ]; then
    echo -e "${RED}Error: Template directory not found at $TEMPLATE_DIR${NC}"
    exit 1
fi

# Check if target directory already exists
if [ -d "$TARGET_DIR" ]; then
    echo -e "${RED}Error: Agent '$AGENT_NAME' already exists at $TARGET_DIR${NC}"
    exit 1
fi

echo -e "${GREEN}Creating new agent: $AGENT_NAME${NC}"
echo "Description: $AGENT_DESC"
echo ""

# Copy template
echo -e "${YELLOW}Copying template...${NC}"
cp -r "$TEMPLATE_DIR" "$TARGET_DIR"

# Remove the gitignore from template (will use repo's gitignore)
rm -f "$TARGET_DIR/.gitignore"

# Create .env from .env.example
echo -e "${YELLOW}Creating .env file...${NC}"
cp "$TARGET_DIR/.env.example" "$TARGET_DIR/.env"

echo -e "${GREEN}✓ Agent '$AGENT_NAME' created successfully!${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. cd $TARGET_DIR"
echo "2. Edit .env with your API keys and configuration"
echo "3. Add your GCP service account JSON file"
echo "4. Customize agent.py with your tools and logic"
echo "5. Update main.py with your API title and description"
echo "6. Install dependencies: pip install -r requirements.txt"
echo "7. Test locally: python test_agent.py"
echo "8. Run server: uvicorn main:app --reload --port 8000"
echo ""
echo -e "${GREEN}See README.md for detailed customization instructions${NC}"
