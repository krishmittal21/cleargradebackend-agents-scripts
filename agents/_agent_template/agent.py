import logging
import os
from typing import Any, Dict, List, Optional
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.tools import StructuredTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

try:
    from langchain.agents import AgentExecutor, create_tool_calling_agent
except ImportError:
    from langchain.agents.agent import AgentExecutor
    from langchain.agents import create_tool_calling_agent

# TODO: Import your API client if needed
# from your_api_client import YourApiClient
from memory import get_chat_history

logger = logging.getLogger(__name__)


class ValidatedInputs:
    """Input validation utilities"""
    
    # TODO: Add your validation methods
    @staticmethod
    def validate_date(date_str: str) -> bool:
        """Validate date format YYYY-MM-DD"""
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
            return True
        except ValueError:
            return False
    
    @staticmethod
    def validate_id(item_id: int) -> bool:
        """Validate that ID is a positive integer"""
        return isinstance(item_id, int) and item_id > 0


def build_tools(client: Optional[Any] = None) -> List[StructuredTool]:
    """Build and return async tools for the agent.
    
    TODO: Replace example tools with your actual tools.
    """
    
    # Example Tool 1: List items
    async def list_items_tool(category: str) -> str:
        """Get list of items for a specific category.
        
        TODO: Implement your actual tool logic.
        """
        if not category or not isinstance(category, str):
            return "Error: Category is required and must be text"
        
        try:
            # TODO: Replace with actual API call
            # result = await client.list_items(category.strip())
            # if not result.get("success"):
            #     return f"Failed to fetch items: {result.get('error')}"
            # return str(result)
            
            return f"Example result for category: {category}"
        except Exception as e:
            logger.error(f"Error in list_items: {e}")
            return f"Error fetching items: {str(e)}"
    
    # Example Tool 2: View item details
    async def view_item_tool(item_id: str) -> str:
        """Get detailed information about a specific item by ID.
        
        TODO: Implement your actual tool logic.
        """
        try:
            iid = int(str(item_id).strip())
        except (ValueError, AttributeError):
            return "Error: Item ID must be a valid number"
        
        if not ValidatedInputs.validate_id(iid):
            return "Error: Item ID must be a positive number"
        
        try:
            # TODO: Replace with actual API call
            # result = await client.view_item(iid)
            # if not result.get("success"):
            #     return f"Item not found: {result.get('error')}"
            # return str(result)
            
            return f"Example details for item ID: {iid}"
        except Exception as e:
            logger.error(f"Error in view_item: {e}")
            return f"Error fetching item details: {str(e)}"
    
    # Example Tool 3: Generate report
    async def generate_report_tool(date_range: str) -> str:
        """Generate report for date range. Input format: 'YYYY-MM-DD to YYYY-MM-DD'.
        
        TODO: Implement your actual tool logic.
        """
        try:
            parts = date_range.split(" to ")
            if len(parts) != 2:
                return "Error: Use format 'YYYY-MM-DD to YYYY-MM-DD'"
            
            from_date = parts[0].strip()
            to_date = parts[1].strip()
            
            if not ValidatedInputs.validate_date(from_date) or \
               not ValidatedInputs.validate_date(to_date):
                return "Error: Dates must be in YYYY-MM-DD format"
            
            # TODO: Replace with actual API call
            # result = await client.generate_report(from_date, to_date)
            # if not result.get("success"):
            #     return f"Failed to generate report: {result.get('error')}"
            # return str(result)
            
            return f"Example report from {from_date} to {to_date}"
        except Exception as e:
            logger.error(f"Error in generate_report: {e}")
            return f"Error generating report: {str(e)}"
    
    # TODO: Add more tools as needed
    
    return [
        StructuredTool.from_function(
            coroutine=list_items_tool,
            name="list_items",
            description="Get list of items for a category"
        ),
        StructuredTool.from_function(
            coroutine=view_item_tool,
            name="view_item",
            description="Get detailed item info by numeric ID"
        ),
        StructuredTool.from_function(
            coroutine=generate_report_tool,
            name="generate_report",
            description="Generate report for date range 'YYYY-MM-DD to YYYY-MM-DD'"
        ),
    ]


def build_llm() -> ChatOpenAI:
    """Build and configure the LLM."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY")
    
    base_url = os.environ.get(
        "OPENROUTER_BASE_URL",
        "https://openrouter.ai/api/v1"
    )
    model = os.environ.get(
        "OPENROUTER_MODEL",
        "google/gemini-2.0-flash-exp:free"
    )
    
    headers = {}
    if os.environ.get("OPENROUTER_SITE_URL"):
        headers["HTTP-Referer"] = os.environ["OPENROUTER_SITE_URL"]
    
    return ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=0,
        max_tokens=2000,
        default_headers=headers,
    )


# TODO: Customize this system prompt for your agent
SYSTEM_PROMPT_TEXT = """You are [AGENT_NAME], an intelligent AI assistant for [PURPOSE].

YOUR GUIDELINES:
1. ALWAYS be accurate with dates, numbers, and information.
2. Use the available tools to fetch real data.
3. Provide clear and helpful responses.
4. Current Date: {current_date}

TODO: Add specific guidelines and personality traits for your agent.
"""


class AgentWrapper:
    """Main agent wrapper class.
    
    TODO: Rename this class to match your agent (e.g., YourAgentWrapper)
    """
    
    def __init__(
        self,
        session_id: str,
        user_id: Optional[str] = None
    ):
        self.session_id = session_id
        self.user_id = user_id
        
        # TODO: Initialize your API client if needed
        # self.client = YourApiClient()
        self.client = None
        
        self.tools = build_tools(self.client)
        self.llm = build_llm()
        
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT_TEXT.format(current_date=current_date)),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        self.agent = create_tool_calling_agent(
            self.llm,
            self.tools,
            self.prompt
        )
        
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            max_iterations=50,
            handle_parsing_errors=True,
        )
    
    async def ainvoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process user input and return agent response."""
        user_input = inputs.get("input", "").strip()
        if not user_input:
            return {"output": "Please provide a question."}
        
        async with get_chat_history(
            self.session_id,
            user_id=self.user_id
        ) as history:
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


def build_agent(
    session_id: str,
    user_id: Optional[str] = None
) -> AgentWrapper:
    """Factory function to create agent instance."""
    return AgentWrapper(session_id, user_id)
