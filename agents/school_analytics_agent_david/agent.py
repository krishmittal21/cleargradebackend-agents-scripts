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

from tiaf_api_client import TIAFApiClient
from memory import get_chat_history

from forecasting_tools import (
    monte_carlo_simulation_tool,
    arima_forecast_tool,
    fee_collection_forecast_tool
)

logger = logging.getLogger(__name__)


class ValidatedInputs:
    """Input validation utilities"""

    @staticmethod
    def validate_date(date_str: str) -> bool:
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
            return True
        except ValueError:
            return False

    @staticmethod
    def validate_class_name(class_name: str) -> bool:
        valid_classes = [
            "Nursery", "KG", "I", "II", "III", "IV", "V", "VI", "VII",
            "VIII", "IX COM", "IX SCI", "X COM", "X SCI", "XI COM",
            "XI SCI", "XII COM", "XII SCI"
        ]
        return class_name.strip() in valid_classes

    @staticmethod
    def validate_student_id(student_id: int) -> bool:
        return isinstance(student_id, int) and student_id > 0


def build_tools(client: TIAFApiClient) -> List[StructuredTool]:
    """Build and return async tools for the agent."""

    async def student_list_tool(class_name: str) -> str:
        """Get list of students for a specific class (e.g., 'X SCI')."""
        if not class_name or not isinstance(class_name, str):
            return "Error: Class name is required and must be text"

        if not ValidatedInputs.validate_class_name(class_name):
            return f"Error: Invalid class '{class_name}'."

        try:
            result = await client.student_list(class_name.strip())
            if not result.get("success"):
                return f"Failed to fetch students: {result.get('error')}"
            return str(result)
        except Exception as e:
            logger.error(f"Error in student_list: {e}")
            return f"Error fetching students: {str(e)}"

    async def student_view_tool(student_id: str) -> str:
        """Get detailed information about a specific student by numeric ID."""
        try:
            sid = int(str(student_id).strip())
        except (ValueError, AttributeError):
            return "Error: Student ID must be a valid number"

        if not ValidatedInputs.validate_student_id(sid):
            return "Error: Student ID must be a positive number"

        try:
            result = await client.student_view(sid)
            if not result.get("success"):
                return f"Student not found: {result.get('error')}"
            return str(result)
        except Exception as e:
            logger.error(f"Error in student_view: {e}")
            return f"Error fetching student details: {str(e)}"

    async def fee_report_tool(date_range: str) -> str:
        """Get fee collection report. Input format: 'YYYY-MM-DD to YYYY-MM-DD'."""
        try:
            parts = date_range.split(" to ")
            if len(parts) != 2:
                return "Error: Use format 'YYYY-MM-DD to YYYY-MM-DD'"

            from_date = parts[0].strip()
            to_date = parts[1].strip()

            if not ValidatedInputs.validate_date(from_date) or \
               not ValidatedInputs.validate_date(to_date):
                return "Error: Dates must be in YYYY-MM-DD format"

            result = await client.fee_report(from_date, to_date)
            if not result.get("success"):
                return f"Failed to generate report: {result.get('error')}"
            return str(result)
        except Exception as e:
            logger.error(f"Error in fee_report: {e}")
            return f"Error generating fee report: {str(e)}"

    async def expense_report_tool(date_range: str) -> str:
        """Get expense report. Input format: 'YYYY-MM-DD to YYYY-MM-DD'."""
        try:
            parts = date_range.split(" to ")
            if len(parts) != 2:
                return "Error: Use format 'YYYY-MM-DD to YYYY-MM-DD'"

            from_date = parts[0].strip()
            to_date = parts[1].strip()

            if not ValidatedInputs.validate_date(from_date) or \
               not ValidatedInputs.validate_date(to_date):
                return "Error: Dates must be in YYYY-MM-DD format"

            result = await client.expense_report(from_date, to_date)
            if not result.get("success"):
                return f"Failed to generate report: {result.get('error')}"
            return str(result)
        except Exception as e:
            logger.error(f"Error in expense_report: {e}")
            return f"Error generating expense report: {str(e)}"

    return [
        StructuredTool.from_function(
            coroutine=student_list_tool,
            name="student_list",
            description="Get list of students for a class"
        ),
        StructuredTool.from_function(
            coroutine=student_view_tool,
            name="student_view",
            description="Get detailed student info by numeric ID"
        ),
        StructuredTool.from_function(
            coroutine=fee_report_tool,
            name="fee_report",
            description="Get fee report for date range 'YYYY-MM-DD to YYYY-MM-DD'"
        ),
        StructuredTool.from_function(
            coroutine=expense_report_tool,
            name="expense_report",
            description="Get expense report for date range 'YYYY-MM-DD to YYYY-MM-DD'"
        ),
        StructuredTool.from_function(
            coroutine=monte_carlo_simulation_tool,
            name="monte_carlo_simulation",
            description="Perform Monte Carlo simulation for risk analysis and forecasting. Input: JSON with initial_value, expected_return, volatility, time_horizon, num_simulations"
        ),
        StructuredTool.from_function(
            coroutine=arima_forecast_tool,
            name="arima_forecast",
            description="Perform ARIMA time series forecasting. Input: JSON with time_series_data (list), forecast_periods, auto_detect, seasonal"
        ),
        StructuredTool.from_function(
            coroutine=fee_collection_forecast_tool,
            name="fee_collection_forecast",
            description="Comprehensive fee collection forecasting with ARIMA and Monte Carlo. Input: JSON with historical_fees (list), forecast_months, include_monte_carlo"
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


SYSTEM_PROMPT_TEXT = """You are David, an intelligent School Business & Analytics Assistant.

YOUR GUIDELINES:
1. ALWAYS be accurate with dates, numbers, and student information.
2. Use the available tools to fetch real data.
3. For financial queries, provide context.
4. Current Date: {current_date}
"""


class SchoolAgentWrapper:
    def __init__(
        self,
        session_id: str,
        user_id: Optional[str] = None
    ):
        self.session_id = session_id
        self.user_id = user_id
        self.client = TIAFApiClient()
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

    async def astream(self, inputs: Dict[str, Any]):
        """Stream agent responses in real-time."""
        user_input = inputs.get("input", "").strip()
        if not user_input:
            yield {"type": "error", "content": "Please provide a question."}
            return

        async with get_chat_history(
            self.session_id,
            user_id=self.user_id
        ) as history:
            try:
                await history.aadd_user_message(user_input)

                full_output = ""
                async for event in self.agent_executor.astream_events(
                    {
                        "input": user_input,
                        "chat_history": history.messages
                    },
                    version="v1"
                ):
                    kind = event.get("event")
                    
                    # Stream agent thinking/tool usage
                    if kind == "on_chat_model_stream":
                        chunk = event.get("data", {}).get("chunk")
                        if chunk and hasattr(chunk, "content"):
                            content = chunk.content
                            if content:
                                full_output += content
                                yield {"type": "token", "content": content}
                    
                    # Stream tool calls
                    elif kind == "on_tool_start":
                        tool_name = event.get("name", "")
                        yield {"type": "tool_start", "tool": tool_name}
                    
                    elif kind == "on_tool_end":
                        tool_name = event.get("name", "")
                        yield {"type": "tool_end", "tool": tool_name}

                # Save the complete response to history
                if full_output:
                    await history.aadd_ai_message(full_output)
                    yield {"type": "done", "content": full_output}
                else:
                    # Fallback if no streaming content
                    yield {"type": "done", "content": "Unable to process request"}

            except Exception as e:
                logger.error(f"Agent streaming error: {e}", exc_info=True)
                yield {"type": "error", "content": "I encountered an internal error."}


def build_agent(
    session_id: str,
    user_id: Optional[str] = None
) -> SchoolAgentWrapper:
    return SchoolAgentWrapper(session_id, user_id)