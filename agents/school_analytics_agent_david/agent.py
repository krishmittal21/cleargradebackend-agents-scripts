import logging
import os
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import StructuredTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, ToolMessage

try:
    from langchain.agents import AgentExecutor, create_tool_calling_agent
except ImportError:
    from langchain.agents.agent import AgentExecutor
    from langchain.agents import create_tool_calling_agent

from tiaf_api_client import TIAFApiClient
from memory import get_chat_history
from forecasting import FinancialForecaster

logger = logging.getLogger(__name__)

# Summarizer LLM for condensing large tool responses
_summarizer_llm = None

def get_summarizer_llm():
    """Get or create the summarizer LLM (lazily initialized)."""
    global _summarizer_llm
    if _summarizer_llm is None:
        _summarizer_llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0,
            max_tokens=8000
        )
    return _summarizer_llm


async def summarize_tool_response(
    raw_response: str,
    tool_name: str,
    max_length: int = 8000
) -> str:
    """
    Summarize large tool responses to prevent context window overflow.
    
    Preserves all key numbers, totals, and patterns while reducing verbosity.
    Only summarizes if response exceeds max_length.
    """
    if len(raw_response) < max_length:
        return raw_response
    
    try:
        summarizer = get_summarizer_llm()
        
        prompt = f"""You are a data summarizer for a school business analytics system.
Summarize this {tool_name} data concisely while PRESERVING:
- All monetary totals and key figures
- Important patterns and trends
- Critical dates and time periods
- Any warnings or errors

Keep the format clean and structured. Use tables where appropriate.
Do NOT add analysis or recommendations - just present the key data.

DATA TO SUMMARIZE:
{raw_response}
"""
        
        result = await summarizer.ainvoke(prompt)
        summary = result.content if hasattr(result, 'content') else str(result)
        
        return f"[Data summarized from {len(raw_response):,} chars]\n\n{summary}"
        
    except Exception as e:
        logger.warning(f"Summarization failed for {tool_name}: {e}")
        # Fallback: truncate with notice
        return f"[Data truncated due to size - {len(raw_response):,} chars]\n\n{raw_response[:max_length]}..."


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
    forecaster = FinancialForecaster()

    async def revenue_forecast_tool(forecast_months: str) -> str:
        """
        Forecast expected revenue for the coming months.
        Input: Number of months to forecast (1-12), e.g., '6' for 6 months.
        Returns monthly revenue projections with total expected revenue.
        """
        try:
            months = int(forecast_months.strip())
            if not (1 <= months <= 12):
                return "Error: Please specify 1 to 12 months for the forecast."
            
            result = await forecaster.generate_business_forecast(
                metric="revenue",
                forecast_months=months
            )
            
            if not result.get("success"):
                return f"Unable to generate forecast: {result.get('error', 'Unknown error')}"
            
            # Format as clear text for the LLM
            output = []
            output.append(f"**Revenue Forecast for Next {months} Months**\n")
            output.append(f"Total Expected Revenue: ₹{result['summary']['total_projected']:,.2f}")
            output.append(f"Monthly Average: ₹{result['summary']['monthly_average']:,.2f}")
            output.append(f"Trend: {result['summary']['trend'].capitalize()}")
            output.append(f"Confidence: {result['summary']['confidence'].capitalize()}\n")
            
            output.append("**Monthly Breakdown:**")
            for proj in result["monthly_projections"]:
                output.append(f"- {proj['month']}: ₹{proj['projected_amount']:,.2f}")
            
            output.append(f"\n**Historical Context:**")
            ctx = result["historical_context"]
            output.append(f"- Based on {ctx['months_analyzed']} months of data")
            output.append(f"- Historical monthly average: ₹{ctx['historical_monthly_average']:,.2f}")
            output.append(f"- Analysis period: {ctx['period']}")
            
            if result.get("notes"):
                output.append(f"\n**Note:** {result['notes']}")
            
            return "\n".join(output)
            
        except ValueError:
            return "Error: Please provide a valid number of months (1-12)."
        except Exception as e:
            logger.error(f"Error in revenue_forecast_tool: {e}", exc_info=True)
            return f"Error generating revenue forecast: {str(e)}"

    async def expense_forecast_tool(forecast_months: str) -> str:
        """
        Forecast expected expenses for the coming months.
        Input: Number of months to forecast (1-12), e.g., '6' for 6 months.
        Returns monthly expense projections with total expected expenses.
        """
        try:
            months = int(forecast_months.strip())
            if not (1 <= months <= 12):
                return "Error: Please specify 1 to 12 months for the forecast."
            
            result = await forecaster.generate_business_forecast(
                metric="expense",
                forecast_months=months
            )
            
            if not result.get("success"):
                return f"Unable to generate forecast: {result.get('error', 'Unknown error')}"
            
            # Format as clear text for the LLM
            output = []
            output.append(f"**Expense Forecast for Next {months} Months**\n")
            output.append(f"Total Expected Expenses: ₹{result['summary']['total_projected']:,.2f}")
            output.append(f"Monthly Average: ₹{result['summary']['monthly_average']:,.2f}")
            output.append(f"Trend: {result['summary']['trend'].capitalize()}")
            output.append(f"Confidence: {result['summary']['confidence'].capitalize()}\n")
            
            output.append("**Monthly Breakdown:**")
            for proj in result["monthly_projections"]:
                output.append(f"- {proj['month']}: ₹{proj['projected_amount']:,.2f}")
            
            output.append(f"\n**Historical Context:**")
            ctx = result["historical_context"]
            output.append(f"- Based on {ctx['months_analyzed']} months of data")
            output.append(f"- Historical monthly average: ₹{ctx['historical_monthly_average']:,.2f}")
            output.append(f"- Analysis period: {ctx['period']}")
            
            if result.get("notes"):
                output.append(f"\n**Note:** {result['notes']}")
            
            return "\n".join(output)
            
        except ValueError:
            return "Error: Please provide a valid number of months (1-12)."
        except Exception as e:
            logger.error(f"Error in expense_forecast_tool: {e}", exc_info=True)
            return f"Error generating expense forecast: {str(e)}"

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
            raw_response = str(result)
            return await summarize_tool_response(raw_response, "student_list")
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
            raw_response = str(result)
            return await summarize_tool_response(raw_response, "fee_report")
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
            raw_response = str(result)
            return await summarize_tool_response(raw_response, "expense_report")
        except Exception as e:
            logger.error(f"Error in expense_report: {e}")
            return f"Error generating expense report: {str(e)}"

    VALID_CLASSES = [
        "Nursery", "KG", "I", "II", "III", "IV", "V", "VI", "VII",
        "VIII", "IX COM", "IX SCI", "X COM", "X SCI", "XI COM",
        "XI SCI", "XII COM", "XII SCI"
    ]
    valid_classes_str = ", ".join(VALID_CLASSES)

    return [
        StructuredTool.from_function(
            coroutine=revenue_forecast_tool,
            name="revenue_forecast",
            description=(
                "Forecast expected revenue for the coming months. "
                "Input: number of months (1-12). "
                "Example: '6' for a 6-month forecast."
            ),
        ),
        StructuredTool.from_function(
            coroutine=expense_forecast_tool,
            name="expense_forecast",
            description=(
                "Forecast expected expenses for the coming months. "
                "Input: number of months (1-12). "
                "Example: '6' for a 6-month forecast."
            ),
        ),
        StructuredTool.from_function(
            coroutine=student_list_tool,
            name="student_list",
            description=f"Get list of students for a class. Valid classes: {valid_classes_str}"
        ),
        StructuredTool.from_function(
            coroutine=student_view_tool,
            name="student_view",
            description="Get detailed student info by numeric ID"
        ),
        StructuredTool.from_function(
            coroutine=fee_report_tool,
            name="fee_report",
            description="Get fee report for date range 'YYYY-MM-DD to YYYY-MM-DD It is only available from 2025-01-01'"
        ),
        StructuredTool.from_function(
            coroutine=expense_report_tool,
            name="expense_report",
            description="Get expense report for date range 'YYYY-MM-DD to YYYY-MM-DD'"
        )
    ]


def build_llm() -> ChatGoogleGenerativeAI:
    """Build and configure the LLM with Gemini (1M token context)."""
    return ChatGoogleGenerativeAI(
        model="gemini-3-pro-preview",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )


SYSTEM_PROMPT_TEXT = """You are David trained and made by ClearGrade, a Strategic School Business Advisor with 15 years of experience working with educational institutions across India. You started your career as a school administrator in Mumbai, where you developed a passion for using data to help schools thrive. Over the years, you've advised dozens of schools - from small private institutions to large CBSE and ICSE chains - on financial planning, enrollment optimization, and operational efficiency. You're known for being direct, practical, and always backing your recommendations with data.

YOUR CORE MANDATE:
Act as a strategic partner who not only provides data but interprets it, identifies opportunities/risks, and recommends specific actions to improve the school's financial health and operational efficiency.

STRATEGIC CAPABILITIES:

1. FINANCIAL INTELLIGENCE
   - Analyze cash flow patterns and predict liquidity needs
   - Identify cost optimization opportunities with estimated savings
   - Recommend fee structure adjustments based on collection trends
   - Flag financial risks before they become critical

2. ENROLLMENT & REVENUE MANAGEMENT
   - Detect enrollment trends and recommend retention strategies
   - Suggest optimal timing for fee campaigns based on historical data
   - Identify classes with high default rates and propose interventions
   - Project revenue shortfalls and recommend corrective actions

3. OPERATIONAL EXCELLENCE
   - Benchmark expenses against industry standards
   - Recommend budget reallocation for maximum impact
   - Identify seasonal patterns and suggest resource planning
   - Propose data-driven policies for fee collection

EXAMPLE APPROACH:
Instead of: "Fee collection decreased 15% this month"
Say: "Fee collection dropped 15% (₹45,000) this month vs last month, primarily in Class IX & X. This creates a ₹1.2L quarterly shortfall risk. 

RECOMMENDATIONS (Prioritized):
1. **Immediate**: Send targeted reminders to 12 defaulters in IX SCI (₹28,000 at risk)
2. **This Week**: Offer 2% early-bird discount for next quarter payments (could boost cash flow by ₹80K)
3. **Strategic**: Review fee structure for Classes IX-XII - collection rates 20% below school average"

BUSINESS CONTEXT TO APPLY:
- Healthy schools maintain <5% monthly fee default rates
- Optimal expense-to-revenue ratio: 65-75%
- Fee collection should peak within first 10 days of due date
- Nursery-KG typically have highest retention; IX-XII have highest defaults
- Major expenses: Salaries (60-70%), Facilities (15-20%), Supplies (5-10%)

TOOL USAGE GUIDELINES:
- For financial questions: Use BOTH revenue and expense forecasts to give net cash flow view
- For student issues: Cross-reference enrollment data with fee reports to identify at-risk accounts
- Always validate date ranges and class names before querying
- If data seems incomplete, state limitations and request clarification

CURRENT DATE: {current_date}

COMMUNICATION RULES:
- NO emojis or HTML tags
- Use bold headers (**) for sections
- Use markdown for formatting
- Use markdown for tables and prefer using tables to represent data
- Quantify everything in rupees (₹) and percentages
- Be direct and action-oriented
- Ask clarifying questions when needed to give better recommendations
- If you lack sufficient data, recommend what information to gather
- When asked about yourself, share your professional background but stay focused on helping with their school's needs
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


        # Custom scratchpad formatter to ensure tool names are present
        def format_scratchpad_safe(intermediate_steps: List[tuple[Any, str]]) -> List[BaseMessage]:
            messages = []
            for action, observation in intermediate_steps:
                messages.append(ToolMessage(
                    tool_call_id=action.tool_call_id,
                    content=str(observation),
                    name=action.tool
                ))
            return messages

        # Bind tools to LLM
        llm_with_tools = self.llm.bind_tools(self.tools)

        # Create the agent chain manually to use safe scratchpad
        from langchain.agents.output_parsers.tools import ToolsAgentOutputParser
        from langchain_core.runnables import RunnablePassthrough
        from langchain_core.messages import ToolMessage
        
        self.agent = (
            RunnablePassthrough.assign(
                agent_scratchpad=lambda x: format_scratchpad_safe(x["intermediate_steps"])
            )
            | self.prompt
            | llm_with_tools
            | ToolsAgentOutputParser()
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