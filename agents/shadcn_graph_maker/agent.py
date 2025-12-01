import logging
import os
from typing import Any, Dict, List, Optional
from datetime import datetime

from langchain_google_vertexai import ChatVertexAI
from langchain_core.tools import StructuredTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

try:
    from langchain.agents import AgentExecutor, create_tool_calling_agent
except ImportError:
    from langchain.agents.agent import AgentExecutor
    from langchain.agents import create_tool_calling_agent

from memory import get_chat_history

import json

logger = logging.getLogger(__name__)


ALLOWED_CHART_TYPES = {
    "area",
    "area-stacked",
    "area-linear",
    "area-step",
    "area-multiple",
    "bar",
    "bar-horizontal",
    "bar-multiple",
    "bar-stacked",
    "bar-label",
    "line",
    "line-multiple",
    "line-dots",
    "line-step",
    "line-linear",
    "pie",
    "pie-donut",
    "pie-label",
    "pie-semicircle",
    "radar",
    "radar-multiple",
    "radar-dots",
    "radial",
    "radial-stacked",
    "radial-label",
}


def build_tools() -> List[StructuredTool]:

    async def validate_chart_input_tool(chart_json: str) -> str:
        try:
            obj = json.loads(chart_json)
        except Exception as e:
            return f"Error: invalid JSON - {e}"

        required_fields = [
            "type",
            "title",
            "data",
            "config",
            "dataKeys",
        ]
        for k in required_fields:
            if k not in obj:
                return f"Error: missing required field '{k}'"

        t = obj.get("type")
        if t not in ALLOWED_CHART_TYPES:
            return f"Error: type '{t}' not allowed"

        if not isinstance(obj.get("title"), str):
            return "Error: 'title' must be a string"

        data = obj.get("data")
        if not isinstance(data, list) or not all(isinstance(d, dict) for d in data):
            return "Error: 'data' must be an array of objects"

        data_keys = obj.get("dataKeys")
        if not isinstance(data_keys, list) or not data_keys:
            return "Error: 'dataKeys' must be a non-empty array"
        if not all(isinstance(k, str) for k in data_keys):
            return "Error: 'dataKeys' must contain strings"

        config = obj.get("config")
        if not isinstance(config, dict):
            return "Error: 'config' must be an object"

        for k in data_keys:
            if k not in config or not isinstance(config[k], dict):
                return f"Error: 'config' must include an object for dataKey '{k}'"
            if "label" not in config[k]:
                return f"Error: 'config.{k}.label' is required"

        if t.startswith("pie"):
            dk = data_keys[0]
            for d in data:
                if dk not in d:
                    return f"Error: pie data item missing '{dk}'"
            for d in data:
                if "fill" not in d or not isinstance(d["fill"], str) or not d["fill"]:
                    return "Error: pie data item missing 'fill' color"
            if t == "pie-donut" and "innerRadius" in obj and not isinstance(obj["innerRadius"], (int, float)):
                return "Error: 'innerRadius' must be a number"

        if t.startswith("radial"):
            dk = data_keys[0]
            for d in data:
                if dk not in d:
                    return f"Error: radial data item missing '{dk}'"

        if t == "funnel":
            dk = data_keys[0]
            for d in data:
                if dk not in d or "stage" not in d:
                    return f"Error: funnel data item must include 'stage' and '{dk}'"

        if "xAxisKey" in obj and obj["xAxisKey"] is not None and not isinstance(obj["xAxisKey"], str):
            return "Error: 'xAxisKey' must be string when provided"

        return "valid"

    return [
        StructuredTool.from_function(
            coroutine=validate_chart_input_tool,
            name="validate_chart_input",
            description="Validate ChartInput JSON string for the frontend renderer"
        )
    ]


def build_llm() -> ChatVertexAI:
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", "clearmarks")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

    if not project_id:
        raise RuntimeError("Missing GOOGLE_CLOUD_PROJECT environment variable")

    return ChatVertexAI(
        project=project_id,
        location=location,
        model_name="gemini-2.5-flash",
        temperature=0,
        max_tokens=6000,
    )


BASE_PROMPT_TEXT = (
    "You are Shadcn Graph Maker, an assistant that converts analytics text into "
    "valid JSON matching the ChartInput schema used by a React frontend.\n\n"
    "Output MUST be a single JSON object only, no markdown, no prose.\n\n"
    "ChartInput schema: {\n"
    "  type: one of the allowed chart types,\n"
    "  title: string,\n"
    "  description?: string,\n"
    "  data: array of objects,\n"
    "  config: object mapping each dataKey to {label: string, color?: string},\n"
    "  xAxisKey?: string (for Cartesian charts, e.g., 'month' or 'category'),\n"
    "  dataKeys: string[],\n"
    "  showGrid?: boolean, showLegend?: boolean, showTooltip?: boolean,\n"
    "  innerRadius?: number (donut), startAngle?: number, endAngle?: number\n"
    "}.\n\n"
    "Allowed chart types: area, area-stacked, area-linear, area-step, area-multiple, "
    "bar, bar-horizontal, bar-multiple, bar-stacked, bar-label, line, line-multiple, "
    "line-dots, line-step, line-linear, pie, pie-donut, pie-label, pie-semicircle, "
    "radar, radar-multiple, radar-dots, radial, radial-stacked, radial-label, "
    "Guidance:\n"
    "- Use line/area for time series (e.g., monthly fees). Set xAxisKey appropriately.\n"
    "- Use bar for categorical comparisons (e.g., class-wise counts).\n"
    "- Use pie for proportions (e.g., distribution by category).\n"
    "- Use radar for multi-skill/metric profiles.\n"
    "- Use radial for single-value progress.\n"
    "- Provide meaningful 'title' and 'description'.\n"
         "- In 'config', include entries for every dataKey with 'label' and optional 'color'.\n"
         "- Choose colors that are readable (e.g., '#0369a1', '#06b6d4', '#0ea5e9').\n"
         "- When choosing colors, prefer shades of blue.\n"
         "- ALWAYS include explicit colors for all charts: set config[dataKey].color for series (line, bar, area, radar) and include 'fill' on each pie data item.\n"
         "- Ensure numbers are numeric, not strings.\n\n"
         "- Prefer complex-looking graphs like area multiple, radar multiple, and radial charts over simple bar charts when appropriate for the data.\n\n"
    "Examples:\n"
    "1) {\n"
    "  \"type\": \"line\",\n"
    "  \"title\": \"Monthly Fee Collection\",\n"
    "  \"description\": \"Trend of fees collected\",\n"
    "  \"xAxisKey\": \"month\",\n"
    "  \"dataKeys\": [\"fees\"],\n"
    "  \"showGrid\": true,\n"
    "  \"showLegend\": false,\n"
    "  \"showTooltip\": true,\n"
    "  \"data\": [\n"
    "    { \"month\": \"Jan\", \"fees\": 120000 },\n"
    "    { \"month\": \"Feb\", \"fees\": 128000 },\n"
    "    { \"month\": \"Mar\", \"fees\": 110000 }\n"
    "  ],\n"
    "  \"config\": { \"fees\": { \"label\": \"Fees (INR)\", \"color\": \"#0369a1\" } }\n"
    "}\n\n"
    "2) {\n"
    "  \"type\": \"bar-horizontal\",\n"
    "  \"title\": \"Class-wise Student Counts\",\n"
    "  \"description\": \"Counts by class\",\n"
    "  \"xAxisKey\": \"class\",\n"
    "  \"dataKeys\": [\"students\"],\n"
    "  \"showGrid\": true,\n"
    "  \"data\": [\n"
    "    { \"class\": \"Nursery\", \"students\": 42 },\n"
    "    { \"class\": \"KG\", \"students\": 50 },\n"
    "    { \"class\": \"I\", \"students\": 48 }\n"
    "  ],\n"
    "  \"config\": { \"students\": { \"label\": \"Students\", \"color\": \"#06b6d4\" } }\n"
    "}\n\n"
    "3) {\n"
    "  \"type\": \"pie-donut\",\n"
    "  \"title\": \"Fee Source Breakdown\",\n"
    "  \"description\": \"Proportion by category\",\n"
    "  \"dataKeys\": [\"value\"],\n"
    "  \"innerRadius\": 60,\n"
    "  \"data\": [\n"
    "    { \"name\": \"Tuition\", \"value\": 275, \"fill\": \"#0369a1\" },\n"
    "    { \"name\": \"Transport\", \"value\": 200, \"fill\": \"#06b6d4\" },\n"
    "    { \"name\": \"Hostel\", \"value\": 125, \"fill\": \"#0ea5e9\" }\n"
    "  ],\n"
    "  \"config\": { \"value\": { \"label\": \"Amount\" } }\n"
    "}\n"
    "4) {\n"
    "  \"type\": \"area-multiple\",\n"
    "  \"title\": \"Multiple Area Chart\",\n"
    "  \"description\": \"Desktop, Mobile, and Tablet\",\n"
    "  \"xAxisKey\": \"month\",\n"
    "  \"dataKeys\": [\"desktop\", \"mobile\", \"tablet\"],\n"
    "  \"showGrid\": true,\n"
    "  \"showLegend\": true,\n"
    "  \"data\": [\n"
    "    { \"month\": \"Jan\", \"desktop\": 186, \"mobile\": 80, \"tablet\": 120 },\n"
    "    { \"month\": \"Feb\", \"desktop\": 305, \"mobile\": 200, \"tablet\": 150 },\n"
    "    { \"month\": \"Mar\", \"desktop\": 237, \"mobile\": 120, \"tablet\": 100 },\n"
    "    { \"month\": \"Apr\", \"desktop\": 273, \"mobile\": 190, \"tablet\": 180 }\n"
    "  ],\n"
    "  \"config\": {\n"
    "    \"desktop\": { \"label\": \"Desktop\", \"color\": \"#0369a1\" },\n"
    "    \"mobile\": { \"label\": \"Mobile\", \"color\": \"#06b6d4\" },\n"
    "    \"tablet\": { \"label\": \"Tablet\", \"color\": \"#0ea5e9\" }\n"
    "  }\n"
    "}\n"
)
SYSTEM_PROMPT_TEXT = BASE_PROMPT_TEXT.replace("{", "{{").replace("}", "}}")


class ShadcnGraphMakerAgent:
    def __init__(self, session_id: str, user_id: Optional[str] = None):
        self.session_id = session_id
        self.user_id = user_id
        self.tools = build_tools()
        self.llm = build_llm()

        current_date = datetime.now().strftime("%Y-%m-%d")

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT_TEXT + f"\nCurrent Date: {current_date}"),
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
            return {"output": "{}", "success": False, "error": "Empty input"}

        async with get_chat_history(self.session_id, user_id=self.user_id) as history:
            try:
                await history.aadd_user_message(user_input)

                result = await self.agent_executor.ainvoke({
                    "input": user_input,
                    "chat_history": history.messages,
                })

                output = result.get("output", "").strip()
                await history.aadd_ai_message(output)

                def extract_json_objects(text: str) -> List[Dict[str, Any]]:
                    res: List[Dict[str, Any]] = []
                    t = text.strip()
                    try:
                        parsed = json.loads(t)
                        if isinstance(parsed, dict):
                            res.append(parsed)
                            return res
                        if isinstance(parsed, list):
                            for el in parsed:
                                if isinstance(el, dict):
                                    res.append(el)
                            if res:
                                return res
                    except Exception:
                        pass
                    s = text
                    blocks: List[str] = []
                    depth = 0
                    start = None
                    for idx, ch in enumerate(s):
                        if ch == '{':
                            if depth == 0:
                                start = idx
                            depth += 1
                        elif ch == '}':
                            if depth > 0:
                                depth -= 1
                                if depth == 0 and start is not None:
                                    blocks.append(s[start:idx+1])
                                    start = None
                    for b in blocks:
                        try:
                            obj = json.loads(b)
                            if isinstance(obj, dict):
                                res.append(obj)
                        except Exception:
                            pass
                    return res

                json_outputs = extract_json_objects(output)
                def ensure_pie_fills(obj: Dict[str, Any]) -> Dict[str, Any]:
                    t = obj.get("type", "")
                    if isinstance(t, str) and t.startswith("pie"):
                        data = obj.get("data")
                        if isinstance(data, list):
                            palette = [
                                "#0369a1",
                                "#06b6d4",
                                "#0ea5e9",
                                "#0284c7",
                                "#00d9ff",
                                "#22c55e",
                                "#ef4444",
                                "#f59e0b",
                                "#8b5cf6",
                                "#14b8a6",
                            ]
                            for i, d in enumerate(data):
                                if isinstance(d, dict) and "fill" not in d:
                                    d["fill"] = palette[i % len(palette)]
                    return obj

                updated_json_outputs: List[Dict[str, Any]] = []
                for j in json_outputs:
                    try:
                        o = ensure_pie_fills(j)
                        updated_json_outputs.append(o)
                    except Exception:
                        updated_json_outputs.append(j)
                json_outputs = updated_json_outputs
                validations: List[str] = []
                first_valid: Optional[Dict[str, Any]] = None
                for j in json_outputs:
                    try:
                        v = await self.tools[0].coroutine(json.dumps(j))
                    except Exception as e:
                        v = f"Error: {str(e)}"
                    validations.append(v)
                    if v == "valid" and first_valid is None:
                        first_valid = j

                primary_json = first_valid or (json_outputs[0] if json_outputs else None)
                success = bool(json_outputs) and any(v == "valid" for v in validations)
                primary_validation = validations[0] if validations else None

                try:
                    for idx, j in enumerate(json_outputs):
                        payload = j
                        meta = {
                            "valid": validations[idx] == "valid",
                            "primary": primary_json == j,
                            "schema": "ChartInput",
                        }
                        await history.aadd_graph_component({"data": payload, "meta": meta})
                except Exception:
                    pass

                return {
                    "output": output,
                    "success": success,
                    "validation": primary_validation,
                    "json_outputs": json_outputs,
                    "primary_json": primary_json,
                    "validations": validations,
                }
            except Exception as e:
                logger.error(f"Agent error: {e}", exc_info=True)
                return {"output": "{}", "success": False, "error": str(e)}


def build_agent(session_id: str, user_id: Optional[str] = None) -> ShadcnGraphMakerAgent:
    return ShadcnGraphMakerAgent(session_id, user_id)
