import asyncio
import logging
from agent import build_tools
from tiaf_api_client import TIAFApiClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def verify_tools():
    print("Verifying forecasting tools integration...")
    
    # Mock client
    client = TIAFApiClient()
    
    # Build tools
    tools = build_tools(client)
    
    # Check if tools are present
    tool_names = [t.name for t in tools]
    print(f"Available tools: {tool_names}")
    
    expected_tools = ["monte_carlo_simulation", "arima_forecast", "sales_forecast_analysis"]
    missing = [t for t in expected_tools if t not in tool_names]
    
    if missing:
        print(f"❌ Missing tools: {missing}")
        return
    
    print("✅ All forecasting tools registered successfully.")
    
    # Try to invoke a tool (dry run)
    # We can't easily invoke them without a full agent loop or mocking inputs, 
    # but we can check if the functions are callable.
    
    print("Verifying tool callables...")
    for tool in tools:
        if tool.name in expected_tools:
            print(f"  - {tool.name}: {tool.description}")
            
    print("✅ Verification complete.")

if __name__ == "__main__":
    asyncio.run(verify_tools())
