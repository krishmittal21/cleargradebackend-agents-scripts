"""
Simple test script for the agent.

Usage:
    python test_agent.py
"""

import asyncio
import logging
from agent import build_agent

logging.basicConfig(level=logging.INFO)


async def test_agent():
    """Test the agent with sample queries."""
    
    # TODO: Update these test queries to match your agent's capabilities
    test_queries = [
        "Hello, what can you help me with?",
        "List items in category 'example'",
        "Show me details for item 123",
        "Generate a report from 2024-01-01 to 2024-01-31",
    ]
    
    print("=" * 80)
    print("Testing Agent")
    print("=" * 80)
    
    # Create agent instance
    agent = build_agent(session_id="test-session", user_id="test-user")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n[Query {i}]: {query}")
        print("-" * 80)
        
        try:
            result = await agent.ainvoke({"input": query})
            
            if result.get("success"):
                print(f"[Response]: {result.get('output')}")
            else:
                print(f"[Error]: {result.get('error')}")
        except Exception as e:
            print(f"[Exception]: {str(e)}")
        
        print("-" * 80)
    
    print("\n" + "=" * 80)
    print("Testing Complete")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_agent())
