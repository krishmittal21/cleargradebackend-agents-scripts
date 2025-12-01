import os
import asyncio
from dotenv import load_dotenv
from agent import build_agent

load_dotenv()

async def main():
    print("Building agent...")
    try:
        agent = build_agent(session_id="test-session", user_id="test-user")
        print("Agent built successfully.")
        
        print("Testing agent invocation...")
        # We'll just test if the object is created and has the right LLM type
        print(f"LLM Type: {type(agent.llm)}")
        
        # Optional: Try a simple invoke if you want to test connectivity
        # response = await agent.ainvoke({"input": "Hello, who are you?"})
        # print(f"Response: {response}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
