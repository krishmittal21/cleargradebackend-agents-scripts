import langchain.agents
print(dir(langchain.agents))
try:
    from langchain.agents import AgentExecutor
    print("Found AgentExecutor in langchain.agents")
except ImportError:
    print("AgentExecutor NOT in langchain.agents")

try:
    import langchain.agents.agent
    print("Found langchain.agents.agent")
except ImportError:
    print("langchain.agents.agent NOT found")
