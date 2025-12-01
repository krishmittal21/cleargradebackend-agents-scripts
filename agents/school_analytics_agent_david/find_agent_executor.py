import pkgutil
import langchain.agents
import importlib

print(f"Langchain agents path: {langchain.agents.__path__}")

def find_agent_executor():
    for loader, module_name, is_pkg in pkgutil.walk_packages(langchain.agents.__path__, langchain.agents.__name__ + "."):
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, "AgentExecutor"):
                print(f"Found AgentExecutor in: {module_name}")
                return
        except Exception as e:
            pass

find_agent_executor()
