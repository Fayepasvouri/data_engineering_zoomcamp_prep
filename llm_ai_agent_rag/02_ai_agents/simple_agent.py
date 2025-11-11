"""
Simple AI Agent - Learn agent fundamentals
"""
from typing import List, Dict, Callable
import json


class SimpleAgent:
    """Basic AI Agent for learning"""

    def __init__(self, name: str):
        self.name = name
        self.tools: Dict[str, Callable] = {}
        self.memory: List[Dict] = []

    def register_tool(self, tool_name: str, tool_func: Callable):
        """Register available tools"""
        self.tools[tool_name] = tool_func

    def list_tools(self) -> list:
        """List available tools"""
        return list(self.tools.keys())

    def think(self, query: str) -> str:
        """Agent reasoning process"""
        # Simple reasoning: choose most relevant tool
        if "calculate" in query.lower() and "calculator" in self.tools:
            return "calculator"
        elif "search" in query.lower() and "search" in self.tools:
            return "search"
        else:
            return "default"

    def execute(self, query: str) -> Dict:
        """Execute agent actions"""
        # Decide which tool to use
        tool_name = self.think(query)

        result = None
        if tool_name in self.tools:
            try:
                result = self.tools[tool_name](query)
            except Exception as e:
                result = f"Error using {tool_name}: {str(e)}"
        else:
            result = "No suitable tool found"

        # Store in memory
        self.memory.append({
            "query": query,
            "tool_used": tool_name,
            "result": result
        })

        return {
            "query": query,
            "tool_used": tool_name,
            "result": result
        }

    def get_memory(self) -> List[Dict]:
        """Get agent's memory"""
        return self.memory

    def clear_memory(self):
        """Clear agent's memory"""
        self.memory = []


if __name__ == "__main__":
    # Create agent
    agent = SimpleAgent("DataAgent")

    # Define some tools
    def calculator(query: str) -> str:
        """Simple calculator tool"""
        return f"Calculated based on: {query}"

    def search(query: str) -> str:
        """Simple search tool"""
        return f"Searched for: {query}"

    # Register tools
    agent.register_tool("calculator", calculator)
    agent.register_tool("search", search)

    print(f"Agent: {agent.name}")
    print(f"Available tools: {agent.list_tools()}\n")

    # Execute queries
    response1 = agent.execute("Calculate 5 + 3")
    print(f"Query 1: {response1}\n")

    response2 = agent.execute("Search for data engineering")
    print(f"Query 2: {response2}\n")

    # Show memory
    print("Agent Memory:")
    for mem in agent.get_memory():
        print(f"- {mem['query']} → {mem['tool_used']}")
