from typing import TypedDict, Annotated, Sequence
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import BaseMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END
import operator

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

def create_agent():
    # Initialize model
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-001",
        temperature=0.7
    )
    
    # Create tools
    search_tool = DuckDuckGoSearchRun(max_results=5)
    tools = [search_tool]
    
    # Bind tools to model
    llm_with_tools = model.bind_tools(tools)
    
    # Define nodes
    def model_node(state: AgentState):
        result =llm_with_tools.invoke(state["messages"])
        return {"messages": [result]}
    
    def tools_route(state: AgentState):
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tool_node"
        return END
    
    # Create tool node
    tool_node = ToolNode(tools)
    
    # Build graph
    graph = StateGraph(AgentState)
    graph.add_node("model", model_node)
    graph.add_node("tool_node", tool_node)
    graph.set_entry_point("model")
    graph.add_conditional_edges("model", tools_route)
    graph.add_edge("tool_node", "model")
    
    return graph.compile(checkpointer=MemorySaver())