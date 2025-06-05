import streamlit as st
import asyncio
from uuid import uuid4
from agent import create_agent, AgentState
from langchain_core.messages import HumanMessage, AIMessage
import json

# Initialize agent
if 'agent' not in st.session_state:
    st.session_state.agent = create_agent()

# Initialize thread ID
if 'thread_id' not in st.session_state:
    st.session_state.thread_id = str(uuid4())

# Initialize conversation
if 'conversation' not in st.session_state:
    st.session_state.conversation = []

# Display conversation
st.title("🚀 Chat Agent")
st.caption("Powered by Gemini Flash and LangGraph")

# Display messages
for msg in st.session_state.conversation:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "tool_used" in msg:
            st.caption(f"🔧 Used tool: {msg['tool_used']}")

# User input
if prompt := st.chat_input("Ask anything..."):
    # Add user message to UI
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add to conversation history
    st.session_state.conversation.append({"role": "user", "content": prompt})
    
    # Create agent input
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    initial_state = {
        "messages": [
            HumanMessage(content=msg["content"]) 
            for msg in st.session_state.conversation 
            if msg["role"] == "user"
        ]
    }
    
    # Run agent
    with st.spinner("Thinking..."):
        final_state = asyncio.run(
            st.session_state.agent.ainvoke(initial_state, config=config)
        )
    
    # Process agent response
    last_message = final_state["messages"][-1]
    response_content = last_message.content
    
    # Check for tool use
    tool_used = None
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        tool_used = last_message.tool_calls[0]["name"]
    
    # Add to conversation
    st.session_state.conversation.append({
        "role": "assistant", 
        "content": response_content,
        "tool_used": tool_used
    })
    
    # Display response
    with st.chat_message("assistant"):
        st.markdown(response_content)
        if tool_used:
            st.caption(f"🔧 Used tool: {tool_used}")

# Sidebar with controls
with st.sidebar:
    st.header("Agent Controls")
    
    # New conversation button
    if st.button("Start New Conversation"):
        st.session_state.thread_id = str(uuid4())
        st.session_state.conversation = []
        st.rerun()
    
    # Display current thread ID
    st.divider()
    st.subheader("Current Session")
    st.caption(f"Thread ID: `{st.session_state.thread_id}`")
    
    # Conversation download
    st.divider()
    st.subheader("Export Conversation")
    if st.download_button(
        label="Download as JSON",
        data=json.dumps(st.session_state.conversation, indent=2),
        file_name="conversation.json",
        mime="application/json"
    ):
        st.success("Conversation downloaded!")
    
    # Agent info
    st.divider()
    st.subheader("Agent Information")
    st.markdown("""
    - **Model**: Gemini 2.0 Flash
    - **Tools**: DuckDuckGo Search
    - **Framework**: LangGraph
    """)

# Required imports at bottom
import json