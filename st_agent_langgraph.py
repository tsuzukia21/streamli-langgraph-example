import streamlit as st
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, AIMessage, SystemMessage
import asyncio
from typing import Literal
from langchain_core.runnables import RunnableConfig
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI

class State(TypedDict):
    messages: Annotated[list, add_messages]

config = {"configurable": {"thread_id": "1"},"run_name":"Chat"}

def should_continue(state: State) -> Literal["__end__", "tools"]:
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return END
    else:
        return "tools"

# async def call_model(state: State, config: RunnableConfig):
#     messages = state["messages"]
#     response = await llm_with_tools.ainvoke(messages, config)
#     return {"messages": response}

def call_model(state: State):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": response}


if "app" not in st.session_state:
    # llm = ChatAnthropic(model="claude-3-haiku-20240307", streaming=True)
    llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)
    tool = TavilySearchResults(max_results=2)
    tools = [tool]
    tool_node = ToolNode(tools)
    llm_with_tools = llm.bind_tools(tools)
    memory = MemorySaver()
    workflow = StateGraph(State)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
    )
    workflow.add_edge("tools", "agent")
    app = workflow.compile(checkpointer=memory)
    st.session_state.app = app

snapshot = st.session_state.app.get_state(config)
messages = snapshot.values.get('messages', [])
# st.markdown(messages)

for message in messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user", avatar="😊"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        if message.content:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(message.content)

if prompt := st.chat_input(placeholder="Send a message"):
    st.session_state.contents = ""
    with st.chat_message("user",avatar="😊"):
        st.markdown(prompt)
    with st.chat_message("assistant",avatar="🤖"):
        message_placeholder = st.empty()
        tool_placeholder = st.empty()
        st.session_state.contents = ""
        async def process_stream():
            async for event in st.session_state.app.astream_events({"messages": prompt}, config ,version="v1"):
                kind = event["event"]
                if kind == "on_chat_model_stream":
                    content = event["data"]["chunk"].content
                    # st.markdown(content)
                    if content:
                        st.session_state.contents += content
                        # st.session_state.contents += content[0]['text']
                        message_placeholder.markdown(st.session_state.contents)
                        # st.markdown(content, end="|")
                elif kind == "on_tool_start":
                    # st.markdown("--")
                    # st.markdown(
                    #     f"Starting tool: {event['name']} with inputs: {event['data'].get('input')}"
                    # )
                    tool_placeholder.markdown(f"Starting tool: {event['name']} with inputs: {event['data'].get('input')}")
                elif kind == "on_tool_end":
                    # st.markdown(f"Done tool: {event['name']}")
                    # st.markdown(f"Tool output was: {event['data'].get('output')}")
                    # st.markdown("--")
                    tool_placeholder.markdown(f"Done tool: {event['name']}")
                    tool_placeholder.markdown(f"Tool output was: {event['data'].get('output')}")
        asyncio.run(process_stream())
        message_placeholder.markdown(st.session_state.contents)
        tool_placeholder.empty()


# 
