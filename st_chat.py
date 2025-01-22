import streamlit as st
from langchain.callbacks.tracers.run_collector import RunCollectorCallbackHandler
from langchain.schema.runnable import RunnableConfig
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import math
from streamlit.components.v1 import html
import tiktoken
import os
from operator import itemgetter
from tiktoken.core import Encoding
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit_antd_components as sac
encoding: Encoding = tiktoken.encoding_for_model("gpt-4o")

def load_html_template(template_name, **kwargs):
    with open(os.path.join('templates', template_name), 'r', encoding='utf-8') as file:
        html_content = file.read()
    for key, value in kwargs.items():
        html_content = html_content.replace(f"{{{key}}}", value)
    return html_content

def _get_openai_type(msg):
    if msg.type == "human":
        return "user"
    if msg.type == "ai":
        return "assistant"
    if msg.type == "chat":
        return msg.role
    return msg.type

def clear_chat():
    st.session_state.messages = []
    st.session_state.Clear=False
    st.session_state.feedback=False
    st.session_state.total_tokens=0
    st.session_state.memory.clear()
    st.session_state.done = True
    st.rerun()

def show_messages(messages,memory,edit,new_message=None):
    st.session_state.total_tokens = 0 
    i=0
    memory.clear()
    for msg in messages:
        streamlit_type = _get_openai_type(msg)
        avatar = "🤖" if streamlit_type == "assistant" else "😊"
        with st.chat_message(streamlit_type, avatar=avatar):
            if streamlit_type=="user":
                st.session_state.total_tokens+=len(encoding.encode(msg.content))
                col1,  col2 = st.columns([9,  1])
                with col1:
                    st.markdown(msg.content.replace("\n","<br>"),unsafe_allow_html=True)
                if edit:
                    with col2:
                        key = st.button("edit",key=f"edit_{i}")
                        if key:
                            st.session_state.edit = True
                    if st.session_state.edit:
                        st.session_state.new_message = st.text_area("編集したらsaveしてください。", value=msg.content)
                        save = st.button("save",key=f"save_{i}")
                        if save:
                            st.session_state.edit = False
                            messages = modify_message(messages, i,memory)
                            st.session_state.save = True
                else:
                    with col2:
                        key = st.button("edit",key=f"dummy_{i}")

            else:
                st.session_state.total_tokens+=len(encoding.encode(msg.content))
                col1,  col2 = st.columns([9,  1])
                with col1:
                    st.markdown(msg.content.replace("\n","  \n"),unsafe_allow_html=True)
        memory.chat_memory.add_message(msg)
        i+=1
    if new_message:
        st.session_state.total_tokens+=len(encoding.encode(new_message))
        with st.chat_message("user", avatar="😊"):
            st.markdown(new_message.replace("\n","<br>"),unsafe_allow_html=True)

def check_token():
    if st.session_state.total_tokens>20000:
        persent=math.floor(st.session_state.total_tokens/200)
        if st.button('clear chat history'):
            clear_chat()
        st.error(f'エラー：文章量が上限に対し、{persent}%となっています。  \n不必要な箇所を削除するか会話をリセットしてください', icon="🚨")
        st.stop()
    if len(st.session_state.messages) > 30:
        if st.button('clear chat history'):
            clear_chat()
        st.error('エラー：会話上限数を超えました。会話をリセットしてください', icon="🚨")
        st.stop()

def modify_message(messages, i, memory):
    memory.clear()
    del messages[i:]
    for msg in messages:
        memory.chat_memory.add_message(msg)
    return messages

def st_Chat():
    st.title("ChatGPT")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    attrs=["Clear"]
    for attr in attrs:
        if attr not in st.session_state:
            st.session_state[attr] = False
    attrs=["trace_link","run_id"]
    for attr in attrs:
        if attr not in st.session_state:
            st.session_state[attr] = None
    if "engine" not in st.session_state:
        st.session_state.engine = "gpt-4"
    if not hasattr(st.session_state, "temperature"):
        st.session_state.temperature = 0.7
    attrs=["edit","save","stop"]
    for attr in attrs:
        if attr not in st.session_state:
            st.session_state[attr] = False
    attrs=["done"]
    for attr in attrs:
        if attr not in st.session_state:
            st.session_state[attr] = True

    attrs=["presence_penalty","frequency_penalty","total_tokens"]
    for attr in attrs:
        if not hasattr(st.session_state, attr):
            st.session_state[attr]= 0.0
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = "あなたは優秀なAIアシスタントです。"
    attrs=["last_response","full_response","prompt"]
    for attr in attrs:
        if attr not in st.session_state:
            st.session_state.attr = ""

    st.write("**ChatGPTと会話することが出来ます。**")

    with st.expander("オプション"):
        engine = st.selectbox("model",("gpt-4o","gpt-4o-mini","claude-3-5-sonnet","gemini-exp-1206"),help="modelを選択できます。")
        system_prompt = st.text_area("system prompt",value="あなたは優秀なAIアシスタントです。",help="systemにpromptを与えられます。初回メッセージ送信時のみ有効です。")
        temperature = st.slider(label="temperature",min_value=0.0, max_value=1.0,value=st.session_state.temperature,help="生成されるテキストのランダム性を制御します。")

    if system_prompt != st.session_state.system_prompt:
        st.session_state.system_prompt = system_prompt
    if temperature != st.session_state.temperature:
        st.session_state.temperature = temperature
    if engine != st.session_state.engine:
        st.session_state.engine = engine

    if engine == "gpt-4o" or engine == "gpt-4o-mini":
        model = ChatOpenAI(model = st.session_state.engine,temperature=st.session_state.temperature)
    elif engine == "claude-3-5-sonnet":
        model = ChatAnthropic(model_name="claude-3-5-sonnet-20240620",temperature=st.session_state.temperature)
    elif engine == "gemini-exp-1206":
        model = ChatGoogleGenerativeAI(model="gemini-exp-1206",temperature=st.session_state.temperature)

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", st.session_state.system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )
    
    st.session_state.memory = ConversationBufferMemory(return_messages=True)
    st.session_state.memory.load_memory_variables({})
    chain = (
        RunnablePassthrough.assign(
            history=RunnableLambda(st.session_state.memory.load_memory_variables) | itemgetter("history")
        )
        | prompt_template
        | model
    )

    show_messages(messages=st.session_state.messages,memory=st.session_state.memory,edit=True)
    if not st.session_state.done:
        st.session_state.memory.chat_memory.add_user_message(st.session_state.prompt)
        st.session_state.memory.chat_memory.add_ai_message(st.session_state.full_response)
        st.session_state.messages = st.session_state.memory.buffer
        st.session_state.Clear = True
        st.session_state.done = True
        st.session_state.save = False
        st.rerun()

    run_collector = RunCollectorCallbackHandler()
    runnable_config = RunnableConfig(
        callbacks=[run_collector],
        tags=["Chat"],
    )
    
    if prompt := st.chat_input(placeholder="Send a message"):
        st.session_state.done = False
        st.session_state.prompt = prompt
        st.session_state.edit = False
        st.session_state.total_tokens+=len(encoding.encode(prompt))
        check_token()

        with st.chat_message("user",avatar="😊"):
            col1,  col2 = st.columns([9,  1])
            with col1:
                st.markdown(prompt.replace("\n","  \n"),unsafe_allow_html=True)
        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            st.session_state.full_response = ""
            message_placeholder.markdown("thinking...")
            col1,  col2 = st.columns([9,  1])
            with col2:
                st.session_state.stop = st.button("stop")
            with col1:
                for chunk in chain.stream({"input": prompt}, config=runnable_config):
                    if not st.session_state.stop:
                        st.session_state.full_response += chunk.content
                        message_placeholder.markdown(st.session_state.full_response.replace("\n","  \n") + "▌",unsafe_allow_html=True)              
                st.session_state.done = True
                message_placeholder.markdown(st.session_state.full_response.replace("\n","  \n"),unsafe_allow_html=True)
                st.session_state.last_response=st.session_state.full_response.replace("\n", "\\n").replace('"', '\\"')
                st.session_state.memory.save_context({"input": prompt}, {"output": st.session_state.full_response})
                st.session_state.messages = st.session_state.memory.buffer
        st.session_state.Clear=True
        st.rerun()

    if st.session_state.save:
        st.session_state.done = False
        prompt = st.session_state.new_message
        st.session_state.prompt = prompt
        show_messages(messages=st.session_state.messages,memory=st.session_state.memory,edit=False,new_message=prompt)
        check_token()
        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            st.session_state.full_response = ""
            message_placeholder.markdown("thinking...")
            col1,  col2 = st.columns([9,  1])
            with col2:
                st.session_state.stop = st.button("stop")
            with col1:
                for chunk in chain.stream({"input": prompt}, config=runnable_config):
                    if not st.session_state.stop:
                        st.session_state.full_response += chunk.content
                        message_placeholder.markdown(st.session_state.full_response.replace("\n","  \n") + "▌",unsafe_allow_html=True)              
                st.session_state.done = True
                message_placeholder.markdown(st.session_state.full_response.replace("\n","  \n"),unsafe_allow_html=True)
                st.session_state.last_response=st.session_state.full_response.replace("\n", "\\n").replace('"', '\\"')
                st.session_state.memory.save_context({"input": prompt}, {"output": st.session_state.full_response})
                st.session_state.messages = st.session_state.memory.buffer

        st.session_state.Clear=True
        st.session_state.save = False
        st.rerun()

    if st.session_state.Clear:
        html_code_message = load_html_template(
            'copyutton_message.html',
            last_response=st.session_state.last_response
        )
        html(html_code_message, height=50)
        if st.button('clear chat history'):
            clear_chat()
                    
if __name__ == "__main__":
    st_Chat()