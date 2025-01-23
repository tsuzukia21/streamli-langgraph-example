from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing import Annotated, Literal, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langgraph.prebuilt import tools_condition
from langchain_core.tools import tool
import re
import os
from dataclasses import dataclass
from typing import List
from langgraph.constants import Send
from langgraph.checkpoint.memory import MemorySaver
import streamlit as st
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
import asyncio

class AgentState(TypedDict):
    """
    グラフの状態を表します。

    属性:
        messages: 会話履歴
        filterd_docs: 文書のリスト
        question: ユーザーからの質問
        query: 検索クエリ
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    filterd_docs: List[dict]
    question: str
    query: str

@dataclass
class dtct:
    source: str
    page: int
    page_content: str
    
async def parse_documents(input_str: str) -> List[dtct]:
    pattern = re.compile(
        r"Document\(metadata=\{'source': '(.*?)', 'page': (\d+)\}, page_content='(.*?)'\)",
        re.DOTALL
    )
    
    documents = []
    
    matches = pattern.findall(input_str)
    
    for match in matches:
        source = match[0].replace('\\\\', '\\')
        page = int(match[1])
        page_content = match[2].replace('\\r\\n', '\r\n').replace("\\n", "\n")
        documents.append(dtct(source=source, page=page, page_content=page_content))
    
    return documents

def update_status_and_messages(message: str, state: str = "running", expanded: bool = True, additional_info: str = ""):
    """ステータス、プレースホルダー、ステータスメッセージを一括更新する関数
    Args:
        message (str): 表示するメッセージ
        state (str, optional): ステータスの状態. Defaults to "running".
        expanded (bool, optional): ステータスを展開するかどうか. Defaults to True.
        additional_info (str, optional): プレースホルダーに表示する追加情報. Defaults to "".
    """
    st.session_state.status.update(label=f"{message}", state=state, expanded=expanded)
    if additional_info:
        st.session_state.placeholder.markdown(additional_info)

@tool
async def retriever(query):
    """
    社内規定を検索できます。
    検索する際のキーワードは会話履歴を考慮してください。
    引数:
        query: 検索する際のキーワード

    戻り値:
        documents: 検索された文書
    """
    update_status_and_messages(
        "**---RETRIEVE---**",
        additional_info=f"RETRIEVING…\n\nKEYWORD : {query}"
    )
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    file="employment_rules"
    db=FAISS.load_local(file, embeddings,allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={'k': 6})

    documents = retriever.invoke(query)
    return Send(documents,{"query": query})

async def grade_documents(state) -> Literal["generate", "rewrite"]:
    """
    取得された文書が質問に関連しているかどうかを判断します。

    引数:
        state (messages): 現在の状態

    戻り値:
        str: 文書が関連しているかどうかの判断
    """
    class grade(BaseModel):
        """Binary score for relevance check."""

        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    update_status_and_messages(
        "**---CHECK RELEVANCE---**",
        expanded=False,
    )
    model = ChatOpenAI(temperature=0, model="gpt-4o-mini", streaming=True)
    llm_with_tool = model.with_structured_output(grade)

    prompt = PromptTemplate(
        template="""あなたは、取得された文書がユーザーの質問に関連しているかどうかを評価する採点者です。\n
        以下が取得された文書です：\n\n {context} \n\n
        以下がユーザーの質問です：{question} \n
        文書にユーザーの質問に関連するキーワードや意味が含まれている場合、それを関連性があると評価してください。\n
        文書が質問に関連しているかどうかを示すために、バイナリスコア「yes」または「no」を与えてください。""",
        input_variables=["context", "question"],
    )

    chain = prompt | llm_with_tool

    messages = state["messages"]
    question = state["question"]

    last_message = messages[-1]
    docs = last_message.content
    
    docs_list = await parse_documents(docs)
    filtered_docs = []
    for d in docs_list:
        file_name = d.source
        file_name = os.path.basename(file_name.replace("\\","/"))
        page = d.page
        content = d.page_content
        score = chain.invoke(
            {"question": question, "context": content}
        )
        grade = score.binary_score
        if grade == "yes":
            filtered_docs.append(d)
            update_status_and_messages(
                "**GRADE: DOCUMENT RELEVANT**",
                additional_info=f"Source Added: {file_name}\nPage: {page}\nContent: {content}",
                state="complete",
            )
        else:
            update_status_and_messages(
                "**GRADE: DOCUMENT NOT RELEVANT**",
                additional_info=f"Source Rejected: {file_name}\nPage: {page}\nContent: {content}",
                state="error",
            )
    if len(filtered_docs) == 0:
        update_status_and_messages(
            "---DECISION: DOCS NOT RELEVANT---",
            state="error",
            expanded=False,
        )
        return "rewrite"
    else:
        update_status_and_messages(
            "---DECISION: DOCS RELEVANT---",
            expanded=False,
        )
        return Send("generate",{"filterd_docs": filtered_docs, "messages": messages,"question": question})
    
async def agent(state):
    """
    現在の状態に基づいて応答を生成するためにエージェントモデルを呼び出します。
    与えられた質問に対して、検索ツールを使用して情報を取得するか、
    単に終了するかを決定します。

    引数:
        state (messages): 現在の状態

    戻り値:
        dict: エージェントの応答がメッセージに追加された更新された状態
    """
    update_status_and_messages(
        "---CALL AGENT---",
        expanded=False,
    )
    messages = state["messages"]
    question = messages[-1].content
    model = ChatOpenAI(temperature=0, streaming=True, model="gpt-4o")
    model = model.bind_tools(tools)
    response = model.invoke(messages)
    state["messages"] = [response]
    state["question"] = question
    return state

async def rewrite(state):
    """
    クエリを変換してより良い質問を生成します。

    引数:
        state (messages): 現在の状態

    戻り値:
        dict: 言い換えられた質問を含む更新された状態
    """
    update_status_and_messages(
        "---TRANSFORM QUERY---",
        expanded=False,
    )
    messages = state["messages"]
    query = messages[-2].tool_calls[0]['args']['query']
    question = messages[-3].content

    msg = [
        HumanMessage(
            content=f"""このキーワードはベクトル化された社内規定に対してベクトル検索するために使用されます。
一回目のキーワードは、良いドキュメントを取得出来なかったので再挑戦です。
質問を見て、質問者の意図/意味について推論してより良いベクトル検索の為のキーワードを作成してください。
キーワードを1個のみ文字列として出力してください。
最初のキーワード:{query} 
ユーザーからの質問:{question}
改善されたキーワードを作成してください。 """,
        )
    ]

    model = ChatOpenAI(temperature=0, model="gpt-4o-mini", streaming=True)
    response = model.invoke(msg)
    update_status_and_messages(
        "**TRANSFORMED QUERY**",
        additional_info=f"Better Query: {response.content}",
        state="complete",
    )
    return {"messages": [response]}

async def generate(state):
    """
    回答を生成します

    引数:
        state (messages): 現在の状態

    戻り値:
         dict: 言い換えられた質問を含む更新された状態
    """
    st.session_state.placeholder.markdown("---GENERATE---")
    question = state["question"]

    filtered_docs = state["filterd_docs"]
    docs_str = ""
    for doc in filtered_docs:
        docs_str += f"Source: {doc.source}\nPage: {doc.page}\nContent: {doc.page_content}\n\n"
    prompt = ChatPromptTemplate.from_messages(
            [
                ("system", """あなたはとある会社の規定アシスタントです。以下の取得された文書を使用して社員からの質問に答えてください。
文書は質問に対し、ベクトル検索して取得されたものです。必ずしも最適な文書があるとは限りませんが、その結果を基に回答してください。
答えがわからない場合や文書が不十分な場合は、嘘の回答を作ったりせずに担当部署へ直接問い合わせるよう伝えてください。
回答に文書を参照した場合には、回答の最後には参照した文書名、ページ、文書の改定日を示してください。"""),
                ("human", """Question: {question} 
Context: {context}"""),
            ]
        )

    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, streaming=True)
    rag_chain = prompt | llm | StrOutputParser()
    response = rag_chain.invoke({"context": docs_str, "question": question})
    return {"messages": ("assistant",response)}

async def run_workflow(inputs):
    with st.status(label="**START**", expanded=True,state="running") as st.session_state.status:
        st.session_state.placeholder = st.empty()
        value = await st.session_state.workflow.ainvoke(inputs, config=st.session_state.config)
        st.session_state.placeholder.empty()
    update_status_and_messages(
        "**FINISH!!**",
        state="complete",
        expanded=False,
    )
    with st.chat_message("assistant", avatar=":material/psychology:"):
        st.session_state.answer = st.empty()
        last_response = value["messages"][-1].content
        if isinstance(last_response, list):
            st.session_state.answer.markdown(last_response[0]["text"])
        else:
            st.session_state.answer.markdown(last_response)
    st.session_state.value = value


if not hasattr(st.session_state, "workflow"):
    tools = [retriever]
    workflow = StateGraph(AgentState)

    workflow.add_node("agent", agent)  # agent
    retrieve = ToolNode([retriever])
    workflow.add_node("retrieve", retrieve)  # retrieval
    workflow.add_node("rewrite", rewrite)  # Re-writing the question
    workflow.add_node(
        "generate", generate
    )  
    workflow.add_edge(START, "agent")

    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "retrieve",
            END: END,
        },
    )

    workflow.add_conditional_edges(
        "retrieve",
        grade_documents,
    )
    workflow.add_edge("generate", END)
    workflow.add_edge("rewrite", "retrieve")
    memory = MemorySaver()
    st.session_state.config = {"configurable": {"thread_id": "debug"}}
    app = workflow.compile(checkpointer=memory)   
    app = app.with_config(recursion_limit=10,run_name="Agent",tags=["Agent","FB"])
    app.name = "Agent"
    st.session_state.workflow = app

if not hasattr(st.session_state, "value"):
    st.session_state.value = None

system_prompt = """あなたはとある株式会社の規定アシスタントです。ベクトルストアを使用して社員からの質問に答えてください。

ベクトルストアには社内規定が含まれています。以下の場合はベクトルストアを使用し質問に答えて下さい:
- 特定の条文の内容を引用・参照する質問。ただし、複数の条文の比較や解釈、外部情報との照合が必要な場合は回答しない。
- 特定の用語の定義に関する質問
- 社内規定やISO規定に直接関連する質問

以下の場合は 回答しないでください。:
- 広範囲に渡る調査や分析が必要な質問
- 複数のデータソースを参照・比較する必要がある質問
- 主観的な判断や創造性を必要とする質問
- 専門的な知識や経験が必要な質問
- 法律や倫理に関わる複雑な質問
- ベクトルストアの内容と明らかに無関係な質問

ベクトルストアを使用しない場合は以下の回答を提供してください。専門用語は使わずに回答すること。:
- 通常の会話: ユーザーが日常的な会話や雑談をしている場合は、それに合わせてフレンドリーに会話を続けてください。
- 曖昧な質問へのアドバイス: ユーザーの質問が曖昧で、何を求めているのか明確でない場合は、より具体的な質問をするように分かりやすくアドバイスしてください。
- 専門的な質問への対応: 専門知識が必要な質問に対しては、専門家や担当者への相談を促すなど、適切な対応を提案してください。
- 対応できない質問への対応: 倫理的な問題や、あなたの能力を超える質問に対しては、対応できない旨を分かりやすく伝えて下さい。

文書は質問に対し、ベクトル検索して取得されたものです。必ずしも最適な文書があるとは限りませんが、その結果を基に回答してください。
答えがわからない場合や文書が不十分な場合は、嘘の回答を作ったりせずに担当部署へ直接問い合わせるよう伝えてください。
回答に文書を参照した場合には、回答の最後には参照した文書名、ページ、文書の改定日を示してください
"""

st.title("自己修正Agentic RAG")
st.write("社内規定のAgentic RAGチャットボットです。")

if prompt := st.chat_input("質問を入力してください"):
    if st.session_state.value is not None:
        messages_history = st.session_state.value["messages"]
        for message in messages_history:
            if message.type == "human":
                with st.chat_message("user", avatar=":material/mood:"):
                    st.markdown(message.content)
            elif message.type == "ai":
                with st.chat_message("assistant", avatar=":material/psychology:"):
                    # if isinstance(message.content, list):
                    if message.content == "":
                        # st.markdown(message.content[0]["text"])
                        tool_calls = message.additional_kwargs['tool_calls'][0]
                        name = tool_calls['function']['name']
                        args = tool_calls['function']['arguments']
                        st.markdown(f"**ツール使用 : {name}**")
                        st.markdown(args)
                    else:
                        st.markdown(message.content)

    if st.session_state.value is not None:
        inputs = {
            "messages": [
                ("user", prompt),
            ]
        }
    else:
        inputs = {
            "messages": [
                ("system", system_prompt),
                ("user", prompt),
            ]
        }
    with st.chat_message("user", avatar=":material/mood:"):
        st.markdown(prompt)
    
    asyncio.run(run_workflow(inputs))