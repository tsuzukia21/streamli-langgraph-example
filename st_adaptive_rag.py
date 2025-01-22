import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph
from typing import List, Literal, TypedDict
from langgraph.graph import END, StateGraph, START
import asyncio
import os
from langgraph.constants import Send

class RouteQuery(BaseModel):
    """ユーザーの質問をデータソースにルーティングするか判断します。"""

    choice: Literal["vectorstore", "no_tool_use"] = Field(
        ...,
        description="ユーザーの質問に対してデータソースにルーティングするかしないか",
    )
    reason: str = Field(
        ...,
        description="判断理由の説明",
    )
    answer: str = Field(
        ...,
        description="追加の回答情報（必要に応じて）",
    )
    query: str = Field(
        ...,
        description="ベクトルストアにルーティングする場合は、質問をベクトル検索に適したクエリを作成してください。",
    )

class GradeDocuments(BaseModel):
    """取得された文書の関連性チェックのためのバイナリスコア。"""
    binary_score: str = Field(
        description="文書が質問に関連しているかどうか、「yes」または「no」"
    )

class GradeHallucinations(BaseModel):
    """生成された回答における幻覚の有無を示すバイナリスコア。"""
    binary_score: str = Field(
        description="回答が事実に基づいているかどうか、「yes」または「no」"
    )

class GradeAnswer(BaseModel):
    """回答が質問に対処しているかどうかを評価するバイナリスコア。"""
    binary_score: str = Field(
        description="回答が質問に対処しているかどうか、「yes」または「no」"
    )

class GraphState(TypedDict):
    """
    グラフの状態を表します。
    属性:
        question: 質問
        query: 検索キーワード
        generation: LLM生成
        documents: 文書のリスト
    """
    question: str
    query: str
    generation: str
    documents: List[str]

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
    st.session_state.status_messages += message + "\n\n"
    if additional_info:
        st.session_state.status_messages += additional_info + "\n\n"

async def route_question(state):
    """
    質問を分析し、ベクトルストアを使用するか判定する関数
    vectorstore: 社内規定に関する質問
    no_tool_use: その他の質問
    """
    update_status_and_messages(
        "**---ROUTE QUESTION---**",
        expanded=False,
    )
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    structured_llm_router = llm.with_structured_output(RouteQuery)
    system = """ユーザーからの質問を分析し、ベクトルストアを使用するか判定してください。回答は以下の形式で提供してください：

1. choice: "vectorstore" または "no_tool_use" を選択
2. reason: 選択した理由を簡潔に説明
3. answer: 追加情報(必要な場合のみ)
4. query: ベクトルストアにルーティングする場合は、質問をベクトル検索に適した質問に変換してください。

ベクトルストア(vectorstore)には社内規定が含まれています。以下の場合は "vectorstore" を選択してください：
- 特定の条文の内容を引用・参照する質問。ただし、複数の条文の比較や解釈、外部情報との照合が必要な場合は、"no_tool_use" を選択する。
- 特定の用語の定義に関する質問
- 社内規定やに直接関連する質問

以下の場合は "no_tool_use" を選択してください：
- 広範囲に渡る調査や分析が必要な質問
- 複数のデータソースを参照・比較する必要がある質問
- 主観的な判断や創造性を必要とする質問
- 専門的な知識や経験が必要な質問
- 法律や倫理に関わる複雑な質問
- ベクトルストアの内容と明らかに無関係な質問

"no_tool_use"を選択した場合answerは以下の回答を提供してください。専門用語は使わずに回答すること。:
- 通常の会話: ユーザーが日常的な会話や雑談をしている場合は、それに合わせてフレンドリーに会話を続けてください。
- 曖昧な質問へのアドバイス: ユーザーの質問が曖昧で、何を求めているのか明確でない場合は、より具体的な質問をするように分かりやすくアドバイスしてください。
- 専門的な質問への対応: 専門知識が必要な質問に対しては、専門家や担当者への相談を促すなど、適切な対応を提案してください。
- 対応できない質問への対応: 倫理的な問題や、あなたの能力を超える質問に対しては、対応できない旨を分かりやすく伝えて下さい。

以下は例です：
question: 初年度に与えられる有給休暇は何日か教えてください。
choice: "vectorstore"
reason: "この質問では、社内規定に記載されている具体的な手順について尋ねています。"
answer: ""
query: "有給休暇 日数"

question: 以前のバージョンの社内規程からの変更点は何ですか？
choice: "no_tool_use"
reason: "この質問は、ベクトルストアの範囲外となる広範な調査が必要になる可能性があります。"
answer: "このアプリの範囲外となる広範な調査が必要になる可能性があります。直接担当者へ確認することをお勧めします。"
query: ""

question: 新車のおすすめは何ですか？
choice: "no_tool_use"
reason: "ベクトルストアの内容と明らかに無関係な質問です。"
answer: "このアプリの範囲外となる質問と推測されます。直接担当者へ確認することをお勧めします。"
query: ""
"""

    route_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )

    question_router = route_prompt | structured_llm_router

    question = state["question"]
    source = question_router.invoke({"question": "question:" + question})
    if source.choice  == "no_tool_use":
        update_status_and_messages(
            "NOT ROUTE QUESTION TO RAG",
        )
        return Send("no_tool_use",{"question": question, "generation": source.answer})
    elif source.choice  == "vectorstore":
        update_status_and_messages(
            "ROUTE QUESTION TO RAG",
            additional_info=f"Query: {source.query}"
        )
        return Send("retrieve",{"question": question,"query": source.query})

async def no_tool_use(state):
    return state

async def retrieve(state):
    """
    FAISSベクトルストアから関連文書を検索する関数
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    update_status_and_messages(
        "**---RETRIEVE---**",
        additional_info=f"RETRIEVING…\n\nKEY WORD:{state['query']}"
    )
    file="employment_rules"
    db=FAISS.load_local(file, embeddings,allow_dangerous_deserialization=True)

    retriever = db.as_retriever(search_kwargs={'k': 6})
    query = state["query"]
    documents = retriever.invoke(query)
    update_status_and_messages(
        "**RETRIEVE SUCCESS!!**",
        state="complete",
    )
    state["documents"] = documents
    return state

async def grade_documents(state):
    """
    検索された文書が質問に関連しているかを評価する関数
    2回目の試行までは文書の関連性をチェック
    """
    st.session_state.number_trial += 1
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    structured_llm_grader = llm.with_structured_output(GradeDocuments)

    system = """あなたは、ユーザーの質問に対して取得されたドキュメントの関連性を評価する採点者です。
ドキュメントにユーザーの質問に関連するキーワードや意味が含まれている場合、それを関連性があると評価してください。
目的は明らかに誤った取得を排除することです。厳密なテストである必要はありません。
ドキュメントが質問に関連しているかどうかを示すために、バイナリスコア「yes」または「no」を与えてください。"""
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )

    retrieval_grader = grade_prompt | structured_llm_grader
    update_status_and_messages(
        "**---CHECK DOCUMENT RELEVANCE TO QUESTION---**",
        expanded=False
    )       
    question = state["question"]
    documents = state["documents"]
    filtered_docs = []
    i = 0
    for d in documents:
        if st.session_state.number_trial <= 2:
            file_name = d.metadata["source"]
            file_name = os.path.basename(file_name.replace("\\","/"))
            page = d.metadata["page"]
            content = d.page_content
            i += 1
            score = retrieval_grader.invoke(
                {"question": question, "document": d.page_content}
            )
            grade = score.binary_score
            if grade == "yes":
                update_status_and_messages(
                    "**GRADE: DOCUMENT RELEVANT**",
                    additional_info=f"DOC {i}/{len(documents)} {file_name} page.{page} : **RELEVANT**\n\n{content}",
                    state="complete",
                )
                filtered_docs.append(d)
            else:
                update_status_and_messages(
                    "**GRADE: DOCUMENT NOT RELEVANT**",
                    state="error",
                    additional_info=f"DOC {i}/{len(documents)} {file_name} page.{page} : **NOT RELEVANT**\n\n{content}"
                )
        else:
            filtered_docs.append(d)

    if not st.session_state.number_trial <= 2:
        update_status_and_messages(
            "**NO NEED TO CHECK**",
            state="complete",
            expanded=False
        )
        update_status_and_messages(
            "**QUERY TRANSFORMATION HAS BEEN COMPLETED**",
            state="complete",
            expanded=False
        )
    state["documents"] = filtered_docs
    return state

async def generate(state):
    """
    検索された文書を基に回答を生成する関数
    """
    update_status_and_messages(
        "**---GENERATE---**",
        expanded=False
    )
    prompt = ChatPromptTemplate.from_messages(
            [
                ("system", """あなたはとある会社の社内規定アシスタントです。以下の取得された文書を使用して社員からの質問に答えてください。
文書は質問に対し、ベクトル検索して取得されたものです。必ずしも最適な文書があるとは限りませんが、その結果を基に回答してください。
答えがわからない場合や文書が不十分な場合は、嘘の回答を作ったりせずに担当部署へ直接問い合わせるよう伝えてください。
回答に文書を参照した場合には、回答の最後には参照した文書名、ページ、文書の改定日を示してください。"""),
                ("human", """Question: {question} 
Context: {context}"""),
            ]
        )
        
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

    rag_chain = prompt | llm | StrOutputParser()
    question = state["question"]
    documents = state["documents"]
    generation = rag_chain.invoke({"context": documents, "question": question})
    state["generation"] = generation
    return state

async def transform_query(state):
    """
    検索クエリを最適化して再検索を行う関数
    より良い検索結果を得るためにクエリを書き換える
    """
    update_status_and_messages(
        "**---TRANSFORM QUERY---**",
        expanded=True
    )
    st.session_state.placeholder.empty()
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    system = """あなたは、検索キーワードをベクトルストア検索に最適化されたより良いバージョンに変換するキーワードリライターです。
このキーワードはベクトル化された社内規定に対してベクトル検索するために使用されます。
一回目のキーワードは、良いドキュメントを取得出来なかったので再挑戦です。
質問を見て、質問者の意図/意味について推論してより良いベクトル検索の為のキーワードを作成してください。
キーワードを1個のみ文字列として出力してください。
"""
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "最初のキーワード: \n{query}\n\nユーザーからの質問:\n{question}\n\n改善されたキーワードを作成してください。",
            ),
        ]
    )

    question_rewriter = re_write_prompt | llm | StrOutputParser()
    question = state["question"]
    query = state["query"]
    better_question = question_rewriter.invoke({"query": query ,"question": question})
    update_status_and_messages(
        f"**Better question: {better_question}**",
        state="complete",
    )
    state["query"] = better_question
    return state

async def decide_to_generate(state):
    """
    文書が見つからない場合はクエリを変換し、
    文書が見つかった場合は回答生成に進む判断を行う関数
    """
    filtered_documents = state["documents"]
    if not filtered_documents:
        update_status_and_messages(
            "**DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY**",
            state="error",
            expanded=False
        )
        return "transform_query"                                     
    else:
        update_status_and_messages(
            "**DECISION: GENERATE**",
            state="complete",
            expanded=False
        )
        return "generate"

async def grade_generation_v_documents_and_question(state):
    """
    生成された回答の品質を評価する関数
    - 文書に基づいているか（幻覚がないか）
    - 質問に適切に答えているか
    をチェック
    """
    st.session_state.number_trial += 1  
    update_status_and_messages(
        "**---CHECK HALLUCINATIONS---**",
        expanded=False
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    structured_llm_grader = llm.with_structured_output(GradeHallucinations)

    system = """あなたは、LLMの生成が取得された事実のセットに基づいているか/サポートされているかを評価する採点者です。
バイナリスコア「yes」または「no」を与えてください。「yes」は、回答が事実のセットに基づいている/サポートされていることを意味します。"""
    hallucination_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
        ]
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    structured_llm_grader = llm.with_structured_output(GradeAnswer)

    system = """あなたは、回答が質問に対処しているか/解決しているかを評価する採点者です。
バイナリスコア「yes」または「no」を与えてください。「yes」は、回答が質問を解決していることを意味します。"""
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
        ]
    )

    answer_grader = answer_prompt | structured_llm_grader
    hallucination_grader = hallucination_prompt | structured_llm_grader
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score.binary_score
    if st.session_state.number_trial <= 3:
        if grade == "yes":
            update_status_and_messages(
                "**DECISION: ANSWER IS BASED ON A SET OF FACTS**",
                state="complete",
            )
            update_status_and_messages(
                "**---GRADE GENERATION vs QUESTION---**",
            )
            score = answer_grader.invoke({"question": question, "generation": generation})
            grade = score.binary_score
            if grade == "yes":
                update_status_and_messages(
                    "**DECISION: GENERATION ADDRESSES QUESTION**",
                    state="complete",
                )
                with st.session_state.placeholder:
                    update_status_and_messages(
                        "**USEFUL!!**",
                        additional_info=f"question : {question}\n\ngeneration : {generation}",
                        state="complete",
                    )
                return "useful"
            else:
                st.session_state.number_trial -= 1
                update_status_and_messages(
                    "**DECISION: GENERATION DOES NOT ADDRESS QUESTION**",
                    additional_info=f"question:{question}\n\ngeneration:{generation}",
                    state="error",
                )
                with st.session_state.placeholder:
                    update_status_and_messages(
                        "**NOT USEFUL**",
                        additional_info=f"question:{question}\n\ngeneration:{generation}",
                        state="error",
                    )
                return "not useful"
        else:
            message = "**DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY**"
            st.session_state.status.update(label=f"{message}", state="error", expanded=True)
            with st.session_state.placeholder:
                update_status_and_messages(
                    "**NOT GROUNDED**",
                    additional_info=f"question:{question}\n\ngeneration:{generation}",
                    state="error",
                )
            return "not supported"
    else:
        update_status_and_messages(
            "**NO NEED TO CHECK**",
            expanded=False,
            state="complete",
        )
        update_status_and_messages(
            "**TRIAL LIMIT EXCEEDED**",
            expanded=False,
            state="complete",
        )
        return "useful"

async def run_workflow(inputs):
    """
    全体のワークフローを実行する関数
    状態管理とUIの更新を行う
    """
    st.session_state.number_trial = 0
    with st.status(label="**GO!!**", expanded=True,state="running") as st.session_state.status:
        st.session_state.placeholder = st.empty()
        value = await st.session_state.workflow.ainvoke(inputs)

    st.session_state.placeholder.empty()
    st.session_state.answer = st.empty()
    update_status_and_messages(
        "**FINISH!!**",
        state="complete",
        expanded=False,
    )
    st.session_state.answer.markdown(value["generation"])
    with st.popover("ログ"):
        st.markdown(st.session_state.status_messages)

# メインのStreamlitアプリケーション部分
if 'status_messages' not in st.session_state:
    st.session_state.status_messages = ""

# ワークフローの初期化（初回のみ実行）
if not hasattr(st.session_state, "workflow"):
    # StateGraphの設定
    workflow = StateGraph(GraphState)
    # ノードの追加
    workflow.add_node("no_tool_use", no_tool_use)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("transform_query", transform_query)

    # エッジの追加（フローの制御）
    workflow.add_conditional_edges(
        START,
        route_question,
        {
            "retrieve": "retrieve",
            "no_tool_use": "no_tool_use",
        },
    )
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_edge("no_tool_use", END)
    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            "not supported": "transform_query",
            "useful": END,
            "not useful": "transform_query",
        },
    )
    app = workflow.compile()
    app = app.with_config(recursion_limit=20,run_name="Agent",tags=["Agent","FB"])
    app.name = "Agent"
    st.session_state.workflow = app

st.title("自己修正RAG")
st.write("社内規定に関する**一問一答形式**のチャットボットです。")

if prompt := st.chat_input("質問を入力してください"):
    st.session_state.status_messages = ""
    with st.chat_message("user", avatar="😊"):
        st.markdown(prompt)

    inputs = {"question": prompt}
    asyncio.run(run_workflow(inputs))