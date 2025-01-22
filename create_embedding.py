from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from glob import glob
from typing import Any
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import PyPDFium2Loader

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

text_splitter = SemanticChunker(
    embeddings, breakpoint_threshold_type="standard_deviation",sentence_split_regex=r"(?<=[、。【])\s+",buffer_size=1
)

file = "https://www.mhlw.go.jp/content/001018385.pdf"

loader = PyPDFium2Loader(file)
document = loader.load()

docs = text_splitter.split_documents(document)

texts = ""
print(len(docs))
for doc in docs:
    texts += doc.page_content+"\n\n///////////////////////////////"
    # print(doc.page_content)
print(texts)

db = FAISS.from_documents(docs, embeddings)
db.save_local("employment_rules")
