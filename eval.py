import flet as ft
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.tools import tool
import os

os.chdir("/Users/abhigna/Desktop/work")

directory = "./chroma_db"
embeddings = OllamaEmbeddings(model="nomic-embed-text")
if os.path.exists(directory):
    vectorstore = Chroma(persist_directory=directory, embedding_function=embeddings)
else:
    files = [
        "Assignment 2.pdf",
        "Assignment 3.pdf",
        "Assignment 4.pdf"
    ]
    all_files = []
    for i in files:
        loader = PyPDFLoader(i)
        all_files.extend(loader.load())

    textsplit = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = textsplit.split_documents(all_files)
    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=directory
    )
  
retriever = vectorstore.as_retriever()

@tool
def retrieve_notes(query: str) -> str:
    """Search and return relevant information from Assignment 2,3 and 4"""
    info = retriever.invoke(query)
    return "\n\n".join([j.page_content for j in info])

model = init_chat_model("ollama:qwen3.5", temperature=0, streaming=True)

prompt= """You are quiz assistant" \
"Always call retrieve_notes before answering any question." \
"Answer only using the information returned from retrieve_notes" \
"Do not answer from general knowledge" \
"""
agent= create_agent(model=model, tools =[retrieve_notes], system_prompt= prompt)

tests= [
        {
            "question":
                "What is the purpose of Assignment 2?",
            "expected":
                "Gantt chart"
        },

        {
            "question":
                "What function detects circular dependencies?",
            "expected":
                "criticalPath"
        },

        {
            "question":
                "What helper function removes trailing spaces or newlines?",
            "expected":
                "remove_trailing"
        },

        {
            "question":
                "What data structures are used in Assignment 3?",
            "expected":
                "linked list"
        },

        {   
            "question":
                "How are Items connected?",
            "expected":
                "singly linked"
        },

        {
            "question":
                "How are Lists connected?",
            "expected":
                "doubly linked"
        },

        {
            "question":
                "What function loads a board from a file?",
            "expected":
                "load_file"
        },

        {
            "question":
                "What function calculates reading speed?",
            "expected":
                "calculate_reading_speed"
        },

        {
            "question":
                "What testing framework is used in Assignment 4?",
            "expected":
                "CUnit"
        },

        {
            "question":
                "Which two functions were unit tested?",
            "expected":
                "calculate_reading_speed"
        }
    ]

for i, test in enumerate(tests, start=1):

    response = agent.invoke(
        {
            "messages": [
                HumanMessage(content=test["question"])
            ]
        }
    )

    answer = response["messages"][-1].content

    passed = test["expected"].lower() in answer.lower()

    print(f"\nTest {i}")
    print("Question :", test["question"])
    print("Expected :", test["expected"])
    print("Answer   :", answer)
    print("Result   :", "PASS" if passed else "FAIL")
