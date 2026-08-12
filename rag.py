#RAG Pipeline 

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.tools import tool
import os

#backend 

"""Tracing
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource

resource= Resource.create({"service.name": "Voice Agent"})
provider= TracerProvider(resource=resource)
trace.set_tracer_provider(provider)
export= OTLPSpanExporter(
    endpoint="localhost:4317",
    insecure=True
)
provider.add_span_processor(
    BatchSpanProcessor(export)
)

tracer = trace.get_tracer("RAG Voice Agent.py")
"""

#Phoenix
#http://localhost:6006
"""from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import trace as otel_trace

tracer_provider = register(project_name="voice-quiz-agent")
LangChainInstrumentor().instrument(tracer_provider=tracer_provider)"""


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