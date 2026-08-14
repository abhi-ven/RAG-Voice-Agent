#RAG Pipeline 

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.tools import tool
from langchain_community.document_loaders import WebBaseLoader
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
    all_documents= []
    for i in files:
        loader = PyPDFLoader(i)
        all_documents.extend(loader.load())

    #for website links 
    websites= [
        "https://pubs.rsna.org/doi/10.1148/radiol.260264",
        "https://pubs.rsna.org/doi/10.1148/radiol.253979",
        "https://pubs.rsna.org/doi/10.1148/radiol.253711"
    ]

    for j in websites:
        loader2= WebBaseLoader(j)
        all_documents.extend(loader2.load())

    textsplit= RecursiveCharacterTextSplitter(
        chunk_size= 1000, 
        chunk_overlap= 200
    )
    chunks= textsplit.split_documents(all_documents)
    vectorstore= Chroma.from_documents(
        documents= chunks, embedding= embeddings, persist_directory= directory
    )

retriever = vectorstore.as_retriever()

@tool
def retrieve_notes(query: str) -> str:
    """Search and return relevant information from Assignment 2,3 and 4 and the websites linked"""
    info = retriever.invoke(query)
    return "\n\n".join([j.page_content for j in info])