# Import LangChain components from updated module paths
from langchain_community.vectorstores import FAISS  # Updated FAISS import
from langchain_openai import OpenAIEmbeddings  # Updated OpenAI embeddings import
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader  # Updated document loaders
from langchain_openai import ChatOpenAI  # Updated ChatOpenAI import
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, Tool, AgentType

# Load environment variables from .env file if needed
from dotenv import load_dotenv

load_dotenv()

# Document Loading (unchanged)
file_path = "./library/nke-10k-2023.pdf"
print(f"Loading PDF document from path: {file_path}...")
pdf_loader = PyPDFLoader(file_path)
pdf_documents = pdf_loader.load()
print(f"PDF loaded successfully with {len(pdf_documents)} page(s).")

# Document Consolidation (unchanged)
all_documents = pdf_documents

# Vector Store Creation (unchanged except for updated OpenAIEmbeddings)
print("Creating embeddings and building the vector store...")
embeddings = OpenAIEmbeddings()  # Now from langchain_openai
vector_store = FAISS.from_documents(
    documents=all_documents,
    embedding=embeddings
)
print("Vector store created successfully.")

# Retrieval System Setup (unchanged)
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_k=6
)
print("Retriever initialized.")

# Language Model Configuration (updated ChatOpenAI)
llm = ChatOpenAI(model_name="gpt-4o-mini")  # Now from langchain_openai

# QA Chain Configuration (unchanged)
retrieval_qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# Tool Definition (unchanged)
document_tool = Tool(
    name="Document Retrieval",
    func=lambda q: retrieval_qa_chain({"query": q})["result"],
    description="Retrieve knowledge from the document database."
)

# Agent Initialization (unchanged)
tools = [document_tool]
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Interactive Query Processing (unchanged)
user_query = input("\n\nType your question about the document: ") # e.g: "What is the company's revenue?"}

# you will be able to see agent logs on your terminal (pay attention on the actions and thoughts!)
response = agent.invoke(user_query)

print(f"\nFinal Answer: {response['output']}")