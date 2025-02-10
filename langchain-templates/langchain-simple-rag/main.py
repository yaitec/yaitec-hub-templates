# This will load our .env variables into our python environment.
from dotenv import load_dotenv

# Import document loader and text splitter (which will spllit our PDF document into smaller chunks)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import language model and embeddings modules
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

# Import chain and prompt-related components (follow docs at: https://python.langchain.com/v0.2/docs/tutorials/pdf_qa/)
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables from .env file if needed
load_dotenv()

# Initialize the language model with the specified model (gpt-4o)
llm = ChatOpenAI(model="gpt-4o-mini")

# Specify the path to the PDF document to be processed
file_path = "./library/nke-10k-2023.pdf"
print(f"Loading PDF document from path: {file_path}")

# Create a PDF loader instance and load the document (an object that represents the document)
loader = PyPDFLoader(file_path)
docs = loader.load()
print(f"Document loaded successfully with {len(docs)} page(s).")

# Display a preview of the first page's content and its metadata
if docs:
    preview = docs[0].page_content[0:100]
    metadata = docs[0].metadata
    print(f"Preview of first page content (first 100 chars): {preview}")
    print(f"Metadata of first page: {metadata}")
else: # if no documents were detected/loaded
    print("No documents were loaded from the PDF.")

# Split the document into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
print(f"Document has been split into {len(splits)} chunk(s).")

# Create a vector store from the document chunks using embeddings (vector representations)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
print("Vector store created successfully.")

# Prepare the retriever from the vector store for later use in retrieval
retriever = vectorstore.as_retriever()

# Create a custom prompt template for the question-answering chain
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
# Create the document question-answering chain using the language model and prompt
question_answer_chain = create_stuff_documents_chain(llm, prompt)

# Combine the retriever with the question-answering chain to form a RAG chain
rag_chain = create_retrieval_chain(retriever, question_answer_chain)
print("Retrieval augmented generation (RAG) chain is ready.")

# Asks/Wait for an user input and run the chain to answer the user's question (ps. \n skips a line)
user_input = input("\n\nType your question about the document: ")

# Invoke the RAG chain with the user's question to generate an answer
results = rag_chain.invoke({"input": str(user_input)}) # e.g: "What is the company's revenue?"}

# Display the final answer to the user
print(f"\nFinal Answer: {results['answer']}")