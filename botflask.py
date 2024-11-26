from fastapi import FastAPI, Request
from pydantic import BaseModel
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from sentence_transformers import SentenceTransformer, util
from langchain.embeddings import HuggingFaceEmbeddings

# FastAPI App
app = FastAPI()

# Chatbot Session State
session_store = {}

# Define chatbot input structure
class ChatRequest(BaseModel):
    user_input: str
    session_id: str

# Prepare and split documents from a specified directory
def prepare_and_split_docs(pdf_directory):
    loader = DirectoryLoader(pdf_directory, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=512,
        chunk_overlap=256,
        disallowed_special=(),
        separators=["\n\n", "\n", " "]
    )
    return splitter.split_documents(documents)

# Function to ingest documents into the vector database
def ingest_into_vectordb(split_docs):
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return FAISS.from_documents(split_docs, embeddings)

# Function to get the conversation chain
def get_conversation_chain(retriever):
    llm = Ollama(model="llama3.2")
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Provide concise responses based on provided documents."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, contextualize_q_prompt)
    retriever_chain = create_retrieval_chain(retriever, question_answer_chain)

    def get_session_history(session_id: str):
        if session_id not in session_store:
            session_store[session_id] = ChatMessageHistory()
        return session_store[session_id]

    return RunnableWithMessageHistory(
        retriever_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

# Initialize chatbot backend
pdf_directory = "Data/SalesDocs"  # Path to your PDF folder
split_docs = prepare_and_split_docs(pdf_directory)
vector_db = ingest_into_vectordb(split_docs)
retriever = vector_db.as_retriever()
chatbot_chain = get_conversation_chain(retriever)

@app.post("/chat")
async def chat_endpoint(chat_request: ChatRequest):
    """
    Chatbot endpoint to process user inputs and return bot responses.
    """
    user_input = chat_request.user_input.lower()
    session_id = chat_request.session_id

    # Check for greetings or thank-you messages
    greetings = ["hi", "hello", "hey"]
    thanks = ["thank you", "thanks", "thank you very much", "thx"]
    if any(greet in user_input for greet in greetings):
        return {"response": "Hello! How can I assist you?"}
    elif any(thank in user_input for thank in thanks):
        return {"response": "You're welcome! Let me know if you have any more questions."}

    # Process query with chatbot chain
    response = chatbot_chain.invoke({"input": user_input}, config={"configurable": {"session_id": session_id}})
    return {"response": response.get("answer", "I'm sorry, I couldn't find an answer to your question.")}
