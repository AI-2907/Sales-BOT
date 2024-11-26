
# +++++++++++++++++++++++++++++++++++++++++
import streamlit as st
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.chains import create_history_aware_retriever
from langchain.embeddings import HuggingFaceEmbeddings
import os

# Define template for bot and user messages
bot_template = '''
<div style="display: flex; align-items: center; margin-bottom: 10px;">
    <div style="flex-shrink: 0; margin-right: 10px;">
        <img src="https://uxwing.com/wp-content/themes/uxwing/download/communication-chat-call/answer-icon.png" 
             style="max-height: 50px; max-width: 50px; border-radius: 50%; object-fit: cover;">
    </div>
    <div style="background-color: #f1f1f1; padding: 10px; border-radius: 10px; max-width: 75%; word-wrap: break-word; overflow-wrap: break-word;">
        {msg}
    </div>
</div>
'''

user_template = '''
<div style="display: flex; align-items: center; margin-bottom: 10px; justify-content: flex-end;">
    <div style="flex-shrink: 0; margin-left: 10px;">
        <img src="https://cdn.iconscout.com/icon/free/png-512/free-q-characters-character-alphabet-letter-36051.png?f=webp&w=512" 
             style="max-height: 50px; max-width: 50px; border-radius: 50%; object-fit: cover;">
    </div>    
    <div style="background-color: #007bff; color: white; padding: 10px; border-radius: 10px; max-width: 75%; word-wrap: break-word; overflow-wrap: break-word;">
        {msg}
    </div>
</div>
'''

# Function to prepare and split documents from a specified directory
def prepare_and_split_docs(documents):
    split_docs = []
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=512,
        chunk_overlap=256,
        disallowed_special=(),
        separators=["\n\n", "\n", " "]
    )
    split_docs = splitter.split_documents(documents)
    return split_docs

# Function to ingest documents into the vector database
def ingest_into_vectordb(split_docs):
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.from_documents(split_docs, embeddings)
    return db

# Function to get the conversation chain
def get_conversation_chain(retriever):
    llm = Ollama(model="llama3.2")
    
    contextualize_q_system_prompt = (
        "Given the chat history and the latest user question, "
        "provide a response that directly addresses the user's query based on the provided documents. "
        "If no relevant answer is found, respond with: "
        "'I'm sorry, but I couldn’t find an answer to your question. Could you rephrase or provide more details?' "
        "Do not rephrase the question or ask follow-up questions."
    )
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    
    system_prompt = (
        "As a chat assistant, provide accurate and relevant information based on the provided documents in 2-3 sentences. "
        "If no relevant information is found, respond with: "
        "'I'm sorry, but I couldn’t find an answer to your question. Could you rephrase or provide more details?' "
        "{context}"
    )
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    return conversational_rag_chain

# Main Streamlit app
logopath=r"C:\Users\Akshada.P\OneDrive - SAKSOFT LIMITED\Documents\SAKSOFT Internal BOT\saksoftimg.png"
st.image(logopath,width=200)
st.title("Sales Bot")
st.write("Welcome! How can I help you today?")

# Define the path to your PDF directory
# pdf_directory = "Data/SalesDocs"  # Update with your actual PDF folder path

# Prepare and ingest documents from the initial folder
# loaded_docs = []
# if os.path.exists(pdf_directory):
#     loader = DirectoryLoader(pdf_directory, glob="**/*.pdf", loader_cls=PyPDFLoader)
#     loaded_docs = loader.load()
# split_docs = prepare_and_split_docs(loaded_docs)
# vector_db = ingest_into_vectordb(split_docs)
# retriever = vector_db.as_retriever()

# Initialize the conversation chain
# conversational_chain = get_conversation_chain(retriever)
# st.session_state.conversational_chain = conversational_chain

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# File upload for additional documents
uploaded_files = st.file_uploader("Upload additional PDF documents", type="pdf", accept_multiple_files=True)

if uploaded_files:
    uploaded_docs = []
    for uploaded_file in uploaded_files:
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        doc_loader = PyPDFLoader(uploaded_file.name)
        uploaded_docs.extend(doc_loader.load())

    all_docs =  uploaded_docs
    split_docs = prepare_and_split_docs(all_docs)
    vector_db = ingest_into_vectordb(split_docs)
    retriever = vector_db.as_retriever()
    conversational_chain = get_conversation_chain(retriever)
    st.session_state.conversational_chain = conversational_chain

# Chat input
user_input = st.text_input("Ask a question about the documents:")

# Check for greetings, thanks, yes/no
greetings = ["hi", "hello", "hey"]
thanks = ["thank you", "thanks", "thank you very much", "thx"]
yes_no = ["yes", "no"]

if st.button("Submit"):
    user_input_lower = user_input.lower()  # Normalize input for checking

    # Respond to greetings
    if any(greet in user_input_lower for greet in greetings):
        response = "Hello! How can I assist you?"
        st.session_state.chat_history.append({"user": user_input, "bot": response, "context_docs": []})

    # Respond to thanks
    elif any(thank in user_input_lower for thank in thanks):
        response = "You're welcome! If you have more questions, just let me know."
        st.session_state.chat_history.append({"user": user_input, "bot": response, "context_docs": []})

    # Respond to yes/no
    elif any(yn in user_input_lower for yn in yes_no):
        response = "I see! If you have any further questions, feel free to ask."
        st.session_state.chat_history.append({"user": user_input, "bot": response, "context_docs": []})

    # Process document queries
    elif user_input and 'conversational_chain' in st.session_state:
        session_id = "user123"  # Static session ID for this demo; you can make it dynamic if needed
        conversational_chain = st.session_state.conversational_chain
        response = conversational_chain.invoke({"input": user_input}, config={"configurable": {"session_id": session_id}})
        context_docs = response.get('context', [])

        # Check if the response is the default message
        if response['answer'] == "I'm sorry, but I couldn’t find an answer to your question. Could you rephrase or provide more details?":
            st.session_state.chat_history.append({"user": user_input, "bot": response['answer'], "context_docs": []})
        else:
            st.session_state.chat_history.append({"user": user_input, "bot": response['answer'], "context_docs": context_docs})

# Display chat history
if st.session_state.chat_history:
    for message in st.session_state.chat_history:
        st.markdown(user_template.format(msg=message['user']), unsafe_allow_html=True)
        st.markdown(bot_template.format(msg=message['bot']), unsafe_allow_html=True)

        # Display the source documents if available
        if message.get('context_docs'):
            with st.expander("Source Documents"):
                for doc in message['context_docs']:
                    st.write(f"Source: [{doc.metadata['source']}]({doc.metadata['source']})")
                    st.write(doc.page_content)
# -----------------------------
# import streamlit as st
# from langchain.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.llms import Ollama
# from langchain.vectorstores import FAISS
# from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains import create_history_aware_retriever
# from langchain.embeddings import HuggingFaceEmbeddings
# import os

# # Define templates for bot and user messages
# bot_template = '''
# <div style="display: flex; align-items: center; margin-bottom: 10px;">
#     <div style="flex-shrink: 0; margin-right: 10px;">
#         <img src="https://uxwing.com/wp-content/themes/uxwing/download/communication-chat-call/answer-icon.png" 
#              style="max-height: 50px; max-width: 50px; border-radius: 50%; object-fit: cover;">
#     </div>
#     <div style="background-color: #f1f1f1; padding: 10px; border-radius: 10px; max-width: 75%; word-wrap: break-word; overflow-wrap: break-word;">
#         {msg}
#     </div>
# </div>
# '''

# user_template = '''
# <div style="display: flex; align-items: center; margin-bottom: 10px; justify-content: flex-end;">
#     <div style="flex-shrink: 0; margin-left: 10px;">
#         <img src="https://cdn.iconscout.com/icon/free/png-512/free-q-characters-character-alphabet-letter-36051.png?f=webp&w=512" 
#              style="max-height: 50px; max-width: 50px; border-radius: 50%; object-fit: cover;">
#     </div>    
#     <div style="background-color: #007bff; color: white; padding: 10px; border-radius: 10px; max-width: 75%; word-wrap: break-word; overflow-wrap: break-word;">
#         {msg}
#     </div>
# </div>
# '''

# # Function to prepare and split documents
# def prepare_and_split_docs(documents):
#     splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
#         chunk_size=512,
#         chunk_overlap=256,
#         disallowed_special=(),
#         separators=["\n\n", "\n", " "]
#     )
#     return splitter.split_documents(documents)

# # Function to ingest documents into the vector database
# def ingest_into_vectordb(split_docs):
#     embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
#     return FAISS.from_documents(split_docs, embeddings)

# # Function to get the conversation chain
# def get_conversation_chain(retriever):
#     llm = Ollama(model="llama3.2")
    
#     contextualize_q_prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", (
#                 "Given the chat history and the latest user question, provide a response that directly addresses the user's query "
#                 "based on the provided documents. If no relevant answer is found, respond with: "
#                 "'I'm sorry, but I couldn’t find an answer to your question. Could you rephrase or provide more details?'"
#             )),
#             MessagesPlaceholder("chat_history"),
#             ("human", "{input}"),
#         ]
#     )
    
#     history_aware_retriever = create_history_aware_retriever(
#         llm, retriever, contextualize_q_prompt
#     )
    
#     qa_prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", (
#                 "As a chat assistant, provide accurate and relevant information based on the provided documents in 2-3 sentences. "
#                 "If no relevant information is found, respond with: "
#                 "'I'm sorry, but I couldn’t find an answer to your question. Could you rephrase or provide more details?'"
#             )),
#             MessagesPlaceholder("chat_history"),
#             ("human", "{input}"),
#         ]
#     )
    
#     question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
#     rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

#     store = {}

#     def get_session_history(session_id: str):
#         if session_id not in store:
#             store[session_id] = ChatMessageHistory()
#         return store[session_id]

#     return RunnableWithMessageHistory(
#         rag_chain,
#         get_session_history,
#         input_messages_key="input",
#         history_messages_key="chat_history",
#         output_messages_key="answer",
#     )

# # Main Streamlit app
# st.title("Sales Bot")
# st.write("Welcome! How can I help you today?")

# # File upload for PDF documents
# uploaded_files = st.file_uploader("Upload PDF documents", type="pdf", accept_multiple_files=True)

# if 'vector_db' not in st.session_state:
#     st.session_state.vector_db = None
# if 'conversational_chain' not in st.session_state:
#     st.session_state.conversational_chain = None
# if 'chat_history' not in st.session_state:
#     st.session_state.chat_history = []

# if uploaded_files:
#     uploaded_docs = []
#     for uploaded_file in uploaded_files:
#         with open(uploaded_file.name, "wb") as f:
#             f.write(uploaded_file.getbuffer())
#         doc_loader = PyPDFLoader(uploaded_file.name)
#         uploaded_docs.extend(doc_loader.load())
    
#     split_docs = prepare_and_split_docs(uploaded_docs)
#     vector_db = ingest_into_vectordb(split_docs)
#     retriever = vector_db.as_retriever()
#     conversational_chain = get_conversation_chain(retriever)
    
#     st.session_state.vector_db = vector_db
#     st.session_state.conversational_chain = conversational_chain

# # Chat input
# user_input = st.text_input("Ask a question about the documents:")

# if st.button("Submit") and user_input:
#     if st.session_state.conversational_chain:
#         response = st.session_state.conversational_chain.invoke({"input": user_input})
#         answer = response.get('answer', "I'm sorry, I couldn't find an answer to your question.")
#         st.session_state.chat_history.append({"user": user_input, "bot": answer})

# # Display chat history
# if st.session_state.chat_history:
#     for message in st.session_state.chat_history:
#         st.markdown(user_template.format(msg=message['user']), unsafe_allow_html=True)
#         st.markdown(bot_template.format(msg=message['bot']), unsafe_allow_html=True)
