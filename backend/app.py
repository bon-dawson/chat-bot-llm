import os
import uvicorn
import tempfile
import asyncio
import json
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, WebSocket, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from bs4 import BeautifulSoup
import requests
import uuid
from datetime import datetime

# Set a user agent for LangChain requests
os.environ["USER_AGENT"] = "ChatbotApp/1.0"

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader, TextLoader
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="LangChain Chatbot API")

# Configure CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Initialize embedding model
embeddings = OpenAIEmbeddings()

# Initialize text splitter for document processing
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

# Streaming callback handler
class StreamingCallbackHandler(BaseCallbackHandler):
    def __init__(self, websocket):
        self.websocket = websocket
        self.response_text = ""

    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.response_text += token
        await self.websocket.send_text(json.dumps({"type": "token", "content": token}))

# Chatbot class
class ChatBot:
    class SessionHistory(BaseChatMessageHistory):
        """Custom chat message history implementation"""
        def __init__(self):
            self.messages = []

        def add_message(self, message):
            self.messages.append(message)

        def clear(self):
            self.messages = []

        def add_user_message(self, message: str) -> None:
            self.add_message(HumanMessage(content=message))

        def add_ai_message(self, message: str) -> None:
            self.add_message(AIMessage(content=message))

    def __init__(self):
        """Initialize the chatbot with chat history store and vector store."""
        self.sessions = {}  # Store chat histories by session ID
        self.vector_store = None
        self.qa_chain = None
        self.last_processed_pdf = None
        self.last_processed_url = None
        self.active_document_id = None
        self.documents = {}

        os.makedirs("./chroma_db", exist_ok=True)

        # Initialize vector store from persistent directory if it exists
        self.vector_store = Chroma(
            collection_name="chatbot_collection",
            embedding_function=embeddings,
            persist_directory="./chroma_db")
        self.initialize_qa_chain()

    def get_session_history(self, session_id):
        """Get or create a session history."""
        if session_id not in self.sessions:
            self.sessions[session_id] = self.SessionHistory()
        return self.sessions[session_id]

    def create_document_filter(self, document_id=None):
        """Create a search filter based on document ID."""
        if not document_id:
            return None

        def filter_func(doc):
            return doc.metadata.get("document_id") == document_id

        return filter_func

    def initialize_qa_chain(self, streaming_handler=None, document_id=None):
        """Initialize or update QA chain with optional streaming and document filtering."""
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            streaming=streaming_handler is not None,
            callbacks=[streaming_handler] if streaming_handler else None,
            verbose=True
        )

        # Create custom RAG prompt
        rag_prompt_template = """You are an assistant that answers questions based on provided context and your general knowledge.

        Context information:
        {context}

        Chat History:
        {chat_history}

        Human Question: {question}

        If the question asks for a summary or information about a document or URL that was previously uploaded, make sure to use the context information to provide a comprehensive answer.

        If the user asks to summarize a PDF or URL, provide a detailed summary of the content based on the context.

        When information is found in the context, incorporate it into your response with proper attribution.

        IMPORTANT: If the context doesn't contain relevant information to answer the question, DO NOT mention that you can't find information in the context. Instead, use your general knowledge to provide a helpful response. Never say that you don't have enough information or context - always attempt to give the most helpful answer possible using your general training.

        Provide accurate, helpful, and comprehensive responses.
        Answer:"""

        prompt = PromptTemplate.from_template(rag_prompt_template)

        # Create filter function for document ID if provided
        filter_dict = None
        if document_id:
            filter_dict = {"document_id": document_id}

        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 10,
                "filter": filter_dict  # Apply document filter
            }
        )

        # Create a chain without built-in memory
        qa_chain_no_memory = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            combine_docs_chain_kwargs={"prompt": prompt},
            return_source_documents=True,
            verbose=True
        )

        # Wrap with session-based message history
        self.qa_chain = RunnableWithMessageHistory(
            qa_chain_no_memory,
            lambda session_id: self.get_session_history(session_id),
            input_messages_key="question",
            history_messages_key="chat_history"
        )

    def initialize_vector_store(self, documents=None):
        """Initialize or update the vector store with documents."""
        if documents:
            texts = text_splitter.split_documents(documents)
            if len(texts) > 0:
                self.vector_store.add_documents(texts)
                return len(texts)
        return 0

    async def process_url(self, url: str) -> Dict[str, Any]:
        """Process URL content and add to vector store."""
        try:
            # Generate a unique ID for this document
            doc_id = str(uuid.uuid4())

            loader = WebBaseLoader(url)
            documents = loader.load()

            # Add URL and ID as metadata to each document
            for doc in documents:
                doc.metadata = {
                    "source": url,
                    "document_id": doc_id,
                    "type": "webpage"
                }

            chunks_count = self.initialize_vector_store(documents)

            # Get a small sample of the content for confirmation
            content_sample = documents[0].page_content[:200] + "..." if documents and documents[0].page_content else ""

            self.documents[doc_id] = {
                "type": "url",
                "name": url,
                "source": url,
                "timestamp": datetime.now().isoformat()
            }

            # Đặt PDF này làm tài liệu hoạt động
            self.active_document_id = doc_id
            self.last_processed_url = url

            return {
                "success": True,
                "message": f"Successfully processed URL with {chunks_count} chunks added to knowledge base.",
                "document_id": doc_id,
                "source": url,
                "content_sample": content_sample,
                "chunks_count": chunks_count
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing URL: {str(e)}")

    async def process_pdf(self, file_path: str, file_name: str) -> Dict[str, Any]:
        """Process PDF file and add to vector store."""
        try:
            # Generate a unique ID for this document
            doc_id = str(uuid.uuid4())

            loader = PyPDFLoader(file_path)
            documents = loader.load()

            # Add filename and ID as metadata
            for doc in documents:
                doc.metadata = {
                    "source": file_name,
                    "document_id": doc_id,
                    "type": "pdf",
                    "page": doc.metadata.get("page", 0)
                }

            chunks_count = self.initialize_vector_store(documents)

            # Get a small sample of the content for confirmation
            content_sample = documents[0].page_content[:200] + "..." if documents and documents[0].page_content else ""

            # Store the last processed PDF
            self.documents[doc_id] = {
                "type": "pdf",
                "name": file_name,
                "source": file_name,
                "timestamp": datetime.now().isoformat()
            }

            # Đặt PDF này làm tài liệu hoạt động
            self.active_document_id = doc_id
            self.last_processed_pdf = file_name

            return {
                "success": True,
                "message": f"Successfully processed PDF '{file_name}' with {chunks_count} chunks added to knowledge base.",
                "document_id": doc_id,
                "source": file_name,
                "content_sample": content_sample,
                "chunks_count": chunks_count
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing PDF: {str(e)}")

    async def process_text(self, text: str, source_name: str) -> Dict[str, Any]:
        """Process plain text and add to vector store."""
        try:
            # Generate a unique ID for this document
            doc_id = str(uuid.uuid4())

            # Create a temporary file to save the text
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as temp_file:
                temp_file.write(text)
                temp_path = temp_file.name

            # Load the text using TextLoader
            loader = TextLoader(temp_path)
            documents = loader.load()

            # Clean up the temporary file
            os.unlink(temp_path)

            # Add metadata to the document
            for doc in documents:
                doc.metadata = {
                    "source": source_name,
                    "document_id": doc_id,
                    "type": "text"
                }

            chunks_count = self.initialize_vector_store(documents)

            # Get a small sample of the content for confirmation
            content_sample = text[:200] + "..." if len(text) > 200 else text

            self.documents[doc_id] = {
                "type": "text",
                "name": source_name,
                "timestamp": datetime.now().isoformat()
            }

            # Đặt PDF này làm tài liệu hoạt động
            self.active_document_id = doc_id
            self.last_processed_pdf = source_name

            return {
                "success": True,
                "message": f"Successfully processed text '{source_name}' with {chunks_count} chunks added to knowledge base.",
                "document_id": doc_id,
                "source": source_name,
                "content_sample": content_sample,
                "chunks_count": chunks_count
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing text: {str(e)}")

    def get_knowledge_sources(self) -> List[Dict[str, Any]]:
        """Get a list of all knowledge sources in the vector store."""
        if not self.vector_store:
            return []

        # Get unique sources from the vector store
        sources = []
        try:
            all_metadatas = self.vector_store.get()["metadatas"]
            unique_sources = {}

            for metadata in all_metadatas:
                if metadata and "source" in metadata:
                    source = metadata["source"]
                    source_type = metadata.get("type", "unknown")

                    if source not in unique_sources:
                        unique_sources[source] = {
                            "name": source,
                            "type": source_type,
                            "count": 1
                        }
                    else:
                        unique_sources[source]["count"] += 1

            sources = list(unique_sources.values())
        except Exception as e:
            print(f"Error getting knowledge sources: {e}")

        return sources

# Initialize chatbot
chatbot = ChatBot()

# Pydantic models
class ChatMessage(BaseModel):
    message: str

class EventSearchRequest(BaseModel):
    query: str

class TextDocument(BaseModel):
    content: str
    name: str

# API routes
@app.post("/api/chat")
async def chat(message: ChatMessage, request: Request):
    """Non-streaming chat endpoint with context awareness"""
    try:
        # Generate a session ID from client information
        session_id = request.client.host

        # Extract URLs from message
        url_pattern = r'https?://[^\s]+'
        import re
        urls = re.findall(url_pattern, message.message)

        # Process any URLs found in the message first
        processed_urls = []
        for url in urls:
            result = await chatbot.process_url(url)
            chatbot.last_processed_url = url
            chatbot.active_document_id = result.get('document_id')

            # Save document info
            chatbot.documents[result.get('document_id')] = {
                "type": "webpage",
                "name": url,
                "source": url,
                "timestamp": datetime.now().isoformat()
            }

            processed_urls.append({
                "url": url,
                "chunks_count": result.get('chunks_count', 0),
                "document_id": result.get('document_id')
            })

        # Handle document context selection from user message
        document_id = None

        # Check if user explicitly wants to use the latest document
        if any(phrase in message.message.lower() for phrase in [
            "use latest document", "sử dụng tài liệu mới nhất",
            "dùng tài liệu mới nhất", "tài liệu gần đây"
        ]):
            document_id = chatbot.active_document_id

        # Check if user is asking about a PDF
        elif any(phrase in message.message.lower() for phrase in [
            "pdf", "tài liệu pdf", "file pdf", "trong pdf"
        ]) and chatbot.last_processed_pdf:
            # Find the ID of the most recent PDF
            for doc_id, doc_info in chatbot.documents.items():
                if doc_info["type"] == "pdf" and doc_info["name"] == chatbot.last_processed_pdf:
                    document_id = doc_id
                    break

        # Check if user is asking about a URL/webpage
        elif any(phrase in message.message.lower() for phrase in [
            "url", "trang web", "link", "website", "bài viết", "article"
        ]) and chatbot.last_processed_url:
            # Find the ID of the most recent URL
            for doc_id, doc_info in chatbot.documents.items():
                if doc_info["type"] == "webpage" and doc_info["source"] == chatbot.last_processed_url:
                    document_id = doc_id
                    break

        # Now handle the actual request
        if message.message.lower().startswith(("http://", "https://")) and len(message.message.split()) <= 2:
            # This is just a URL submission without a specific question
            # We've already processed it above, so just return confirmation
            url_info = processed_urls[0] if processed_urls else {"url": message.message, "chunks_count": 0}
            content = f"URL processed: {url_info['url']}. I've added {url_info['chunks_count']} paragraphs from this webpage to the database. You can ask about its content or request a summary by asking 'Please summarize this article'."
            return {
                "response": content,
                "active_document": chatbot.documents.get(chatbot.active_document_id, None)
            }

        # Check if the message is a summarization request
        elif any(phrase in message.message.lower() for phrase in [
            "tóm tắt", "summarize", "tóm lược", "tổng hợp", "tóm tắt nội dung",
            "summary", "summarize the content", "summarize the document",
            "summarize the article", "tóm tắt bài viết", "tóm tắt tài liệu"
        ]):
            # If it's a summarization request, create appropriate prompt
            summary_prompt = message.message

            # Determine which document to summarize
            target_document_id = document_id or chatbot.active_document_id

            # If no specific document was identified yet but we just processed URLs
            if not target_document_id and processed_urls and len(processed_urls) > 0:
                target_document_id = processed_urls[-1]["document_id"]
                latest_url = processed_urls[-1]["url"]
                summary_prompt = f"Hãy tóm tắt chi tiết nội dung của bài viết tại URL: {latest_url}"
            # If we have a specific document ID to use
            elif target_document_id:
                doc_info = chatbot.documents.get(target_document_id)
                if doc_info:
                    summary_prompt = f"Hãy tóm tắt chi tiết nội dung của: {doc_info['name']}"
            # Fall back to the most recently processed document
            elif not target_document_id:
                last_doc = chatbot.last_processed_pdf or chatbot.last_processed_url
                if last_doc:
                    summary_prompt = f"Hãy tóm tắt chi tiết nội dung của: {last_doc}"
                else:
                    summary_prompt = "Hãy tóm tắt chi tiết nội dung của tài liệu đã được tải lên gần đây nhất"

            # Create a filter for the retriever if we have a target document
            if not chatbot.qa_chain or target_document_id:
                chatbot.initialize_qa_chain(document_id=target_document_id)

            # Generate the response with the appropriate document filter
            response = chatbot.qa_chain.invoke(
                {"question": summary_prompt},
                config={"configurable": {"session_id": session_id}}
            )

            # Extract source documents information
            sources = []
            if "source_documents" in response:
                for doc in response["source_documents"]:
                    if doc.metadata and "source" in doc.metadata:
                        source = {
                            "name": doc.metadata["source"],
                            "type": doc.metadata.get("type", "unknown"),
                            "document_id": doc.metadata.get("document_id", "")
                        }
                        # Add page number for PDF documents
                        if "page" in doc.metadata:
                            source["page"] = doc.metadata["page"]
                        if source not in sources:
                            sources.append(source)

            # Determine which document was used for context
            used_document = None
            if target_document_id and target_document_id in chatbot.documents:
                used_document = chatbot.documents[target_document_id]

            return {
                "response": response["answer"],
                "sources": sources,
                "processed_urls": processed_urls if processed_urls else None,
                "active_document": used_document
            }
        else:
            # Regular question
            # Determine which document context to use
            target_document_id = document_id or chatbot.active_document_id

            # If we just processed URLs and no explicit document context was specified
            prefix = ""
            if processed_urls and not target_document_id:
                url_info = ", ".join([f"{u['url']} ({u['chunks_count']} chunks)" for u in processed_urls])
                prefix = f"I've processed these URLs: {url_info}. Now answering your question: "
                # Default to using the most recently processed URL
                target_document_id = processed_urls[-1]["document_id"]

            # Initialize or update the QA chain with the correct document filter
            if not chatbot.qa_chain or target_document_id:
                chatbot.initialize_qa_chain(document_id=target_document_id)

            # Generate the response
            response = chatbot.qa_chain.invoke(
                {"question": message.message},
                config={"configurable": {"session_id": session_id}}
            )

            # Extract source documents information if available
            sources = []
            if "source_documents" in response:
                for doc in response["source_documents"]:
                    if doc.metadata and "source" in doc.metadata:
                        source = {
                            "name": doc.metadata["source"],
                            "type": doc.metadata.get("type", "unknown"),
                            "document_id": doc.metadata.get("document_id", "")
                        }
                        # Add page number for PDF documents
                        if "page" in doc.metadata:
                            source["page"] = doc.metadata["page"]
                        if source not in sources:
                            sources.append(source)

            # Determine which document was used for context
            used_document = None
            if target_document_id and target_document_id in chatbot.documents:
                used_document = chatbot.documents[target_document_id]

            return {
                "response": prefix + response["answer"] if prefix else response["answer"],
                "sources": sources,
                "processed_urls": processed_urls if processed_urls else None,
                "active_document": used_document
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/api/chat/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Streaming chat endpoint via WebSocket with context awareness"""
    await websocket.accept()

    # Generate a session ID from client information
    session_id = f"ws_{id(websocket)}"

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)

            # Extract URLs from message
            url_pattern = r'https?://[^\s]+'
            import re
            urls = re.findall(url_pattern, message_data["message"])

            # Process any URLs found in the message first
            processed_urls = []
            if urls:
                await websocket.send_text(json.dumps({"type": "token", "content": "Phát hiện URL. Đang xử lý... "}))

            for url in urls:
                result = await chatbot.process_url(url)
                chatbot.last_processed_url = url
                chatbot.active_document_id = result.get('document_id')

                # Save document info
                chatbot.documents[result.get('document_id')] = {
                    "type": "webpage",
                    "name": url,
                    "source": url,
                    "timestamp": datetime.now().isoformat()
                }

                processed_urls.append({
                    "url": url,
                    "chunks_count": result.get('chunks_count', 0),
                    "document_id": result.get('document_id')
                })
                await websocket.send_text(json.dumps({"type": "token", "content": f"Đã xử lý URL: {url} ({result.get('chunks_count', 0)} đoạn). "}))

            # Handle document context selection from user message
            document_id = None

            # Check if user explicitly wants to use the latest document
            if any(phrase in message_data["message"].lower() for phrase in [
                "use latest document", "sử dụng tài liệu mới nhất",
                "dùng tài liệu mới nhất", "tài liệu gần đây"
            ]):
                document_id = chatbot.active_document_id
                await websocket.send_text(json.dumps({"type": "token", "content": "Đang sử dụng tài liệu mới nhất. "}))

            # Check if user is asking about a PDF
            elif any(phrase in message_data["message"].lower() for phrase in [
                "pdf", "tài liệu pdf", "file pdf", "trong pdf"
            ]) and chatbot.last_processed_pdf:
                # Find the ID of the most recent PDF
                for doc_id, doc_info in chatbot.documents.items():
                    if doc_info["type"] == "pdf" and doc_info["name"] == chatbot.last_processed_pdf:
                        document_id = doc_id
                        await websocket.send_text(json.dumps({"type": "token", "content": f"Đang tìm kiếm trong file PDF: {chatbot.last_processed_pdf}. "}))
                        break

            # Check if user is asking about a URL/webpage
            elif any(phrase in message_data["message"].lower() for phrase in [
                "url", "trang web", "link", "website", "bài viết", "article"
            ]) and chatbot.last_processed_url:
                # Find the ID of the most recent URL
                for doc_id, doc_info in chatbot.documents.items():
                    if doc_info["type"] == "webpage" and doc_info["source"] == chatbot.last_processed_url:
                        document_id = doc_id
                        await websocket.send_text(json.dumps({"type": "token", "content": f"Đang tìm kiếm trong trang web: {chatbot.last_processed_url}. "}))
                        break

            # Now handle the actual request
            if message_data["message"].lower().startswith(("http://", "https://")) and len(message_data["message"].split()) <= 2:
                # This is just a URL submission without a specific question
                # We've already processed it above, so just return confirmation
                url_info = processed_urls[0] if processed_urls else {"url": message_data["message"], "chunks_count": 0}
                content = f"URL đã được xử lý: {url_info['url']}. Tôi đã thêm {url_info['chunks_count']} đoạn từ trang web này vào cơ sở dữ liệu. Bạn có thể hỏi về nội dung hoặc yêu cầu tóm tắt bằng cách hỏi 'Hãy tóm tắt bài viết này'."

                # Send completion signal with active document info
                active_doc = None
                if chatbot.active_document_id in chatbot.documents:
                    active_doc = chatbot.documents[chatbot.active_document_id]

                await websocket.send_text(json.dumps({
                    "type": "complete",
                    "content": content,
                    "active_document": active_doc
                }))

            # Check if the message is a summarization request
            elif any(phrase in message_data["message"].lower() for phrase in [
                "tóm tắt", "summarize", "tóm lược", "tổng hợp", "tóm tắt nội dung",
                "summary", "summarize the content", "summarize the document",
                "summarize the article", "tóm tắt bài viết", "tóm tắt tài liệu"
            ]):
                # If it's a summarization request, create appropriate prompt
                summary_prompt = message_data["message"]

                # Determine which document to summarize
                target_document_id = document_id or chatbot.active_document_id

                # If no specific document was identified yet but we just processed URLs
                if not target_document_id and processed_urls and len(processed_urls) > 0:
                    target_document_id = processed_urls[-1]["document_id"]
                    latest_url = processed_urls[-1]["url"]
                    summary_prompt = f"Hãy tóm tắt chi tiết nội dung của bài viết tại URL: {latest_url}"
                # If we have a specific document ID to use
                elif target_document_id:
                    doc_info = chatbot.documents.get(target_document_id)
                    if doc_info:
                        summary_prompt = f"Hãy tóm tắt chi tiết nội dung của: {doc_info['name']}"
                # Fall back to the most recently processed document
                elif not target_document_id:
                    last_doc = chatbot.last_processed_pdf or chatbot.last_processed_url
                    if last_doc:
                        summary_prompt = f"Hãy tóm tắt chi tiết nội dung của: {last_doc}"
                    else:
                        summary_prompt = "Hãy tóm tắt chi tiết nội dung của tài liệu đã được tải lên gần đây nhất"

                await websocket.send_text(json.dumps({"type": "token", "content": "Đang tóm tắt nội dung... "}))

                # Generate streaming response
                streaming_handler = StreamingCallbackHandler(websocket)

                # Create a filter for the retriever if we have a target document
                chatbot.initialize_qa_chain(streaming_handler, document_id=target_document_id)

                response = await chatbot.qa_chain.ainvoke(
                    {"question": summary_prompt},
                    config={"configurable": {"session_id": session_id}}
                )

                # Extract source documents information
                sources = []
                if "source_documents" in response:
                    for doc in response["source_documents"]:
                        if doc.metadata and "source" in doc.metadata:
                            source = {
                                "name": doc.metadata["source"],
                                "type": doc.metadata.get("type", "unknown"),
                                "document_id": doc.metadata.get("document_id", "")
                            }
                            # Add page number for PDF documents
                            if "page" in doc.metadata:
                                source["page"] = doc.metadata["page"]
                            if source not in sources:
                                sources.append(source)

                # Determine which document was used for context
                used_document = None
                if target_document_id and target_document_id in chatbot.documents:
                    used_document = chatbot.documents[target_document_id]

                # Send completion signal with sources
                await websocket.send_text(json.dumps({
                    "type": "complete",
                    "content": streaming_handler.response_text,
                    "sources": sources,
                    "processed_urls": processed_urls if processed_urls else None,
                    "active_document": used_document
                }))
            else:
                # Regular question
                # Determine which document context to use
                target_document_id = document_id or chatbot.active_document_id

                # If we just processed URLs and no explicit document context was specified
                if processed_urls and not target_document_id:
                    url_info = ", ".join([f"{u['url']} ({u['chunks_count']} chunks)" for u in processed_urls])
                    await websocket.send_text(json.dumps({
                        "type": "token",
                        "content": f"Tôi đã xử lý các URL: {url_info}. Bây giờ trả lời câu hỏi của bạn: "
                    }))
                    # Default to using the most recently processed URL
                    target_document_id = processed_urls[-1]["document_id"]

                # Generate streaming response
                streaming_handler = StreamingCallbackHandler(websocket)

                # Initialize or update the QA chain with the correct document filter
                chatbot.initialize_qa_chain(streaming_handler, document_id=target_document_id)

                response = await chatbot.qa_chain.ainvoke(
                    {"question": message_data["message"]},
                    config={"configurable": {"session_id": session_id}}
                )

                # Extract source documents information
                sources = []
                if "source_documents" in response:
                    for doc in response["source_documents"]:
                        if doc.metadata and "source" in doc.metadata:
                            source = {
                                "name": doc.metadata["source"],
                                "type": doc.metadata.get("type", "unknown"),
                                "document_id": doc.metadata.get("document_id", "")
                            }
                            # Add page number for PDF documents
                            if "page" in doc.metadata:
                                source["page"] = doc.metadata["page"]
                            if source not in sources:
                                sources.append(source)

                # Determine which document was used for context
                used_document = None
                if target_document_id and target_document_id in chatbot.documents:
                    used_document = chatbot.documents[target_document_id]

                # Send completion signal with sources
                await websocket.send_text(json.dumps({
                    "type": "complete",
                    "content": streaming_handler.response_text,
                    "sources": sources,
                    "processed_urls": processed_urls if processed_urls else None,
                    "active_document": used_document
                }))
    except Exception as e:
        await websocket.send_text(json.dumps({"type": "error", "content": str(e)}))
    finally:
        await websocket.close()

@app.post("/api/upload-pdf")
async def upload_pdf(request: Request, file: UploadFile = File(...)):
    """Upload and process PDF file"""
    session_id = request.client.host

    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            # Read file content
            content = await file.read()
            # Write to temporary file
            temp_file.write(content)
            temp_path = temp_file.name

        # Process PDF
        result = await chatbot.process_pdf(temp_path, file.filename)

        # Clean up
        os.unlink(temp_path)

        return {
            "message": f"Successfully processed PDF '{file.filename}' with {result.get('chunks_count', 0)} chunks. You can now ask questions about its content.",
            "document_count": result.get('chunks_count', 0),
            "filename": file.filename
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/add-text")
async def add_text(document: TextDocument):
    """Add plain text to the knowledge base"""
    try:
        result = await chatbot.process_text(document.content, document.name)

        return {
            "message": f"Successfully processed text '{document.name}' with {result.get('chunks_count', 0)} chunks. You can now ask questions about its content.",
            "document_count": result.get('chunks_count', 0),
            "name": document.name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/search-events")
async def search_events(request: EventSearchRequest, req: Request):
    """Search for recent events"""
    session_id = req.client.host

    try:
        llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
        search_prompt = f"Find the latest information about: {request.query}. Focus on recent events and developments."
        response = llm.invoke(search_prompt)
        return {"response": response.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/knowledge-sources")
async def get_knowledge_sources():
    """Get a list of all knowledge sources in the vector store"""
    try:
        sources = chatbot.get_knowledge_sources()
        return {"sources": sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/clear-history/{session_id}")
async def clear_history(session_id: str):
    """Clear chat history for a session"""
    try:
        if session_id in chatbot.sessions:
            chatbot.sessions[session_id].clear()
            return {"message": f"Chat history cleared for session {session_id}"}
        return {"message": "Session not found or already empty"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run application
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
