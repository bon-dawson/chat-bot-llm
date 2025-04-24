import React, { useState, useRef, useEffect } from "react";
import SearchBar from "./SearchBar";
import styles from "./ChatInterface.module.css";
import axios from "axios";

const API_BASE_URL = "http://localhost:8000";

const ChatInterface = () => {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [ws, setWs] = useState(null);
  const [activeDocument, setActiveDocument] = useState(null);
  const messagesEndRef = useRef(null);

  // Initialize WebSocket connection
  useEffect(() => {
    const websocket = new WebSocket(`ws://localhost:8000/api/chat/ws`);

    websocket.onopen = () => {
      console.log("WebSocket connection established");
      setWs(websocket);
    };

    websocket.onmessage = handleWebSocketMessage;

    websocket.onclose = () => {
      console.log("WebSocket connection closed");
      setWs(null);
    };

    return () => {
      if (websocket.readyState === WebSocket.OPEN) {
        websocket.close();
      }
    };
  }, []);

  // Scroll to bottom when messages change
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  // Handle streaming messages from WebSocket
  const handleWebSocketMessage = (event) => {
    const data = JSON.parse(event.data);

    if (data.type === "token") {
      // Append token to current response
      setMessages((prevMessages) => {
        const updatedMessages = [...prevMessages];
        const lastIndex = updatedMessages.length - 1;

        // If last message is from assistant, update it
        if (lastIndex >= 0 && updatedMessages[lastIndex].role === "assistant") {
          updatedMessages[lastIndex] = {
            ...updatedMessages[lastIndex],
            content: updatedMessages[lastIndex].content + data.content,
          };
        } else {
          // Otherwise add a new assistant message
          updatedMessages.push({
            role: "assistant",
            content: data.content,
          });
        }

        return updatedMessages;
      });
    } else if (data.type === "complete") {
      // Reset loading state
      setIsLoading(false);

      // Update active document if provided
      if (data.active_document) {
        setActiveDocument(data.active_document);
      }
    } else if (data.type === "error") {
      console.error("WebSocket error:", data.content);
      setIsLoading(false);
      // Add error message
      setMessages((prevMessages) => [
        ...prevMessages,
        {
          role: "assistant",
          content: "Có lỗi xảy ra. Vui lòng thử lại.",
        },
      ]);
    }
  };

  // Handle text submissions (including URLs)
  const handleMessageSubmit = async (text) => {
    if (!text.trim()) return;

    // Add user message to chat
    const userMessage = { role: "user", content: text };
    setMessages((prevMessages) => [...prevMessages, userMessage]);

    // Set loading state
    setIsLoading(true);

    if (ws && ws.readyState === WebSocket.OPEN) {
      // Send message via WebSocket for streaming response
      ws.send(JSON.stringify({ message: text }));
    } else {
      // Fallback to REST API if WebSocket is not available
      try {
        const response = await axios.post(`${API_BASE_URL}/api/chat`, {
          message: text,
        });

        // Add assistant response to chat
        const assistantMessage = {
          role: "assistant",
          content: response.data.response,
        };
        setMessages((prevMessages) => [...prevMessages, assistantMessage]);

        // Update active document if provided
        if (response.data.active_document) {
          setActiveDocument(response.data.active_document);
        }
      } catch (error) {
        console.error("Error communicating with the chatbot:", error);
        const errorMessage = {
          role: "assistant",
          content:
            "Rất tiếc, đã xảy ra lỗi khi xử lý yêu cầu của bạn. Vui lòng thử lại.",
        };
        setMessages((prevMessages) => [...prevMessages, errorMessage]);
      } finally {
        setIsLoading(false);
      }
    }
  };

  // Handle file uploads
  const handleFileUpload = async (file) => {
    if (!file) return;

    // Add uploading message
    const uploadMessage = {
      role: "user",
      content: `Đang tải lên PDF: ${file.name}`,
    };
    setMessages((prevMessages) => [...prevMessages, uploadMessage]);

    // Set loading state
    setIsLoading(true);

    try {
      // Create FormData to send file
      const formData = new FormData();
      formData.append("file", file);

      // Send file to backend
      const response = await axios.post(
        `${API_BASE_URL}/api/upload-pdf`,
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        },
      );

      // Add assistant response about successful upload
      const assistantMessage = {
        role: "assistant",
        content: response.data.message,
      };
      setMessages((prevMessages) => [...prevMessages, assistantMessage]);

      // Update active document with the newly uploaded PDF
      setActiveDocument({
        type: "pdf",
        name: file.name,
        source: file.name,
        timestamp: new Date().toISOString(),
      });
    } catch (error) {
      console.error("Error uploading PDF:", error);
      const errorMessage = {
        role: "assistant",
        content:
          error.response?.data?.detail ||
          "Có lỗi xảy ra khi tải lên tập tin. Vui lòng thử lại.",
      };
      setMessages((prevMessages) => [...prevMessages, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  // Render active document information if available
  const renderActiveDocument = () => {
    if (!activeDocument) return null;

    return (
      <div className={styles.activeDocumentBanner}>
        <span className={styles.activeDocumentLabel}>
          Tài liệu đang sử dụng:
        </span>
        <span className={styles.activeDocumentName}>
          {activeDocument.type === "webpage" ? "🌐 " : "📄 "}
          {activeDocument.name}
        </span>
      </div>
    );
  };

  return (
    <div className={styles.chatContainer}>
      {renderActiveDocument()}
      <div className={styles.messageContainer}>
        {messages.length === 0 && (
          <div className={styles.emptyState}>
            <h2>Tôi có thể giúp gì cho bạn hôm nay?</h2>
            <p>
              Hãy đặt câu hỏi, dán liên kết URL, hoặc tải lên file PDF để phân
              tích
            </p>
          </div>
        )}

        {messages.map((message, index) => (
          <div
            key={index}
            className={`${styles.messageWrapper} ${
              message.role === "user"
                ? styles.userMessage
                : styles.assistantMessage
            }`}
          >
            <div className={styles.messageContent}>
              <div className={styles.messageHeader}>
                {message.role === "user" ? "Bạn" : "Assistant"}
              </div>
              <div className={styles.messageBody}>{message.content}</div>
            </div>
          </div>
        ))}

        {isLoading && (
          <div
            className={`${styles.messageWrapper} ${styles.assistantMessage}`}
          >
            <div className={styles.messageContent}>
              <div className={styles.messageHeader}>Assistant</div>
              <div className={styles.messageBody}>
                <div className={styles.loadingIndicator}>
                  <span className={styles.dot}></span>
                  <span className={styles.dot}></span>
                  <span className={styles.dot}></span>
                </div>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      <SearchBar
        onSubmit={handleMessageSubmit}
        onFileUpload={handleFileUpload}
        isLoading={isLoading}
      />
    </div>
  );
};

export default ChatInterface;
