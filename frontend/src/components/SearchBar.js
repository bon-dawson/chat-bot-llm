import React, { useState, useRef } from "react";
import styles from "./SearchBar.module.css";
import { IoAdd, IoArrowUp } from "react-icons/io5";

function SearchBar({ onSubmit, onFileUpload, isLoading }) {
  const [inputValue, setInputValue] = useState("");
  const fileInputRef = useRef(null);

  // --- Event Handlers ---
  const handleInputChange = (event) => {
    setInputValue(event.target.value);
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    if (inputValue.trim() && !isLoading) {
      onSubmit(inputValue);
      setInputValue(""); // Clear input after submission
    }
  };

  // Handle keyboard shortcuts
  const handleKeyDown = (event) => {
    // Enter to submit, but only if not pressing shift (shift+enter for new line)
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      if (inputValue.trim() && !isLoading) {
        onSubmit(inputValue);
        setInputValue("");
      }
    }
  };

  // Trigger the hidden file input when the '+' button is clicked
  const handleAddClick = () => {
    if (fileInputRef.current && !isLoading) {
      fileInputRef.current.click();
    }
  };

  // Handle the file selection from the hidden input
  const handleFileChange = (event) => {
    const file = event.target.files && event.target.files[0];
    if (file && !isLoading) {
      onFileUpload(file);
      event.target.value = null; // Reset file input
    }
  };

  // --- Render ---
  return (
    <form onSubmit={handleSubmit} className={styles.searchBarContainer}>
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        accept=".pdf,application/pdf"
        style={{ display: "none" }}
        aria-hidden="true"
      />
      <div className={styles.leftSection}>
        <div className={styles.actionButtonsStart}>
          <button
            type="button"
            onClick={handleAddClick}
            className={`${styles.button} ${styles.roundButton} ${styles.iconOnlyButton}`}
            aria-label="Thêm file PDF"
            disabled={isLoading}
          >
            <IoAdd />
          </button>
        </div>
        <input
          type="text"
          value={inputValue}
          onChange={handleInputChange}
          onKeyDown={handleKeyDown}
          placeholder="Hỏi điều gì đó, dán URL, hoặc thêm file"
          className={styles.searchInput}
          aria-label="Nhập câu hỏi"
          disabled={isLoading}
        />
      </div>
      <div className={styles.rightSection}>
        <button
          type="submit"
          className={`${styles.button} ${styles.roundButton} ${styles.iconOnlyButton} ${styles.sendButton}`}
          aria-label="Gửi tin nhắn"
          disabled={!inputValue.trim() || isLoading}
        >
          <IoArrowUp />
        </button>
      </div>
    </form>
  );
}

export default SearchBar;
