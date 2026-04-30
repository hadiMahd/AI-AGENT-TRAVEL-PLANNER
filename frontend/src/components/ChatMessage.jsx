import { useState } from "react";
import { User, Bot, ChevronDown, ChevronRight, Mail } from "lucide-react";
import ReactMarkdown from "react-markdown";
import ToolLogPanel from "./ToolLogPanel";

export default function ChatMessage({ message, onEmailYes, onEmailNo }) {
  const isUser = message.role === "user";
  const isThinking = message.role === "thinking";

  if (isThinking) {
    return (
      <div className="chat-msg thinking">
        <Bot size={18} className="msg-icon" />
        <span className="thinking-text">{message.content}</span>
      </div>
    );
  }

  return (
    <div className={`chat-msg ${isUser ? "user" : "agent"}`}>
      <div className="msg-header">
        {isUser ? <User size={14} /> : <Bot size={14} />}
        <span>{isUser ? "You" : "Travel Planner"}</span>
      </div>
      <div className="msg-body">
        {message.content && (
          <div className="msg-text">
            <ReactMarkdown>{message.content}</ReactMarkdown>
          </div>
        )}
        {message.isQuestion && (
          <p className="msg-question">Please answer in the input box below.</p>
        )}
        {message.isEmailAsk && (
          <div className="email-ask-buttons">
            <button className="email-btn email-btn-yes" onClick={onEmailYes}>
              <Mail size={14} /> Yes, send it!
            </button>
            <button className="email-btn email-btn-no" onClick={onEmailNo}>
              No thanks
            </button>
          </div>
        )}
        {message.emailAccepted && (
          <p className="msg-question">Enter your email address below.</p>
        )}
        {message.emailDeclined && (
          <p className="msg-question" style={{ color: "#888" }}>No problem!</p>
        )}
        {message.toolLogs && message.toolLogs.length > 0 && (
          <ToolLogPanel tools={message.toolLogs} />
        )}
      </div>
    </div>
  );
}
