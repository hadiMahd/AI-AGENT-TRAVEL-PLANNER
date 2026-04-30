import { useState, useRef, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Send, Loader2 } from "lucide-react";
import { sendChat, sendPlanEmail, getToken } from "../api";
import ChatMessage from "./ChatMessage";

export default function ChatPanel() {
  const navigate = useNavigate();
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [pendingQuestion, setPendingQuestion] = useState(null);
  const [originCountry, setOriginCountry] = useState(null);
  const [chatHistory, setChatHistory] = useState([]);

  // Email flow state
  const [pendingEmail, setPendingEmail] = useState(null); // { plan, destination }
  const [lastPlan, setLastPlan] = useState(null);         // persists after user presses No
  const [emailInput, setEmailInput] = useState("");
  const [emailLoading, setEmailLoading] = useState(false);

  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function handleSend(e) {
    e.preventDefault();
    const text = input.trim();
    if (!text || loading) return;

    setInput("");

    // Intercept email-send requests locally — no need to hit the agent
    const lower = text.toLowerCase();
    const isEmailRequest =
      lastPlan &&
      (lower.includes("email") || lower.includes("mail") || lower.includes("send")) &&
      (lower.includes("plan") || lower.includes("it") || lower.includes("send"));

    if (isEmailRequest) {
      setMessages((prev) => [
        ...prev,
        { role: "user", content: text },
        {
          role: "agent",
          content: "Sure! Would you like me to send your travel plan to your email?",
          toolLogs: [],
          streaming: false,
          isEmailAsk: true,
        },
      ]);
      setPendingEmail(lastPlan);
      return;
    }

    setMessages((prev) => [...prev, { role: "user", content: text }]);

    // Capture origin before async state update so sendChat gets the right value
    const originForThisRequest = pendingQuestion ? text : originCountry;
    if (pendingQuestion) {
      setOriginCountry(text);
      setPendingQuestion(null);
    }

    setLoading(true);

    try {
      const toolLogs = [];
      let agentResponse = "";
      let needsInput = false;
      let userQuestion = null;
      let emailAskData = null;

      await sendChat(text, originForThisRequest, chatHistory, (event) => {
        if (event.type === "thinking") {
          setMessages((prev) => [
            ...prev.filter((m) => m.role !== "thinking"),
            { role: "thinking", content: event.data.message || "Thinking..." },
          ]);
        } else if (event.type === "tool_start") {
          toolLogs.push({
            tool_name: event.data.tool,
            input_payload: event.data.input,
            status: "running",
            latency_ms: null,
          });
          setMessages((prev) => [
            ...prev.filter((m) => m.role !== "thinking" && !(m.role === "agent" && m.streaming)),
            {
              role: "agent",
              content: "",
              toolLogs: [...toolLogs],
              streaming: true,
            },
          ]);
        } else if (event.type === "tool_result") {
          const idx = toolLogs.findIndex((t) => t.tool_name === event.data.tool);
          if (idx >= 0) {
            toolLogs[idx] = {
              ...toolLogs[idx],
              status: event.data.error ? "error" : "success",
              output_payload: event.data.output || event.data.error,
              latency_ms: event.data.latency_ms,
            };
          }
          setMessages((prev) => [
            ...prev.filter((m) => m.role !== "thinking" && !(m.role === "agent" && m.streaming)),
            {
              role: "agent",
              content: "",
              toolLogs: [...toolLogs],
              streaming: true,
            },
          ]);
        } else if (event.type === "token") {
          setMessages((prev) => {
            const last = prev[prev.length - 1];
            if (last && last.role === "agent" && last.streaming) {
              const updated = [...prev];
              updated[updated.length - 1] = {
                ...last,
                content: last.content + event.data.text,
              };
              return updated;
            }
            return [
              ...prev.filter((m) => m.role !== "thinking"),
              { role: "agent", content: event.data.text, toolLogs: [], streaming: true },
            ];
          });
        } else if (event.type === "needs_input") {
          needsInput = true;
          userQuestion = event.data.question;
        } else if (event.type === "final") {
          agentResponse = event.data.response;
          const finalLogs = event.data.tool_logs || toolLogs;
          setMessages((prev) => [
            ...prev.filter((m) => m.role !== "thinking" && !(m.role === "agent" && m.streaming)),
            {
              role: "agent",
              content: agentResponse,
              toolLogs: finalLogs,
              streaming: false,
            },
          ]);
        } else if (event.type === "ask_email") {
          emailAskData = event.data;
          setLastPlan({ plan: event.data.plan, destination: event.data.destination || "" });
        }
      });

      if (needsInput && userQuestion) {
        setPendingQuestion(userQuestion);
        setMessages((prev) => [
          ...prev.filter((m) => m.role !== "thinking"),
          {
            role: "agent",
            content: userQuestion,
            toolLogs: [],
            streaming: false,
            isQuestion: true,
          },
        ]);
        setChatHistory((prev) => [
          ...prev,
          { role: "user", content: text },
          { role: "assistant", content: userQuestion },
        ]);
      } else if (agentResponse) {
        setChatHistory((prev) => [
          ...prev,
          { role: "user", content: text },
          { role: "assistant", content: agentResponse },
        ]);

        // Show email prompt after travel plan
        if (emailAskData) {
          setPendingEmail({ plan: emailAskData.plan, destination: emailAskData.destination || "" });
          setMessages((prev) => [
            ...prev,
            {
              role: "agent",
              content: emailAskData.question,
              toolLogs: [],
              streaming: false,
              isEmailAsk: true,
            },
          ]);
        }
      }
    } catch (err) {
      if (err.message.includes("Session expired") || err.message.includes("Not authenticated")) {
        navigate("/login", { replace: true });
        return;
      }
      setMessages((prev) => [
        ...prev.filter((m) => m.role !== "thinking"),
        {
          role: "agent",
          content: `Error: ${err.message}`,
          toolLogs: [],
          streaming: false,
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  function handleEmailDecline() {
    setPendingEmail(null);
    setMessages((prev) =>
      prev.map((m) => (m.isEmailAsk ? { ...m, isEmailAsk: false, emailDeclined: true } : m))
    );
  }

  async function handleEmailSubmit(e) {
    e.preventDefault();
    const email = emailInput.trim();
    if (!email || !pendingEmail) return;

    setEmailLoading(true);
    try {
      await sendPlanEmail(email, pendingEmail.plan, pendingEmail.destination);
      setPendingEmail(null);
      setEmailInput("");
      setMessages((prev) => [
        ...prev.filter((m) => !m.isEmailAsk),
        {
          role: "agent",
          content: `Done! Your travel plan has been sent to **${email}**. Check your inbox!`,
          toolLogs: [],
          streaming: false,
        },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev.filter((m) => !m.isEmailAsk),
        {
          role: "agent",
          content: `Couldn't send the email: ${err.message}`,
          toolLogs: [],
          streaming: false,
        },
      ]);
      setPendingEmail(null);
      setEmailInput("");
    } finally {
      setEmailLoading(false);
    }
  }

  return (
    <div className="chat-panel">
      <div className="chat-messages">
        {messages.length === 0 && (
          <div className="chat-empty">
            <p>Ask me anything about travel destinations!</p>
            <p className="chat-hint">
              Try: "Plan a trip to Bali" or "What's the weather in Paris?"
            </p>
          </div>
        )}
        {messages.map((msg, i) => (
          <ChatMessage
            key={i}
            message={msg}
            onEmailYes={() => {
              setMessages((prev) =>
                prev.map((m) => (m.isEmailAsk ? { ...m, isEmailAsk: false, emailAccepted: true } : m))
              );
            }}
            onEmailNo={handleEmailDecline}
          />
        ))}
        {loading && !messages.some((m) => m.role === "thinking") && (
          <div className="chat-thinking">
            <Loader2 size={16} className="spin" />
            <span>Thinking...</span>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Email input bar — shown after user taps Yes */}
      {pendingEmail && messages.some((m) => m.emailAccepted) ? (
        <form onSubmit={handleEmailSubmit} className="chat-input-bar">
          <input
            type="email"
            value={emailInput}
            onChange={(e) => setEmailInput(e.target.value)}
            placeholder="Enter your email address..."
            disabled={emailLoading}
            className="chat-input"
            autoFocus
          />
          <button
            type="submit"
            disabled={emailLoading || !emailInput.trim()}
            className="chat-send"
          >
            {emailLoading ? <Loader2 size={18} className="spin" /> : <Send size={18} />}
          </button>
        </form>
      ) : (
        <form onSubmit={handleSend} className="chat-input-bar">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={
              pendingQuestion
                ? pendingQuestion
                : "Ask about a travel destination..."
            }
            disabled={loading}
            className="chat-input"
          />
          <button type="submit" disabled={loading || !input.trim()} className="chat-send">
            <Send size={18} />
          </button>
        </form>
      )}
    </div>
  );
}
