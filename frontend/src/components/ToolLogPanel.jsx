import { useState } from "react";
import { ChevronDown, ChevronRight, Check, X, Loader2 } from "lucide-react";

export default function ToolLogPanel({ tools }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="tool-log-panel">
      <button
        className="tool-log-toggle"
        onClick={() => setExpanded(!expanded)}
      >
        {expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        <span>{tools.length} tool{tools.length !== 1 ? "s" : ""} used</span>
        <span className="tool-log-summary">
          {tools.map((t) => (
            <span key={t.tool_name} className={`tool-badge ${t.status}`}>
              {t.status === "running" && <Loader2 size={10} className="spin" />}
              {t.status === "success" && <Check size={10} />}
              {t.status === "error" && <X size={10} />}
              {t.tool_name.replace("_", " ")}
            </span>
          ))}
        </span>
      </button>

      {expanded && (
        <div className="tool-log-details">
          {tools.map((t, i) => (
            <div key={i} className={`tool-log-item ${t.status}`}>
              <div className="tool-log-name">
                {t.tool_name}
                {t.latency_ms != null && (
                  <span className="tool-latency">{t.latency_ms}ms</span>
                )}
                <span className={`tool-status-badge ${t.status}`}>
                  {t.status}
                </span>
              </div>
              {t.input_payload && (
                <div className="tool-log-section">
                  <span className="tool-log-label">Input:</span>
                  <pre>{JSON.stringify(t.input_payload, null, 2)}</pre>
                </div>
              )}
              {t.output_payload && (
                <div className="tool-log-section">
                  <span className="tool-log-label">Output:</span>
                  <pre>
                    {typeof t.output_payload === "string"
                      ? t.output_payload.slice(0, 300)
                      : JSON.stringify(t.output_payload, null, 2).slice(0, 300)}
                  </pre>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
