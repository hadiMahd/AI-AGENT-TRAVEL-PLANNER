import { SquarePen, MessageSquare, Trash2 } from "lucide-react";

export default function Sidebar({ open, sessions, activeId, onSelect, onNew, onDelete }) {
  return (
    <aside className={`sidebar ${open ? "sidebar--open" : ""}`}>
      <div className="sidebar-inner">
        <button className="sidebar-new-btn" onClick={onNew}>
          <SquarePen size={16} />
          New Chat
        </button>

        <div className="sidebar-list">
          {sessions.length === 0 && (
            <p className="sidebar-empty">No chats yet</p>
          )}
          {sessions.map((s) => (
            <div
              key={s.id}
              className={`sidebar-item ${s.id === activeId ? "sidebar-item--active" : ""}`}
            >
              <button
                className="sidebar-item-body"
                onClick={() => onSelect(s.id)}
              >
                <MessageSquare size={14} className="sidebar-item-icon" />
                <span className="sidebar-item-title">{s.title}</span>
              </button>
              <button
                className="sidebar-item-delete"
                onClick={(e) => { e.stopPropagation(); onDelete(s.id); }}
                aria-label="Delete chat"
                title="Delete chat"
              >
                <Trash2 size={13} />
              </button>
            </div>
          ))}
        </div>
      </div>
    </aside>
  );
}
