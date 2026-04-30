import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { getMe, clearToken, getUserStats } from "../api";
import { LogOut, MapPin, BarChart3, Menu } from "lucide-react";
import ChatPanel from "./ChatPanel";
import Sidebar from "./Sidebar";

const HISTORY_KEY = "travel_planner_chats";
const MAX_SESSIONS = 30;

function loadSessions() {
  try { return JSON.parse(localStorage.getItem(HISTORY_KEY)) || []; }
  catch { return []; }
}

function persistSessions(sessions) {
  localStorage.setItem(
    HISTORY_KEY,
    JSON.stringify(sessions.slice(0, MAX_SESSIONS))
  );
}

export default function Dashboard() {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [sessions, setSessions] = useState(loadSessions);
  const [activeId, setActiveId] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    async function load() {
      try {
        const me = await getMe();
        setUser(me);
      } catch {
        clearToken();
        navigate("/login", { replace: true });
      } finally {
        setLoading(false);
      }
    }
    load();
  }, [navigate]);

  function handleLogout() {
    clearToken();
    navigate("/login", { replace: true });
  }

  function startNewChat() {
    setActiveId(null);
  }

  function selectSession(id) {
    setActiveId(id);
    setSidebarOpen(false);
  }

  function deleteSession(id) {
    setSessions((prev) => {
      const updated = prev.filter((s) => s.id !== id);
      persistSessions(updated);
      return updated;
    });
    if (activeId === id) setActiveId(null);
  }

  function saveSession(session) {
    setSessions((prev) => {
      const idx = prev.findIndex((s) => s.id === session.id);
      let updated;
      if (idx >= 0) {
        updated = [...prev];
        updated[idx] = session;
      } else {
        updated = [session, ...prev];
      }
      persistSessions(updated);
      return updated;
    });
    setActiveId(session.id);
  }

  const activeSession = sessions.find((s) => s.id === activeId) || null;

  if (loading) {
    return (
      <div className="dashboard-page">
        <div className="dashboard-loading">Loading...</div>
      </div>
    );
  }

  return (
    <div className="dashboard-page">
      <header className="dashboard-header">
        <div className="dashboard-header-left">
          <button
            className="hamburger"
            onClick={() => setSidebarOpen((o) => !o)}
            aria-label="Toggle sidebar"
          >
            <Menu size={20} />
          </button>
          <div className="dashboard-brand">
            <MapPin size={24} />
            <span>Smart Travel Planner</span>
          </div>
        </div>
        <div className="dashboard-user-bar">
          <span className="dashboard-email">{user?.email}</span>
          <button onClick={handleLogout} className="logout-button">
            <LogOut size={16} />
            Logout
          </button>
        </div>
      </header>

      <div className="dashboard-body">
        <Sidebar
          open={sidebarOpen}
          sessions={sessions}
          activeId={activeId}
          onSelect={selectSession}
          onNew={startNewChat}
          onDelete={deleteSession}
        />
        <ChatPanel
          key={activeId ?? "new"}
          initialState={activeSession}
          onSave={saveSession}
        />
      </div>
    </div>
  );
}
