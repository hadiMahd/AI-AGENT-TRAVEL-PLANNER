import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { getMe, clearToken, getUserStats } from "../api";
import { LogOut, MapPin, BarChart3 } from "lucide-react";
import ChatPanel from "./ChatPanel";

export default function Dashboard() {
  const [user, setUser] = useState(null);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    async function load() {
      try {
        const [me, s] = await Promise.all([getMe(), getUserStats()]);
        setUser(me);
        setStats(s);
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
        <div className="dashboard-brand">
          <MapPin size={24} />
          <span>Smart Travel Planner</span>
        </div>
        <div className="dashboard-user-bar">
          <span className="dashboard-email">{user?.email}</span>
          <button onClick={handleLogout} className="logout-button">
            <LogOut size={16} />
            Logout
          </button>
        </div>
      </header>

      <ChatPanel />
    </div>
  );
}
