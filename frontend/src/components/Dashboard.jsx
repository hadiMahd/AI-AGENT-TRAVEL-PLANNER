import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { getMe, clearToken, getUserStats } from "../api";
import { LogOut, MapPin, BarChart3 } from "lucide-react";

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

      <main className="dashboard-main">
        <section className="dashboard-welcome">
          <h1>Welcome, {user?.email}</h1>
          <p>Your AI-powered travel planning assistant</p>
        </section>

        <section className="dashboard-stats">
          <div className="stat-card">
            <BarChart3 size={20} />
            <div>
              <span className="stat-value">{stats?.agent_runs ?? 0}</span>
              <span className="stat-label">Agent Runs</span>
            </div>
          </div>
        </section>

        <section className="dashboard-placeholder">
          <MapPin size={48} />
          <h2>Travel Planner Chat</h2>
          <p>
            The AI agent chat interface will appear here once the LangGraph
            agent is built. It will let you ask travel questions and see the
            agent's reasoning — tools fired, inputs/outputs, and your
            personalized plan.
          </p>
        </section>
      </main>
    </div>
  );
}
