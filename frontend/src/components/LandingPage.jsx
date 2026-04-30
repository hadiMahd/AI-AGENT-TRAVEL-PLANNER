import { useEffect } from "react";
import { useNavigate, Link } from "react-router-dom";
import { MapPin, Plane, CloudSun, Sparkles, ArrowRight, Compass, Globe, Star } from "lucide-react";
import { getToken } from "../api";

const FEATURES = [
  {
    icon: <Sparkles size={28} />,
    title: "Smart Itineraries",
    desc: "AI-crafted day-by-day plans tailored to your style, budget, and travel group.",
  },
  {
    icon: <Plane size={28} />,
    title: "Live Flights & Rates",
    desc: "Real-time flight options and currency exchange so your budget is always accurate.",
  },
  {
    icon: <CloudSun size={28} />,
    title: "Weather-Aware Planning",
    desc: "Destination weather pulled at planning time so you pack right and time it well.",
  },
];

// Floating background elements — position, size, color, animation delay/duration
const BG_ITEMS = [
  { icon: Plane,    size: 52, color: "#38bdf8", style: { top: "12%", left: "6%"  }, cls: "lbg-drift-r", delay: 0,  dur: 18 },
  { icon: CloudSun, size: 80, color: "#fbbf24", style: { top: "6%",  left: "70%" }, cls: "lbg-float",   delay: 2,  dur: 22 },
  { icon: CloudSun, size: 54, color: "#bae6fd", style: { top: "52%", left: "87%" }, cls: "lbg-float",   delay: 6,  dur: 28 },
  { icon: MapPin,   size: 42, color: "#f43f5e", style: { top: "28%", left: "83%" }, cls: "lbg-float",   delay: 1,  dur: 16 },
  { icon: Compass,  size: 48, color: "#f59e0b", style: { top: "62%", left: "4%"  }, cls: "lbg-spin-s",  delay: 0,  dur: 32 },
  { icon: Globe,    size: 68, color: "#34d399", style: { top: "70%", left: "66%" }, cls: "lbg-float",   delay: 4,  dur: 20 },
  { icon: Star,     size: 28, color: "#fbbf24", style: { top: "18%", left: "44%" }, cls: "lbg-twinkle", delay: 3,  dur: 14 },
  { icon: Star,     size: 20, color: "#a78bfa", style: { top: "46%", left: "58%" }, cls: "lbg-twinkle", delay: 7,  dur: 10 },
  { icon: Plane,    size: 36, color: "#60a5fa", style: { top: "80%", left: "28%" }, cls: "lbg-drift-r", delay: 9,  dur: 24 },
  { icon: MapPin,   size: 32, color: "#fb923c", style: { top: "36%", left: "14%" }, cls: "lbg-float",   delay: 5,  dur: 19 },
];

export default function LandingPage() {
  const navigate = useNavigate();

  useEffect(() => {
    if (getToken()) navigate("/", { replace: true });
  }, [navigate]);

  return (
    <div className="landing-page">
      <nav className="landing-nav">
        <div className="landing-brand">
          <MapPin size={22} />
          <span>Smart Travel Planner</span>
        </div>
        <Link to="/login" className="landing-nav-link">Sign In</Link>
      </nav>

      <section className="landing-hero">
        {/* Animated background layer */}
        <div className="landing-bg" aria-hidden="true">
          {BG_ITEMS.map(({ icon: Icon, size, color, style, cls, delay, dur }, i) => (
            <div
              key={i}
              className={`lbg-item ${cls}`}
              style={{
                ...style,
                color,
                animationDelay: `${delay}s`,
                animationDuration: `${dur}s`,
              }}
            >
              <Icon size={size} />
            </div>
          ))}
        </div>

        <h1 className="landing-headline">
          Plan your next adventure<br />
          <span className="landing-accent">with AI</span>
        </h1>
        <p className="landing-sub">
          Flights, weather, currency, and personalised itineraries — all in one conversation.
        </p>
        <div className="landing-cta-group">
          <Link to="/signup" className="landing-cta-primary">
            Get Started <ArrowRight size={16} />
          </Link>
          <Link to="/login" className="landing-cta-secondary">
            Sign In
          </Link>
        </div>
      </section>

      <section className="landing-features">
        {FEATURES.map((f) => (
          <div key={f.title} className="landing-feature-card">
            <div className="landing-feature-icon">{f.icon}</div>
            <h3>{f.title}</h3>
            <p>{f.desc}</p>
          </div>
        ))}
      </section>
    </div>
  );
}
