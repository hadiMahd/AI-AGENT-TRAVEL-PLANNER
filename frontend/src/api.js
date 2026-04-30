const API_BASE = "http://localhost:8000";

const TOKEN_KEY = "travel_planner_token";

export function getToken() {
  return localStorage.getItem(TOKEN_KEY);
}

export function setToken(token) {
  localStorage.setItem(TOKEN_KEY, token);
}

export function clearToken() {
  localStorage.removeItem(TOKEN_KEY);
}

function authHeaders() {
  const token = getToken();
  const headers = { "Content-Type": "application/json" };
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }
  return headers;
}

async function apiFetch(path, options = {}) {
  let res;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      ...options,
      headers: {
        ...authHeaders(),
        ...options.headers,
      },
    });
  } catch {
    throw new Error(
      "Cannot reach the server. Make sure the backend is running at " +
        API_BASE
    );
  }

  if (res.status === 401) {
    clearToken();
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || "Session expired — please log in again.");
  }

  if (res.status === 409) {
    throw new Error("This email is already registered. Try logging in instead.");
  }

  if (res.status === 422) {
    const body = await res.json().catch(() => ({}));
    const details = body.detail;
    if (Array.isArray(details)) {
      const msgs = details.map((d) => d.msg || String(d)).join("; ");
      throw new Error("Invalid input: " + msgs);
    }
    throw new Error(details || "Invalid input — please check your data.");
  }

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed (${res.status})`);
  }

  return res.json();
}

export async function checkBackendHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`, { method: "GET" });
    const data = await res.json();
    return data;
  } catch {
    return null;
  }
}

export async function signup(email, password) {
  const data = await apiFetch("/auth/signup", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  });
  setToken(data.access_token);
  return data;
}

export async function login(email, password) {
  const data = await apiFetch("/auth/login", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  });
  setToken(data.access_token);
  return data;
}

export async function getMe() {
  return apiFetch("/user/me");
}

export async function getUserStats() {
  return apiFetch("/user/stats");
}

export async function sendPlanEmail(email, plan, destination) {
  return apiFetch("/agent/send-email", {
    method: "POST",
    body: JSON.stringify({ email, plan, destination }),
  });
}

export async function sendChat(query, originCountry, history, onEvent) {
  const token = getToken();
  if (!token) throw new Error("Not authenticated");

  const body = { query, history: history || [] };
  if (originCountry) {
    body.origin_country = originCountry;
  }

  const res = await fetch(`${API_BASE}/agent/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify(body),
  });

  if (res.status === 401) {
    clearToken();
    throw new Error("Session expired — please log in again.");
  }

  if (!res.ok) {
    const errBody = await res.json().catch(() => ({}));
    throw new Error(errBody.detail || `Chat failed (${res.status})`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    let currentEvent = null;

    for (const line of lines) {
      if (line.startsWith("event: ")) {
        currentEvent = line.slice(7).trim();
      } else if (line.startsWith("data: ") && currentEvent) {
        try {
          const data = JSON.parse(line.slice(6));
          onEvent({ type: currentEvent, data });
        } catch {
          // skip malformed JSON
        }
        currentEvent = null;
      }
    }
  }
}
