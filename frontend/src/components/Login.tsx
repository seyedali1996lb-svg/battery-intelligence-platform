import { useState } from "react";
import { login, setToken, ApiError } from "../api";
import type { LoginResponse } from "../types";

interface Props {
  onLoggedIn: (user: LoginResponse) => void;
}

export default function Login({ onLoggedIn }: Props) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      const result = await login(username, password);
      setToken(result.access_token);
      onLoggedIn(result);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Unable to reach the server. Please check your connection and try again.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <form className="login-form" onSubmit={handleSubmit}>
      <h1 style={{ letterSpacing: "-0.03em" }}>Battery Intelligence</h1>
      <p className="subtitle">
        Welcome back. Sign in with your team account to access diagnostics,
        fleet monitoring, and compliance tools.
      </p>
      <input
        placeholder="Username"
        value={username}
        onChange={(e) => setUsername(e.target.value)}
        autoFocus
      />
      <input
        type="password"
        placeholder="Password"
        value={password}
        onChange={(e) => setPassword(e.target.value)}
      />
      <button type="submit" disabled={loading || !username || !password}>
        {loading ? "Signing in…" : "Sign in"}
      </button>
      {error && <div className="error-text">{error}</div>}
      <p className="subtitle" style={{ fontSize: 12, lineHeight: 1.5 }}>
        Try the demo: <code>engineer</code> / <code>battery</code> (Demo Org)
      </p>
    </form>
  );
}
