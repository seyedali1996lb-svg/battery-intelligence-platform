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
      setError(err instanceof ApiError ? err.message : "Unable to reach the API.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <form className="login-form" onSubmit={handleSubmit}>
      <h1>Battery Intelligence</h1>
      <p className="subtitle">Sign in with the same account used in the Streamlit app.</p>
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
      <p className="subtitle">
        Demo credentials (Demo Org): <code>engineer</code> / <code>battery</code>
      </p>
    </form>
  );
}
