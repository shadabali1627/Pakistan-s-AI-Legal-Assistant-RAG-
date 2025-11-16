import { Link, useNavigate } from "react-router-dom";
import { useState } from "react";
import { useAuth } from "../auth";

export default function Login() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const { signin } = useAuth();
  const nav = useNavigate();

  const onSubmit = async (e) => {
    e.preventDefault();
    await signin(email, password);
    nav("/chat");
  };

  return (
    <div className="auth-wrap">
      <div className="brand">
        <div className="logo">⚖️</div>
        <h1>AI Legal Assistant</h1>
        <p>Your intelligent legal companion</p>
      </div>

      <form className="card" onSubmit={onSubmit}>
        <h2>Sign In</h2>
        <p className="muted">Enter your credentials to access your legal assistant</p>

        <label>Email</label>
        <input
          placeholder="you@example.com"
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
        />

        <label>Password</label>
        <input
          placeholder="••••••••"
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
        />

        <div className="muted small" style={{ marginTop: -8 }}>
          <a href="#">Forgot password?</a>
        </div>

        <button className="primary" type="submit">Sign In</button>

        <div className="muted center small">
          Don’t have an account? <Link to="/signup">Sign up</Link>
        </div>
        <p className="tiny center muted">
          Protected by attorney-client privilege guidelines
        </p>
      </form>
    </div>
  );
}
