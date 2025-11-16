import { Link, useNavigate } from "react-router-dom";
import { useState } from "react";
import { useAuth } from "../auth";

export default function Signup() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const { signup } = useAuth();
  const nav = useNavigate();

  const onSubmit = async (e) => {
    e.preventDefault();
    await signup(email, password);
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
        <h2>Create Account</h2>
        <p className="muted">Sign up with any email and password (demo)</p>

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

        <button className="primary" type="submit">Sign Up</button>

        <div className="muted center small">
          Already have an account? <Link to="/login">Sign in</Link>
        </div>
      </form>
    </div>
  );
}
