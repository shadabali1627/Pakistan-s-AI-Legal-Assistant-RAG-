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
        <div className="logo" style={{ marginBottom: '16px' }}><img src="/logo.svg" alt="Logo" style={{ width: '64px', height: '64px' }} /></div>
        <h1>AI Legal Assistant</h1>
        <p>Your intelligent legal companion</p>
      </div>

      <form className="card" onSubmit={onSubmit}>
        <h2>Create Account</h2>
        <p className="muted" style={{ marginBottom: '24px' }}>Sign up with any email and password (demo)</p>

        <label>Email</label>
        <input
          placeholder="you@example.com"
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
        />

        <label style={{ marginTop: '16px' }}>Password</label>
        <input
          placeholder="••••••••"
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
        />

        <button className="primary" type="submit" style={{ marginTop: '24px' }}>Sign Up</button>

        <div className="muted center small" style={{ marginTop: '16px' }}>
          Already have an account? <Link to="/login">Sign in</Link>
        </div>
      </form>
    </div>
  );
}
