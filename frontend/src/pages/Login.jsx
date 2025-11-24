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
        <div className="logo" style={{ marginBottom: '16px' }}><img src="/logo.svg" alt="Logo" style={{ width: '64px', height: '64px' }} /></div>
        <h1>AI Legal Assistant</h1>
        <p>Your intelligent legal companion</p>
      </div>

      <form className="card" onSubmit={onSubmit}>
        <h2>Sign In</h2>
        <p className="muted" style={{ marginBottom: '24px' }}>Enter your credentials to access your legal assistant</p>

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

        <div className="muted small" style={{ marginTop: '8px', textAlign: 'right' }}>
          <a href="#" className="muted">Forgot password?</a>
        </div>

        <button className="primary" type="submit" style={{ marginTop: '24px' }}>Sign In</button>

        <div className="muted center small" style={{ marginTop: '16px' }}>
          Don’t have an account? <Link to="/signup">Sign up</Link>
        </div>
      </form>

      <p className="tiny center muted" style={{ maxWidth: '400px' }}>
        Protected by attorney-client privilege guidelines. By signing in, you agree to our Terms of Service and Privacy Policy.
      </p>
    </div>
  );
}
