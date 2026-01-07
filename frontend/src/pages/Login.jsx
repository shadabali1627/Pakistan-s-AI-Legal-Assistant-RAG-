import { Link, useNavigate } from "react-router-dom";
import { useState } from "react";
import { useAuth } from "../auth";
import { GoogleLogin } from '@react-oauth/google';

export default function Login() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const { signin, googleLogin } = useAuth();
  const nav = useNavigate();

  const onSubmit = async (e) => {
    e.preventDefault();
    try {
      await signin(email, password);
      nav("/chat");
    } catch (err) {
      alert("Login failed: " + err.message);
    }
  };

  const handleGoogleSuccess = async (credentialResponse) => {
    try {
      await googleLogin(credentialResponse.credential);
      nav("/chat");
    } catch (e) {
      alert("Google Login Failed");
    }
  };

  const handleGoogleError = () => {
    alert("Google Login Failed");
  };

  return (
    <div className="auth-container">
      <div className="auth-header">
        <div className="brand-logo-placeholder">AI</div>
        <h1 className="auth-title">AI Legal Assistant</h1>
        <p className="auth-subtitle">Your intelligent legal companion</p>
      </div>

      <div className="auth-card">
        <form onSubmit={onSubmit}>
          <h2 style={{ fontSize: '1.5rem', fontWeight: '600', marginBottom: '24px', textAlign: 'center', color: 'var(--text-main)' }}>Sign In</h2>

          <div className="form-group">
            <label className="form-label">Email</label>
            <input
              className="form-input"
              placeholder="name@company.com"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
          </div>

          <div className="form-group">
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
              <label className="form-label" style={{ marginBottom: 0 }}>Password</label>
              <Link to="/forgot-password" style={{ fontSize: '0.875rem', fontWeight: '500' }}>Forgot password?</Link>
            </div>
            <input
              className="form-input"
              placeholder="••••••••"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
          </div>

          <button className="btn btn-primary" type="submit">Sign In</button>
        </form>

        <div style={{ display: 'flex', justifyContent: 'center', marginTop: '16px', width: '100%' }}>
          <GoogleLogin
            onSuccess={handleGoogleSuccess}
            onError={handleGoogleError}
            shape="rectangular"
            theme="outline"
            size="large"
            width="100%"
          />
        </div>

        <div className="auth-footer">
          Don't have an account? <Link to="/signup">Sign up</Link>
        </div>
      </div>

      <div className="auth-footer" style={{ marginTop: '32px', maxWidth: '400px', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
        <p>Protected by attorney-client privilege guidelines.</p>
      </div>
    </div>
  );
}
