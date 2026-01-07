import { Link, useNavigate } from "react-router-dom";
import { useState } from "react";
import { useAuth } from "../auth";
import { GoogleLogin } from '@react-oauth/google';

export default function Signup() {
  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [agreed, setAgreed] = useState(false);

  const { signup, googleLogin } = useAuth();
  const nav = useNavigate();

  const handleGoogleSuccess = async (credentialResponse) => {
    try {
      await googleLogin(credentialResponse.credential);
      nav("/chat");
    } catch (e) {
      alert("Google Sign-Up Failed");
    }
  };

  const handleGoogleError = () => {
    alert("Google Sign-Up Failed");
  };

  const onSubmit = async (e) => {
    e.preventDefault();
    if (password !== confirmPassword) {
      alert("Passwords do not match");
      return;
    }
    if (!agreed) {
      alert("Please agree to the Terms of Service");
      return;
    }
    // Note: The original auth context might only accept email/password. 
    // We will send what we can or expand the auth context later if needed.
    // For now, we simulate full signup by passing email/password as before.
    try {
      await signup(email, password, fullName);
      nav("/chat");
    } catch (err) {
      alert("Sign up failed: " + err.message);
    }
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
          <h2 style={{ fontSize: '1.5rem', fontWeight: '600', marginBottom: '24px', textAlign: 'center', color: 'var(--text-main)' }}>Create Account</h2>

          <div className="form-group">
            <label className="form-label">Full Name</label>
            <input
              className="form-input"
              placeholder="John Doe"
              type="text"
              value={fullName}
              onChange={(e) => setFullName(e.target.value)}
              required
            />
          </div>

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
            <label className="form-label">Password</label>
            <input
              className="form-input"
              placeholder="••••••••"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
          </div>

          <div className="form-group">
            <label className="form-label">Confirm Password</label>
            <input
              className="form-input"
              placeholder="••••••••"
              type="password"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              required
            />
          </div>

          <div className="checkbox-wrapper">
            <input
              type="checkbox"
              id="terms"
              checked={agreed}
              onChange={(e) => setAgreed(e.target.checked)}
            />
            <label htmlFor="terms" style={{ cursor: 'pointer', display: 'inline' }}>
              I agree to the <Link to="#">Terms of Service</Link> and <Link to="#">Privacy Policy</Link>
            </label>
          </div>

          <button className="btn btn-primary" type="submit">Create Account</button>
        </form>

        <div style={{ display: 'flex', justifyContent: 'center', marginTop: '16px', width: '100%' }}>
          <GoogleLogin
            onSuccess={handleGoogleSuccess}
            onError={handleGoogleError}
            shape="rectangular"
            theme="outline"
            size="large"
            width="100%"
            text="signup_with"
          />
        </div>

        <div className="auth-footer">
          Already have an account? <Link to="/login">Sign in</Link>
        </div>
      </div>
    </div>
  );
}
