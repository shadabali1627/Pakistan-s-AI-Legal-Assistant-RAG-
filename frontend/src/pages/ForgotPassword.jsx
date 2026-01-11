import { Link, useNavigate } from "react-router-dom";
import { useState } from "react";
import { useAuth } from "../hooks/useAuth";

export default function ForgotPassword() {
    const [email, setEmail] = useState("");
    const [newPassword, setNewPassword] = useState("");
    const [confirmPassword, setConfirmPassword] = useState("");
    const { resetPassword } = useAuth();
    const nav = useNavigate();

    const onSubmit = async (e) => {
        e.preventDefault();
        if (newPassword !== confirmPassword) {
            alert("Passwords do not match");
            return;
        }

        try {
            await resetPassword(email, newPassword);
            alert("Password reset successfully! Please login.");
            nav("/login");
        } catch (err) {
            alert("Reset failed: " + err.message);
        }
    };

    return (
        <div className="auth-container">
            <div className="auth-header">
                <div className="brand-logo-placeholder">AI</div>
                <h1 className="auth-title">Reset Password</h1>
                <p className="auth-subtitle">Enter your email and new password</p>
            </div>

            <div className="auth-card">
                <form onSubmit={onSubmit}>
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
                        <label className="form-label">New Password</label>
                        <input
                            className="form-input"
                            placeholder="••••••••"
                            type="password"
                            value={newPassword}
                            onChange={(e) => setNewPassword(e.target.value)}
                            required
                        />
                    </div>

                    <div className="form-group">
                        <label className="form-label">Confirm New Password</label>
                        <input
                            className="form-input"
                            placeholder="••••••••"
                            type="password"
                            value={confirmPassword}
                            onChange={(e) => setConfirmPassword(e.target.value)}
                            required
                        />
                    </div>

                    <button className="btn btn-primary" type="submit">Reset Password</button>
                </form>

                <div className="auth-footer">
                    Remember your password? <Link to="/login">Sign in</Link>
                </div>
            </div>
        </div>
    );
}
