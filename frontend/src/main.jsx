import React from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import App from "./App.jsx";
import "./index.css";

// --- FIX FOR HTTP DEPLOYMENT (Polyfill for crypto.randomUUID) ---
// Browsers block randomUUID on insecure HTTP. This manually creates it.
if (!window.crypto) {
  window.crypto = {};
}
if (!window.crypto.randomUUID) {
  window.crypto.randomUUID = function () {
    return "10000000-1000-4000-8000-100000000000".replace(/[018]/g, c =>
      (c ^ (Math.random() * 16) >> c / 4).toString(16)
    );
  };
}
// -------------------------------------------------------------

import { GoogleOAuthProvider } from '@react-oauth/google';
import { AuthProvider } from './hooks/useAuth';

const GOOGLE_CLIENT_ID = import.meta.env.VITE_GOOGLE_CLIENT_ID;

createRoot(document.getElementById("root")).render(
  <GoogleOAuthProvider clientId={GOOGLE_CLIENT_ID}>
    <AuthProvider>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </AuthProvider>
  </GoogleOAuthProvider>
);