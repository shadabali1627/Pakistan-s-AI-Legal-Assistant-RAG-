import { useSyncExternalStore } from "react";

const LS_KEY = "ala_user";

// In-memory cached user (stable reference)
let currentUser = readFromStorage();

function readFromStorage() {
  try {
    return JSON.parse(localStorage.getItem(LS_KEY) || "null");
  } catch {
    return null;
  }
}

const listeners = new Set(); // ✅ JS (no generics)

function emit() {
  for (const l of Array.from(listeners)) l();
}

function subscribe(cb) {
  listeners.add(cb);
  return () => listeners.delete(cb);
}

function getSnapshot() {
  return currentUser; // stable reference unless changed
}

function getServerSnapshot() {
  return null;
}

export function useAuth() {
  const user = useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);

  return {
    user,
    signin: async (email, password) => {
      try {
        const res = await fetch('http://localhost:8000/api/auth/login', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ email, password })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Login failed");

        currentUser = {
          email: data.user.email,
          name: data.user.name,
          picture: data.user.picture
        };
        localStorage.setItem(LS_KEY, JSON.stringify(currentUser));
        emit();
        return currentUser;
      } catch (err) {
        throw err;
      }
    },
    googleLogin: async (token) => {
      try {
        const res = await fetch('http://localhost:8000/api/auth/google', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ token })
        });
        const data = await res.json();

        if (data.status === 'success') {
          currentUser = {
            email: data.user.email,
            name: data.user.name,
            picture: data.user.picture
          };
          localStorage.setItem(LS_KEY, JSON.stringify(currentUser));
          emit();
          return currentUser;
        } else {
          throw new Error("Google Auth Failed");
        }
      } catch (err) {
        console.error(err);
        throw err;
      }
    },
    signup: async (email, password, fullName = "User") => {
      try {
        const res = await fetch('http://localhost:8000/api/auth/signup', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ email, password, full_name: fullName })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Signup failed");

        currentUser = {
          email: data.user.email,
          name: data.user.name
        };
        localStorage.setItem(LS_KEY, JSON.stringify(currentUser));
        emit();
        return currentUser;
      } catch (err) {
        throw err;
      }
    },
    resetPassword: async (email, newPassword) => {
      try {
        const res = await fetch('http://localhost:8000/api/auth/reset-password', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ email, new_password: newPassword })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Reset failed");
        return true;
      } catch (err) {
        throw err;
      }
    },
    signout: () => {
      if (currentUser !== null) {
        currentUser = null;
        localStorage.removeItem(LS_KEY);
        emit();
      }
    },
  };
}
