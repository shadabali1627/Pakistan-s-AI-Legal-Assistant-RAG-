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
    signin: async (email, _password) => {
      const next = { email, name: email.split("@")[0] || "User" };
      const same =
        currentUser &&
        currentUser.email === next.email &&
        currentUser.name === next.name;

      if (!same) {
        currentUser = next;
        localStorage.setItem(LS_KEY, JSON.stringify(currentUser));
        emit();
      }
      return currentUser;
    },
    signup: async (email, _password) => {
      const next = { email, name: email.split("@")[0] || "User" };
      const same =
        currentUser &&
        currentUser.email === next.email &&
        currentUser.name === next.name;

      if (!same) {
        currentUser = next;
        localStorage.setItem(LS_KEY, JSON.stringify(currentUser));
        emit();
      }
      return currentUser;
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
