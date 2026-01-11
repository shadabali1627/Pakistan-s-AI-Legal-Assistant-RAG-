import { useState, useEffect, useContext, createContext } from "react";
import { googleLogout, useGoogleLogin, useGoogleOneTapLogin } from "@react-oauth/google";
import { api } from "../services/api";

const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(true);
    const [googleLoading, setGoogleLoading] = useState(false);

    // Check if user is already logged in
    useEffect(() => {
        const checkAuth = async () => {
            try {
                const token = localStorage.getItem("token");
                if (token) {
                    const savedUser = JSON.parse(localStorage.getItem("user_info") || "null");
                    if (savedUser) setUser(savedUser);
                    api.defaults.headers.common["Authorization"] = `Bearer ${token}`;
                }
            } catch (error) {
                console.error("Auth check failed", error);
                localStorage.removeItem("token");
            } finally {
                setLoading(false);
            }
        };
        checkAuth();
    }, []);

    // --- Actions ---

    const signin = async (email, password) => {
        try {
            const response = await api.post("/auth/login", { email, password });
            const { access_token, user: userData } = response.data;

            localStorage.setItem("token", access_token);
            localStorage.setItem("user_info", JSON.stringify(userData));
            api.defaults.headers.common["Authorization"] = `Bearer ${access_token}`;

            setUser(userData);
            return userData;
        } catch (error) {
            console.error("Login failed", error);
            throw new Error(error.response?.data?.detail || "Login failed");
        }
    };

    const signup = async (email, password, name) => {
        try {
            await api.post("/auth/signup", { email, password, full_name: name });
            return await signin(email, password);
        } catch (error) {
            throw new Error(error.response?.data?.detail || "Signup failed");
        }
    };

    const resetPassword = async (email, newPassword) => {
        try {
            await api.post("/auth/reset-password", { email, new_password: newPassword });
            return true;
        } catch (error) {
            throw new Error(error.response?.data?.detail || "Reset failed");
        }
    };

    const signout = () => {
        googleLogout();
        localStorage.removeItem("token");
        localStorage.removeItem("user_info");
        delete api.defaults.headers.common["Authorization"];
        setUser(null);
    };

    // --- Google Auth Handlers ---

    // 1. Verify Google Code (Redirect Flow)
    const verifyGoogleCode = async (code, redirectUri, isSignup = false) => {
        setGoogleLoading(true);
        try {
            console.log("Verifying Google Code...", { code: "REDACTED", redirectUri, isSignup });
            const res = await api.post("/auth/google", {
                code: code,
                redirect_uri: redirectUri,
                is_signup: isSignup
            });
            const { access_token, user: userData } = res.data;

            localStorage.setItem("token", access_token);
            localStorage.setItem("user_info", JSON.stringify(userData));
            api.defaults.headers.common["Authorization"] = `Bearer ${access_token}`;

            setUser(userData);
            return userData;
        } catch (err) {
            console.error("Redirect verify failed", err);
            throw err;
        } finally {
            setGoogleLoading(false);
        }
    }

    // 2. Initialize Manual Button (Redirect Flow)
    const loginWithGoogle = useGoogleLogin({
        flow: 'auth-code',
        ux_mode: 'redirect',
        redirect_uri: window.location.origin + '/auth/callback',
        onError: (err) => {
            console.error("Google Login Error:", err);
            setGoogleLoading(false);
        },
        onNonOAuthError: () => {
            setGoogleLoading(false);
        }
    });

    const handleGoogleLoginTrigger = () => {
        setGoogleLoading(true);
        loginWithGoogle();
    };

    const value = {
        user,
        loading,
        signin,
        signup,
        signout,
        googleLogin: handleGoogleLoginTrigger,
        googleLoading,
        verifyGoogleCode,
        resetPassword,
    };

    return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

export const useAuth = () => {
    return useContext(AuthContext);
};
