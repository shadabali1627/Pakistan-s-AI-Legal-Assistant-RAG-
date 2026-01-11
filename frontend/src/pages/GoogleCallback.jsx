import { useEffect, useRef } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { useAuth } from '../hooks/useAuth';
import { Loader2 } from 'lucide-react';

export default function GoogleCallback() {
    const [searchParams] = useSearchParams();
    const { verifyGoogleCode } = useAuth();
    const nav = useNavigate();
    const processedRef = useRef(false);

    useEffect(() => {
        const code = searchParams.get('code');
        const error = searchParams.get('error');

        if (error) {
            console.error("Google Auth Error from callback:", error);
            nav('/login');
            return;
        }

        if (code && !processedRef.current) {
            processedRef.current = true; // Prevent double-execution in Strict Mode

            const handleCallback = async () => {
                try {
                    // The redirect_uri must EXACTLY match what was registered and sent during the auth request.
                    const redirectUri = window.location.origin + '/auth/callback';

                    const intent = localStorage.getItem("auth_intent");
                    const isSignup = intent === "signup";

                    await verifyGoogleCode(code, redirectUri, isSignup);

                    // Clear intent after success
                    localStorage.removeItem("auth_intent");
                    nav('/chat');
                } catch (err) {
                    console.error("Callback verification failed", err);

                    const errorMessage = err.response?.data?.detail || "Authentication failed";

                    // If intent was signup and user exists, redirect back to signup with error
                    if (localStorage.getItem("auth_intent") === "signup" && errorMessage.toLowerCase().includes("account already exists")) {
                        nav('/signup', { state: { error: errorMessage } });
                    } else {
                        // Default fallback
                        nav('/login', { state: { error: errorMessage } });
                    }
                    localStorage.removeItem("auth_intent");
                }
            };

            handleCallback();
        } else if (!code && !processedRef.current) {
            // specific case: no code found, maybe accidental navigation
            nav('/login');
        }
    }, [searchParams, verifyGoogleCode, nav]);

    return (
        <div className="min-h-screen flex flex-col items-center justify-center bg-gray-50">
            <div className="flex flex-col items-center gap-4">
                <Loader2 className="w-12 h-12 text-blue-600 animate-spin" />
                <h2 className="text-xl font-semibold text-slate-700">Authenticating...</h2>
                <p className="text-slate-500 text-sm">Please wait while we log you in.</p>
            </div>
        </div>
    );
}
