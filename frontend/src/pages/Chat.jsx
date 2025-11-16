import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../auth";
import { streamAnswer } from "../api";

// --- SVG Icons ---
function CopyIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true" focusable="false">
      <rect x="3" y="3" width="12" height="12" rx="2" ry="2" fill="none" stroke="currentColor" strokeWidth="1.8" />
      <rect x="9" y="9" width="12" height="12" rx="2" ry="2" fill="none" stroke="currentColor" strokeWidth="1.8" />
    </svg>
  );
}
function CheckIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true" focusable="false">
      <path d="M20 6L9 17l-5-5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
function MenuIcon() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" aria-hidden="true" focusable="false">
      <path d="M3 6h18M3 12h18M3 18h18" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
    </svg>
  );
}
function SettingsIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12.22 2h-4.44l-1.94 4.5-.44.9-.86.2a10 10 0 00-3.16 2.06l-.68.68-.86-.2-1.94-4.5v4.44l4.5 1.94.9.44.2.86a10 10 0 002.06 3.16l.68.68-.2.86-4.5 1.94v4.44l4.5-1.94.9-.44.2-.86a10 10 0 003.16-2.06l.68-.68.86.2 1.94 4.5h4.44l1.94-4.5.44-.9.86-.2a10 10 0 003.16-2.06l.68-.68.86.2 1.94 4.5v-4.44l-4.5-1.94-.9-.44-.2-.86a10 10 0 00-2.06-3.16l-.68-.68.2-.86 4.5-1.94V2l-4.5 1.94-.9.44-.2.86a10 10 0 00-3.16 2.06l-.68.68-.86-.2L12.22 2z"/>
      <circle cx="12" cy="12" r="3" />
    </svg>
  );
}
// --- End SVG Icons ---

function Bubble({ role, text }) {
  const isUser = role === "user";
  const [copied, setCopied] = useState(false);
  const hasText = (text ?? "").trim().length > 0;

  async function onCopy() {
    try {
      await navigator.clipboard.writeText(text || "");
      setCopied(true);
      setTimeout(() => setCopied(false), 900);
    } catch {}
  }

  return (
    <div className={`msg ${isUser ? "me" : "bot"}`}>
      {!isUser && <div className="avatar">⚖️</div>}
      <div className={`bubble ${isUser ? "user" : "assistant"}`}>
        {!isUser && hasText && (
          <button className="copy" onClick={onCopy} title="Copy" type="button">
            {copied ? <CheckIcon /> : <CopyIcon />}
          </button>
        )}
        {text}
      </div>
    </div>
  );
}

// const QUICK = ["Family Law", "Criminal Law", "Property Law", "Constitution"]; // <-- REMOVED
const WELCOME_MESSAGE = {
  role: "assistant",
  text:
    "السلام علیکم! I'm your AI Legal Assistant for Pakistan law. I can help you with questions about Pakistani family law, criminal law, property law, constitutional matters, and more. How can I assist you today?",
};

// Helper to create a new chat object
function createNewChat() {
  return {
    id: crypto.randomUUID(),
    title: "New Conversation",
    messages: [WELCOME_MESSAGE],
  };
}

export default function Chat() {
  const { user, signout } = useAuth();
  const nav = useNavigate();

  // --- responsive sidebar state ---
  const mq = "(max-width: 900px)";
  const [isMobile, setIsMobile] = useState(() => window.matchMedia(mq).matches);
  const [sidebarOpen, setSidebarOpen] = useState(() => !window.matchMedia(mq).matches);

  useEffect(() => {
    const mql = window.matchMedia(mq);
    const onChange = (e) => {
      setIsMobile(e.matches);
      setSidebarOpen(!e.matches); // open on desktop, close on mobile
    };
    mql.addEventListener("change", onChange);
    return () => mql.removeEventListener("change", onChange);
  }, []);

  // --- CHAT HISTORY STATE ---
  const [chatHistory, setChatHistory] = useState(() => [createNewChat()]);
  const [activeChatId, setActiveChatId] = useState(chatHistory[0].id);
  
  const [draft, setDraft] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef(null);

  // --- NEW SETTINGS MENU STATE ---
  const [settingsOpen, setSettingsOpen] = useState(false);
  const profileRef = useRef(null);

  useEffect(() => { if (!user) nav("/login"); }, [user, nav]);

  // --- Effect to close settings menu on click outside ---
  useEffect(() => {
    function handleClickOutside(event) {
      if (profileRef.current && !profileRef.current.contains(event.target)) {
        setSettingsOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [profileRef]);

  // --- Memoized active chat ---
  const activeChat = useMemo(() => {
    // Find the active chat, or default to the first one if not found
    return chatHistory.find(c => c.id === activeChatId) || chatHistory[0];
  }, [chatHistory, activeChatId]);

  const title = useMemo(() => {
    // Use the active chat's title
    return activeChat?.title || "AI Legal Assistant";
  }, [activeChat]);

  // Scroll to bottom when messages in the active chat change
  useEffect(() => { 
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [activeChat?.messages, loading]);

  // --- Handle New Chat Button ---
  function onNewChat() {
    const newChat = createNewChat();
    setChatHistory(prev => [newChat, ...prev]); // Add new chat to the top of the list
    setActiveChatId(newChat.id);
    setDraft("");
    setLoading(false);
    if (isMobile) setSidebarOpen(false); // Close sidebar on mobile
  }

  async function send(query) {
    if (!query.trim() || loading) return;
    
    const userMessage = { role: "user", text: query };
    const botMessage = { role: "assistant", text: "" };
    
    // Update state immutably: find the active chat and add new messages
    setChatHistory(prevHistory => 
      prevHistory.map(chat => 
        chat.id === activeChatId
          ? { ...chat, messages: [...chat.messages, userMessage, botMessage] }
          : chat
      )
    );
    
    setDraft("");
    setLoading(true);

    try {
      let acc = "";
      for await (const token of streamAnswer(query)) {
        acc += token ?? "";
        
        // Update state immutably: find active chat, find last message, update its text
        setChatHistory(prevHistory => 
          prevHistory.map(chat => 
            chat.id === activeChatId
              ? { 
                  ...chat, 
                  messages: [
                    ...chat.messages.slice(0, -1), // all messages except the last one
                    { ...chat.messages[chat.messages.length - 1], text: acc } // update the last message
                  ] 
                }
              : chat
          )
        );
      }
    } catch (e) {
      console.error(e);
      // Handle error: update the last message with an error text
      setChatHistory(prevHistory => 
        prevHistory.map(chat => 
          chat.id === activeChatId
            ? { 
                ...chat, 
                messages: [
                  ...chat.messages.slice(0, -1),
                  { role: "assistant", text: "Sorry, I couldn't complete that request." }
                ] 
              }
            : chat
        )
      );
    } finally {
      setLoading(false);
      
      // --- Auto-title the chat ---
      // If title is "New Conversation" and we have a user prompt, set title
      setChatHistory(prevHistory =>
        prevHistory.map(chat =>
          (chat.id === activeChatId && chat.title === "New Conversation" && chat.messages.length > 2)
            ? { ...chat, title: chat.messages[1].text.substring(0, 40) + "..." } // Title from user's first prompt
            : chat
        )
      );
    }
  }

  const onSubmit = (e) => {
    e.preventDefault();
    send(draft);
  };

  return (
    <>
      {/* fixed top-left hamburger (always visible) */}
      <button
        className="hamburger fixed-left"
        aria-label="Toggle sidebar"
        onClick={() => setSidebarOpen((v) => !v)}
        type="button"
      >
        <MenuIcon />
      </button>

      <div className={`chat-layout ${!sidebarOpen ? "collapsed" : ""}`}>
        {/* Sidebar */}
        <aside className={`sidebar ${sidebarOpen ? "open" : ""}`}>
          <div className="brand-row">
            <div className="logo sm">⚖️</div>
            <div>
              <div className="app-title">AI Legal Assistant</div>
              <div className="badge">Pakistan Law</div>
            </div>
          </div>

          <button className="new-chat" type="button" onClick={onNewChat}>
            ＋ New Chat
          </button>

          {/* CHAT HISTORY LIST */}
          <div className="convo">
            {chatHistory.map(chat => (
              <div 
                key={chat.id} 
                className={`convo-item ${chat.id === activeChatId ? 'active' : ''}`}
                onClick={() => {
                  setActiveChatId(chat.id);
                  if (isMobile) setSidebarOpen(false); // Close drawer on selection
                }}
              >
                <span>{chat.title}</span>
                <span className="muted small">{chat.messages.length} messages</span>
              </div>
            ))}
          </div>

          {/* --- UPDATED PROFILE SECTION --- */}
          <div className="profile" ref={profileRef}>
          
            {/* 1. THE SETTINGS MENU (POPS UP) */}
            {settingsOpen && (
              <div className="settings-menu">
                <button
                  className="settings-menu-item"
                  onClick={() => {
                    signout();
                    nav("/login");
                  }}
                  type="button"
                >
                  Log Out
                </button>
              </div>
            )}

            {/* 2. THE AVATAR (Same as before) */}
            <div className="avatar circle">{(user?.name?.[0] || "U").toUpperCase()}</div>
            
            {/* 3. THE USER META (Same as before) */}
            <div className="profile-meta">
              <div>{user?.name || "User"}</div>
              <div className="muted small">{user?.email}</div>
            </div>

            {/* 4. THE NEW SETTINGS BUTTON (Replaces eject button) */}
            <button className="ghost" onClick={() => setSettingsOpen(v => !v)} type="button">
              <SettingsIcon />
            </button>
          </div>
          {/* --- END UPDATED PROFILE SECTION --- */}

        </aside>

        {/* Backdrop for mobile drawer */}
        {isMobile && sidebarOpen && (
          <div className="backdrop show" onClick={() => setSidebarOpen(false)} />
        )}

        {/* Main */}
        <main className="main">
          <header className="topbar">
            <div className="title">{title}</div>
            <button className="ghost" title="Open in new window" type="button">⤢</button>
          </header>

          <section className="messages">
            {activeChat?.messages.map((m, i) => <Bubble key={i} role={m.role} text={m.text} />)}
            <div ref={bottomRef} />
          </section>

          {/* --- THIS ENTIRE SECTION HAS BEEN REMOVED --- */}
          {/* <section className="quick">
            <div className="chips">
              {QUICK.map(q => (
                <button key={q} className="chip" onClick={() => send(`Tell me about ${q}`)} type="button">
                  {q}
                </button>
              ))}
            </div>
          </section>
          */}
          {/* --- END REMOVED SECTION --- */}

          <form className="composer" onSubmit={onSubmit}>
            <input
              placeholder="Ask a question about Pakistan law..."
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              disabled={loading}
            />
            <button className="send" disabled={loading} title="Send" type="submit">➤</button>
          </form>

          <p className="disclaimer">
            This AI assistant provides general information about Pakistan law and does not constitute legal advice.
            Please consult a qualified Pakistani lawyer for specific legal matters.
          </p>
        </main>
      </div>
    </>
  );
}