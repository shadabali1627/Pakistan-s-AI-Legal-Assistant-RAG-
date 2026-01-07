import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../auth";
import { streamAnswer } from "../api";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  Send, User, Bot, Menu, Plus, MessageSquare, StopCircle,
  Mic, Paperclip, FileText, ChevronRight, Settings, LogOut
} from "lucide-react";

// --- Components ---

function ResourcePanel({ isOpen, citations }) {
  return (
    <div className={`resource-panel ${isOpen ? "" : "closed"}`}>
      <div className="resource-title">Legal Resources & Citations</div>
      {citations.length > 0 ? (
        citations.map((cite, idx) => (
          <div key={idx} className="citation-card" title={cite.text || "View Source"}>
            <div style={{ fontWeight: 600, color: '#1E40AF', marginBottom: '4px' }}>
              {cite.title || "Legal Document"}
            </div>
            <div style={{ fontSize: '0.8rem', color: '#64748B' }}>
              {cite.source ? cite.source.split(/[\\/]/).pop() : "Unknown Source"} • Page {cite.page || "1"}
            </div>
            {cite.score && (
              <div style={{ fontSize: '0.75rem', color: '#059669', marginTop: '6px', fontWeight: 500 }}>
                Match Confidence: {Math.round(cite.score * 100)}%
              </div>
            )}
          </div>
        ))
      ) : (
        <div style={{ color: '#94A3B8', fontSize: '0.9rem', fontStyle: 'italic', textAlign: 'center', marginTop: '40px' }}>
          Relevant case law, statutes, and citations will appear here after your query.
        </div>
      )}
    </div>
  );
}

function Bubble({ role, content, status, isStreaming }) {
  const isUser = role === "user";
  const hasContent = (content || "").trim().length > 0;

  return (
    <div className={`msg-row`}>
      <div className={`msg-container ${isUser ? "me" : ""}`}>
        {!isUser && (
          <div className="avatar">
            <Bot size={20} />
          </div>
        )}

        <div className={`bubble ${isUser ? "user" : "assistant"}`}>
          {/* Status for AI (Thinking/Searching) */}
          {!isUser && status && !hasContent && (
            <div className="status-indicator">
              <div className="status-dot"></div>
              <span className="status-text">{status}</span>
            </div>
          )}

          {isUser ? (
            <div>{content}</div>
          ) : (
            <div className="markdown-content">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {content}
              </ReactMarkdown>
              {isStreaming && <span className="cursor-blink" />}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

const WELCOME_MESSAGE = {
  role: "assistant",
  text: "Salam! I am your advanced AI Legal Assistant. I can help you research Pakistani case law, drafting legal documents, and understanding statutes. How can I assist you today?",
  citations: [],
};

function createNewChat() {
  return {
    id: crypto.randomUUID(),
    title: "New Legal Conversation",
    messages: [WELCOME_MESSAGE],
  };
}

export default function Chat() {
  const { user, signout } = useAuth();
  const nav = useNavigate();

  // Layout State
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isResourceOpen, setIsResourceOpen] = useState(true);
  const [isMobile, setIsMobile] = useState(false);

  // Chat State
  const [chatHistory, setChatHistory] = useState(() => [createNewChat()]);
  const [activeChatId, setActiveChatId] = useState(chatHistory[0].id);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  // Refs
  const bottomRef = useRef(null);
  const abortControllerRef = useRef(null);
  const profileRef = useRef(null);
  const [settingsOpen, setSettingsOpen] = useState(false);

  // Responsive Check
  useEffect(() => {
    const checkMobile = () => {
      const mobile = window.innerWidth <= 900;
      setIsMobile(mobile);
      if (mobile) {
        setIsSidebarOpen(false);
        setIsResourceOpen(false);
      }
    };
    checkMobile();
    window.addEventListener('resize', checkMobile);
    return () => window.removeEventListener('resize', checkMobile);
  }, []);

  const activeChat = useMemo(() =>
    chatHistory.find(c => c.id === activeChatId) || chatHistory[0],
    [chatHistory, activeChatId]);

  // Combined Citations for Resource Panel
  const allCitations = useMemo(() => {
    return activeChat.messages
      .filter(m => m.citations && m.citations.length > 0)
      .flatMap(m => m.citations);
  }, [activeChat]);

  // Scroll to bottom
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [activeChat.messages, loading]);

  function stopGeneration() {
    abortControllerRef.current?.abort();
    setLoading(false);
  }

  function onNewChat() {
    const newChat = createNewChat();
    setChatHistory(prev => [newChat, ...prev]);
    setActiveChatId(newChat.id);
    setInput("");
    if (isMobile) setIsSidebarOpen(false);
  }

  async function send() {
    if (!input.trim() || loading) return;

    const userMsg = { role: "user", text: input };
    const botMsg = { role: "assistant", text: "", citations: [], status: "Initializing..." };

    const updatedHistory = chatHistory.map(chat =>
      chat.id === activeChatId
        ? { ...chat, messages: [...chat.messages, userMsg, botMsg] }
        : chat
    );
    setChatHistory(updatedHistory);
    setInput("");
    setLoading(true);

    abortControllerRef.current = new AbortController();

    try {
      const apiHistory = activeChat.messages.map(m => ({ role: m.role, content: m.text }));
      apiHistory.push({ role: 'user', content: userMsg.text }); // Add current

      let acc = "";
      let currentCitations = [];
      let currentStatus = "Analyzing...";

      for await (const event of streamAnswer(userMsg.text, apiHistory, abortControllerRef.current.signal)) {
        if (event.type === 'content') {
          acc += event.data;
          currentStatus = "";
        } else if (event.type === 'citations') {
          currentCitations = event.data;
        } else if (event.type === 'status') {
          currentStatus = event.data;
        } else if (event.type === 'error') {
          acc += `\n\n**Error:** ${event.data}`;
        }

        setChatHistory(prev => prev.map(chat => {
          if (chat.id !== activeChatId) return chat;
          const newMsgs = [...chat.messages];
          newMsgs[newMsgs.length - 1] = {
            ...newMsgs[newMsgs.length - 1],
            text: acc,
            citations: currentCitations,
            status: currentStatus
          };
          return { ...chat, messages: newMsgs };
        }));
      }
    } catch (e) {
      if (e.name !== 'AbortError') console.error(e);
    } finally {
      setLoading(false);
      abortControllerRef.current = null;
    }
  }

  return (
    <div className={`chat-layout ${!isSidebarOpen ? "collapsed" : ""}`}>
      {/* --- Sidebar --- */}
      <aside className={`sidebar ${isSidebarOpen ? "open" : ""}`}>
        <div className="brand-row">
          <div style={{ width: 32, height: 32, borderRadius: 8, background: '#1E40AF', color: 'white', display: 'grid', placeItems: 'center', fontWeight: 'bold' }}>S</div>
          <div>
            <div className="app-title">AI Legal Assistant</div>
            <div className="badge">PRO</div>
          </div>
        </div>

        <button className="new-chat" onClick={onNewChat}>
          <Plus size={18} /> New Conversation
        </button>

        <div className="convo-list">
          {chatHistory.map(chat => (
            <div
              key={chat.id}
              className={`convo-item ${chat.id === activeChatId ? 'active' : ''}`}
              onClick={() => { setActiveChatId(chat.id); if (isMobile) setIsSidebarOpen(false); }}
            >
              {chat.title}
            </div>
          ))}
        </div>

        <div className="profile" ref={profileRef} style={{ position: 'relative' }}>
          {settingsOpen && (
            <div style={{ position: 'absolute', bottom: '60px', left: '10px', width: '260px', background: '#1E293B', border: '1px solid #334155', borderRadius: '8px', padding: '8px', zIndex: 100 }}>
              <button
                onClick={() => { signout(); nav('/login'); }}
                style={{ display: 'flex', gap: 8, alignItems: 'center', width: '100%', padding: 10, background: 'transparent', border: 'none', color: '#EF4444', cursor: 'pointer', fontWeight: 500 }}
              >
                <LogOut size={16} /> Log Out
              </button>
            </div>
          )}
          <div className="avatar"><User size={16} /></div>
          <div style={{ flex: 1, overflow: 'hidden' }}>
            <div style={{ fontWeight: 500 }}>{user?.name || "Legal Professional"}</div>
            <div style={{ fontSize: '0.75rem', opacity: 0.7 }}>{user?.email}</div>
          </div>
          <button style={{ background: 'transparent', border: 'none', color: 'white', cursor: 'pointer' }} onClick={() => setSettingsOpen(!settingsOpen)}>
            <Settings size={18} />
          </button>
        </div>
      </aside>

      {/* --- Main Area --- */}
      <div className="main">
        <div className="topbar">
          <button className="hamburger" onClick={() => setIsSidebarOpen(!isSidebarOpen)}>
            <Menu size={20} />
          </button>

          <div className="status-indicator">
            {loading && <><div className="status-dot"></div><span className="status-text">Processing Legal Inquiry...</span></>}
          </div>

          <button style={{ background: 'transparent', border: 'none', color: '#64748B', cursor: 'pointer' }} onClick={() => setIsResourceOpen(!isResourceOpen)} title="Toggle Citations">
            <FileText size={20} color={isResourceOpen ? '#1E40AF' : 'currentColor'} />
          </button>
        </div>

        <div className="messages">
          {activeChat.messages.map((msg, i) => (
            <Bubble
              key={i}
              role={msg.role}
              content={msg.text}
              status={msg.status}
              isStreaming={loading && i === activeChat.messages.length - 1}
            />
          ))}
          <div ref={bottomRef} style={{ height: 1 }} />
        </div>

        <div className="composer-area">
          <div className={`composer-box ${loading ? "pulsing" : ""}`}>
            <button className="action-btn" title="Attach Document"><Paperclip size={20} /></button>
            <input
              className="composer-input"
              placeholder="Ask a question about Pakistani law..."
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && !loading && send()}
              disabled={loading}
            />
            <button className="action-btn" title="Voice"><Mic size={20} /></button>

            {loading ? (
              <button className="action-btn stop-btn" onClick={stopGeneration}><StopCircle size={20} /></button>
            ) : (
              <button className="action-btn send-btn" onClick={send} disabled={!input.trim()}><Send size={18} /></button>
            )}
          </div>
          <div className="disclaimer">
            AI-generated insights are for informational purposes only and do not constitute legal advice.
          </div>
        </div>
      </div>

      {/* --- Resource Panel --- */}
      <ResourcePanel isOpen={isResourceOpen} citations={allCitations} />
    </div>
  );
}