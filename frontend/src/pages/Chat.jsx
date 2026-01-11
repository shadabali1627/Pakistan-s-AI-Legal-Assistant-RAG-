import { useEffect, useState, useMemo, useRef } from "react";
import ChatLayout from "../components/layout/ChatLayout";
import Modal from "../components/ui/Modal";
import MessageBubble from "../components/features/MessageBubble";
import ChatInput from "../components/features/ChatInput";
import ResourcePanel from "../components/features/ResourcePanel";
import { useChatStream } from "../hooks/useChatStream";
import { useAuth } from "../hooks/useAuth";
import { FileText, BookOpen, Scale, Search } from 'lucide-react';
import { api } from "../services/api";
import clsx from 'clsx';

export default function Chat() {
  const { signout, user } = useAuth();
  const { messages, loading, sendMessage, stopGeneration, resetChat, setMessages } = useChatStream();

  // Layout State
  const [isSidebarOpen, setIsSidebarOpen] = useState(typeof window !== 'undefined' ? window.innerWidth >= 1024 : true);
  const [isResourceOpen, setIsResourceOpen] = useState(true);
  const [input, setInput] = useState("");
  const [chatHistoryList, setChatHistoryList] = useState([]);
  const [activeChatId, setActiveChatId] = useState(null);

  // Auto-scroll logic
  const messagesEndRef = useRef(null);
  const scrollContainerRef = useRef(null);
  const [shouldAutoScroll, setShouldAutoScroll] = useState(true);

  const scrollToBottom = (smooth = true) => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: smooth ? "smooth" : "auto" });
    }
  };

  // Monitor scroll position to determine if we should stick to bottom
  const handleScroll = () => {
    const container = scrollContainerRef.current;
    // ... logic same as previous success ...
    if (!container) return;

    const { scrollTop, scrollHeight, clientHeight } = container;
    // If we are within 100px of the bottom, we are "at the bottom"
    const isAtBottom = scrollHeight - scrollTop - clientHeight < 100;
    setShouldAutoScroll(isAtBottom);
  };

  // Scroll on new messages if we should
  useEffect(() => {
    if (shouldAutoScroll) {
      scrollToBottom();
    }
  }, [messages, loading, shouldAutoScroll]);

  // Force scroll on new session or user message
  useEffect(() => {
    scrollToBottom(false);
    setShouldAutoScroll(true);
  }, [activeChatId]);


  useEffect(() => {
    const checkScreenSize = () => {
      if (window.innerWidth >= 1024) {
        setIsResourceOpen(true);
      } else {
        setIsResourceOpen(false);
        setIsSidebarOpen(false); // Close sidebar on mobile only
      }
    };

    // Initial check
    checkScreenSize();

    // Listen for resize (e.g. Inspect Element -> Toggle Device)
    window.addEventListener('resize', checkScreenSize);
    return () => window.removeEventListener('resize', checkScreenSize);
  }, []);

  const [historyLoading, setHistoryLoading] = useState(false);

  // Fetch History on specific user load
  useEffect(() => {
    if (user?.email) {
      const fetchHistory = async () => {
        setHistoryLoading(true);
        try {
          console.log("Fetching chat history for user:", user.email);
          const res = await api.get(`/chat/history?user_email=${encodeURIComponent(user.email)}`);
          console.log("Chat history response:", res.data);
          setChatHistoryList(res.data);
        } catch (e) {
          console.error("Failed to load history", e);
          console.error("Error details:", e.response?.data);
        } finally {
          setHistoryLoading(false);
        }
      };
      fetchHistory();
    } else {
      console.log("No user email found, user object:", user);
    }
  }, [user]);

  // Handle Chat Selection
  const handleSelectChat = async (id) => {
    setActiveChatId(id);

    try {
      const res = await api.get(`/chat/session/${id}`);
      setMessages(res.data);
      setShouldAutoScroll(true); // Reset scroll on chat switch
    } catch (e) {
      console.error("Failed to load session", e);
    }
  };

  // Derived state for citations
  const allCitations = useMemo(() => {
    return messages
      .filter(m => m.citations && m.citations.length > 0)
      .flatMap(m => m.citations);
  }, [messages]);

  const handleSend = () => {
    if (!input.trim()) return;

    // If no active chat, create one ID
    let currentChatId = activeChatId;
    if (!currentChatId) {
      currentChatId = crypto.randomUUID();
      setActiveChatId(currentChatId);
      // Optimistically add to history
      setChatHistoryList(prev => [{
        id: currentChatId,
        title: input.substring(0, 30) + "...",
        last_interaction: new Date().toISOString()
      }, ...prev]);
    }

    sendMessage(input, messages, user?.email, currentChatId);
    setInput("");
    setShouldAutoScroll(true); // Force scroll on send
  };

  const handleNewChat = () => {
    resetChat();
    setActiveChatId(null);
    setInput("");
  };

  // Delete & Rename Modals State
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [chatToDelete, setChatToDelete] = useState(null);
  const [isRenameModalOpen, setIsRenameModalOpen] = useState(false);
  const [chatToRename, setChatToRename] = useState(null);
  const [renameTitle, setRenameTitle] = useState("");

  const handleDeleteChat = (e, id) => {
    e.stopPropagation();
    setChatToDelete(id);
    setIsDeleteModalOpen(true);
  };

  const confirmDelete = async () => {
    if (!chatToDelete) return;
    try {
      await api.delete(`/chat/session/${chatToDelete}`);
      setChatHistoryList(prev => prev.filter(c => c.id !== chatToDelete));
      if (activeChatId === chatToDelete) {
        setActiveChatId(null);
        resetChat();
      }
      setIsDeleteModalOpen(false);
      setChatToDelete(null);
    } catch (e) {
      console.error("Failed to delete chat", e);
    }
  };

  const handleRenameChat = (e, id, currentTitle) => {
    e.stopPropagation();
    setChatToRename(id);
    setRenameTitle(currentTitle);
    setIsRenameModalOpen(true);
  };

  const confirmRename = async () => {
    if (!chatToRename) return;
    try {
      await api.put(`/chat/session/${chatToRename}`, { title: renameTitle });
      setChatHistoryList(prev => prev.map(c =>
        c.id === chatToRename ? { ...c, title: renameTitle } : c
      ));
      setIsRenameModalOpen(false);
      setChatToRename(null);
    } catch (e) {
      console.error("Failed to rename chat", e);
    }
  };

  return (
    <ChatLayout
      isSidebarOpen={isSidebarOpen}
      setIsSidebarOpen={setIsSidebarOpen}
      user={user}
      onSignOut={signout}
      chatHistory={chatHistoryList}
      activeChatId={activeChatId}
      onSelectChat={handleSelectChat}
      onNewChat={handleNewChat}
      historyLoading={historyLoading}
      onDeleteChat={handleDeleteChat}
      onRenameChat={handleRenameChat}
      isResourceOpen={isResourceOpen}
      setIsResourceOpen={setIsResourceOpen}
    >
      <div className="flex w-full h-full overflow-hidden relative">

        {/* Messages Area */}
        <div className="flex-1 flex flex-col h-full relative">

          <div
            className="flex-1 overflow-y-auto px-3 sm:px-4 lg:px-6 custom-scrollbar scroll-smooth"
            ref={scrollContainerRef}
            onScroll={handleScroll}
          >
            <div className="max-w-4xl mx-auto pt-6 sm:pt-8 pb-10 min-h-full flex flex-col justify-start">
              {messages.length === 0 && (
                <div className="flex-1 flex flex-col justify-center items-center text-center mt-4 lg:-mt-20">

                  {/* Background Accents (Subtle Glow) */}
                  <div className="absolute inset-0 pointer-events-none overflow-hidden">
                    <div className="absolute top-[30%] left-[50%] -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-gradient-to-tr from-blue-50/40 to-indigo-50/40 rounded-full blur-3xl opacity-70"></div>
                  </div>

                  {/* Hero Icon */}
                  <div className="relative z-10 mb-6 sm:mb-8">
                    <div className="w-16 h-16 sm:w-20 sm:h-20 mx-auto bg-gradient-to-br from-white to-blue-50 rounded-2xl shadow-xl shadow-blue-100/50 flex items-center justify-center ring-1 ring-blue-100/50 mb-4 sm:mb-6">
                      <Scale className="w-8 h-8 sm:w-10 sm:h-10 text-blue-600" strokeWidth={1.5} />
                    </div>
                    {/* Main Title */}
                    <h1 className="text-3xl sm:text-4xl lg:text-5xl font-bold text-slate-900 mb-4 sm:mb-6 tracking-tight px-4">
                      <span className="bg-clip-text text-transparent bg-gradient-to-r from-slate-900 via-blue-800 to-slate-900">Pakistan AI</span>
                      <br />
                      <span className="text-blue-600 relative inline-block">
                        Legal Assistant
                        <svg className="absolute w-full h-3 -bottom-1 left-0 text-blue-200 -z-10" viewBox="0 0 100 10" preserveAspectRatio="none"><path d="M0 5 Q 50 10 100 5" stroke="currentColor" strokeWidth="8" fill="none" opacity="0.6" /></svg>
                      </span>
                    </h1>
                    {/* Subtitle */}
                    <p className="text-slate-500 max-w-2xl mx-auto mb-12 sm:mb-16 text-base sm:text-lg leading-relaxed font-light px-4">
                      Your intelligent companion for navigating <span className="text-slate-700 font-medium">Pakistani Laws</span>, analyzing <span className="text-slate-700 font-medium">Case Precedents</span>, and generating <span className="text-slate-700 font-medium">Legal Insights</span>.
                    </p>
                  </div>

                  <div className="hidden"></div>
                </div>
              )}

              {messages.map((msg, i) => (
                <MessageBubble
                  key={msg.id || i}
                  {...msg}
                  citations={msg.citations || []}
                  onCitationClick={(cite) => setIsResourceOpen(true)}
                  userPhoto={user?.picture || user?.photoURL || user?.avatar_url}
                  isStreaming={loading && i === messages.length - 1 && msg.role === 'assistant'}
                />
              ))}
              <div ref={messagesEndRef} className="h-4" />
            </div>
          </div>

          <ChatInput
            input={input}
            setInput={setInput}
            onSend={handleSend}
            loading={loading}
            onStop={stopGeneration}
          />
        </div>

        {/* Resource Panel */}
        {/* Desktop: Shifts content */}
        <div className={`hidden lg:block transition-all duration-300 ease-in-out border-l border-slate-200 bg-white ${isResourceOpen ? 'w-80' : 'w-0 overflow-hidden'}`}>
          <div className="w-80 h-full"> {/* Inner container to prevent squishing */}
            <ResourcePanel
              isOpen={isResourceOpen}
              onClose={() => setIsResourceOpen(false)}
              citations={allCitations}
              isMobile={false}
            />
          </div>
        </div>

        {/* Mobile/Tablet: Overlay with Backdrop */}
        <div className={clsx("lg:hidden fixed inset-0 z-50 pointer-events-none", isResourceOpen ? "pointer-events-auto" : "")}>
          {/* Backdrop */}
          <div
            className={clsx(
              "absolute inset-0 bg-black/40 backdrop-blur-sm transition-opacity duration-300",
              isResourceOpen ? "opacity-100" : "opacity-0"
            )}
            onClick={() => setIsResourceOpen(false)}
          />
          {/* Panel */}
          <div className={clsx(
            "absolute inset-y-0 right-0 h-full w-80 max-w-[85vw] shadow-2xl transition-transform duration-300 ease-out transform",
            isResourceOpen ? "translate-x-0" : "translate-x-full"
          )}>
            <ResourcePanel
              isOpen={true} // Always render internal content, parent controls visibility via transform
              onClose={() => setIsResourceOpen(false)}
              citations={allCitations}
              isMobile={true}
            />
          </div>
        </div>

      </div>

      {/* Delete Confirmation Modal */}
      {isDeleteModalOpen && (
        <Modal
          isOpen={isDeleteModalOpen}
          onClose={() => setIsDeleteModalOpen(false)}
          title="Delete Conversation"
        >
          <div className="space-y-4">
            <p className="text-slate-600">
              Are you sure you want to delete this conversation? This action cannot be undone.
            </p>
            <div className="flex justify-end gap-3 pt-2">
              <button
                onClick={() => setIsDeleteModalOpen(false)}
                className="px-4 py-2 text-sm font-medium text-slate-600 hover:bg-slate-100 rounded-lg transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={confirmDelete}
                className="px-4 py-2 text-sm font-medium text-white bg-red-600 hover:bg-red-700 rounded-lg transition-colors shadow-sm"
              >
                Delete
              </button>
            </div>
          </div>
        </Modal>
      )}

      {/* Rename Modal */}
      {isRenameModalOpen && (
        <Modal
          isOpen={isRenameModalOpen}
          onClose={() => setIsRenameModalOpen(false)}
          title="Rename Conversation"
        >
          <form onSubmit={(e) => { e.preventDefault(); confirmRename(); }} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">
                Conversation Title
              </label>
              <input
                type="text"
                value={renameTitle}
                onChange={(e) => setRenameTitle(e.target.value)}
                className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
                placeholder="Enter a new title..."
                autoFocus
              />
            </div>
            <div className="flex justify-end gap-3 pt-2">
              <button
                type="button"
                onClick={() => setIsRenameModalOpen(false)}
                className="px-4 py-2 text-sm font-medium text-slate-600 hover:bg-slate-100 rounded-lg transition-colors"
              >
                Cancel
              </button>
              <button
                type="submit"
                className="px-4 py-2 text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors shadow-sm"
              >
                Save
              </button>
            </div>
          </form>
        </Modal>
      )}
    </ChatLayout>
  );
}