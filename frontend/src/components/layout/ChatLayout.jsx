import React from 'react';
import { Plus, User, LogOut, Settings, Menu, FileText, X, Pencil, Trash2, BookOpen } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from "../../hooks/useAuth";

export default function ChatLayout({
    children,
    isSidebarOpen,
    setIsSidebarOpen,
    isResourceOpen,
    setIsResourceOpen,
    onNewChat,
    chatHistory,
    activeChatId,
    onSelectChat,
    onDeleteChat,
    onRenameChat,
    user
}) {
    const { signout } = useAuth();
    const nav = useNavigate();
    const [isLogoutModalOpen, setIsLogoutModalOpen] = React.useState(false);

    return (
        <div className="flex h-screen bg-white overflow-hidden">
            {/* --- Sidebar Overlay for Mobile --- */}
            {isSidebarOpen && (
                <div
                    className="fixed inset-0 z-40 bg-gray-900/50 backdrop-blur-sm lg:hidden"
                    onClick={() => setIsSidebarOpen(false)}
                />
            )}

            {/* --- Sidebar --- */}
            <aside
                className={`
                    bg-slate-900 text-white flex flex-col transition-all duration-300 ease-in-out border-r border-slate-800 w-72
                    ${isSidebarOpen
                        ? 'relative' // On desktop when open: take up space
                        : 'fixed inset-y-0 left-0 -translate-x-full lg:w-0 lg:border-0' // When closed: slide out
                    }
                    lg:relative lg:transition-[width] lg:duration-300
                    ${!isSidebarOpen ? 'lg:w-0 lg:overflow-hidden' : ''}
                    max-lg:fixed max-lg:inset-y-0 max-lg:left-0 max-lg:z-50
                    ${isSidebarOpen ? 'max-lg:translate-x-0' : 'max-lg:-translate-x-full'}
                `}
            >
                {/* Header */}
                <div className="flex items-center h-16 px-6 border-b border-slate-800 bg-slate-900/50 backdrop-blur-sm">
                    <div className="flex items-center gap-3">
                        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-blue-600 font-bold text-white shadow-lg shadow-blue-900/20">
                            AI
                        </div>
                        <div>
                            <div className="font-semibold leading-none tracking-tight text-slate-100">Legal Assistant</div>
                            <div className="mt-0.5 text-[10px] font-medium text-blue-400">PROFESSIONAL</div>
                        </div>
                    </div>
                    {/* Mobile Close Button */}
                    <button
                        onClick={() => setIsSidebarOpen(false)}
                        className="ml-auto lg:hidden text-slate-400 hover:text-white"
                    >
                        <X size={20} />
                    </button>
                </div>

                {/* New Chat Button */}
                <div className="p-4">
                    <button
                        onClick={() => {
                            onNewChat();
                            if (window.innerWidth < 1024) setIsSidebarOpen(false);
                        }}
                        className="group flex w-full items-center justify-center gap-2 rounded-xl bg-blue-600 px-4 py-3 text-sm font-semibold text-white shadow-lg shadow-blue-900/20 hover:bg-blue-500 hover:shadow-blue-600/30 transition-all active:scale-[0.98]"
                    >
                        <Plus size={18} className="transition-transform group-hover:rotate-90" />
                        <span>New Conversation</span>
                    </button>
                </div>

                {/* History List */}
                <div className="flex-1 overflow-y-auto px-3 py-2 custom-scrollbar space-y-1">
                    <div className="px-3 pb-2 text-xs font-medium uppercase tracking-wider text-slate-500">
                        History
                    </div>
                    {chatHistory.length > 0 ? (
                        chatHistory.map(chat => (
                            <div
                                key={chat.id}
                                className={`
                                    group flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-left text-sm transition-all cursor-pointer relative
                                    ${chat.id === activeChatId
                                        ? 'bg-slate-800 text-white shadow-sm'
                                        : 'text-slate-400 hover:bg-slate-800/50 hover:text-slate-200'
                                    }
                                `}
                                onClick={() => {
                                    onSelectChat(chat.id);
                                    if (window.innerWidth < 1024) setIsSidebarOpen(false);
                                }}
                            >
                                <div className={`mt-0.5 shrink-0 transition-colors ${chat.id === activeChatId ? 'text-blue-400' : 'text-slate-600 group-hover:text-slate-500'}`}>
                                    <FileText size={16} />
                                </div>
                                <span className="line-clamp-1 flex-1 break-all pr-8">
                                    {chat.title || "Untitled Conversation"}
                                </span>

                                {/* Edit/Delete Actions - Visible on Hover */}
                                <div className="absolute right-2 top-1/2 -translate-y-1/2 flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity bg-slate-800/90 rounded-md p-0.5 shadow-sm">
                                    <button
                                        onClick={(e) => onRenameChat(e, chat.id, chat.title)}
                                        className="p-1 text-slate-400 hover:text-blue-400 hover:bg-slate-700/50 rounded"
                                        title="Rename"
                                    >
                                        <Pencil size={12} />
                                    </button>
                                    <button
                                        onClick={(e) => onDeleteChat(e, chat.id)}
                                        className="p-1 text-slate-400 hover:text-red-400 hover:bg-slate-700/50 rounded"
                                        title="Delete"
                                    >
                                        <Trash2 size={12} />
                                    </button>
                                </div>
                            </div>
                        ))
                    ) : (
                        <div className="px-3 py-8 text-center text-sm text-slate-600 italic">
                            No active conversations
                        </div>
                    )}
                </div>

                {/* User Profile */}
                <div className="border-t border-slate-800 p-4 bg-slate-950/30">
                    <button
                        onClick={() => setIsLogoutModalOpen(true)}
                        className="group flex w-full items-center gap-3 rounded-xl p-2 transition-all duration-200 hover:bg-slate-800"
                    >
                        <div className="flex h-9 w-9 items-center justify-center rounded-full bg-slate-700 text-slate-300 ring-2 ring-slate-800 overflow-hidden shrink-0 group-hover:ring-slate-600 transition-all">
                            {user?.picture || user?.photoURL || user?.avatar_url ? (
                                <img
                                    src={user.picture || user.photoURL || user.avatar_url}
                                    alt={user.name || "User"}
                                    className="w-full h-full object-cover"
                                    referrerPolicy="no-referrer"
                                />
                            ) : (
                                <User size={18} />
                            )}
                        </div>
                        <div className="flex-1 overflow-hidden text-left">
                            <div className="truncate text-sm font-medium text-white group-hover:text-blue-100 transition-colors">
                                {user?.name || "Legal Professional"}
                            </div>
                            <div className="truncate text-xs text-slate-400">
                                {user?.email}
                            </div>
                        </div>
                        <LogOut size={16} className="text-slate-500 group-hover:text-red-400 transition-colors" />
                    </button>
                </div>
            </aside>

            {/* --- Left Vertical Strip (when sidebar is closed on desktop) --- */}
            {!isSidebarOpen && (
                <aside className="hidden lg:flex flex-col items-center justify-between w-16 bg-slate-900 border-r border-slate-800 py-6">
                    {/* New Chat Icon */}
                    <button
                        onClick={onNewChat}
                        className="flex items-center justify-center h-10 w-10 rounded-xl bg-blue-600 text-white hover:bg-blue-500 transition-all shadow-lg shadow-blue-900/20 hover:shadow-blue-600/30 active:scale-95"
                        title="New Conversation"
                    >
                        <Plus size={20} />
                    </button>

                    {/* User Profile Picture */}
                    <button
                        onClick={() => setIsLogoutModalOpen(true)}
                        className="flex h-10 w-10 items-center justify-center rounded-full bg-slate-700 text-slate-300 ring-2 ring-slate-800 overflow-hidden hover:ring-slate-600 transition-all"
                        title={user?.name || "User Profile"}
                    >
                        {user?.picture || user?.photoURL || user?.avatar_url ? (
                            <img
                                src={user.picture || user.photoURL || user.avatar_url}
                                alt={user.name || "User"}
                                className="w-full h-full object-cover"
                                referrerPolicy="no-referrer"
                            />
                        ) : (
                            <User size={18} />
                        )}
                    </button>
                </aside>
            )}

            {/* --- Main Content Area --- */}
            <main className="flex flex-1 flex-col h-full min-w-0 bg-white relative">
                {/* Desktop Header */}
                <header className="hidden lg:flex h-16 items-center justify-between border-b border-slate-100 bg-white px-6">
                    <div className="flex items-center gap-3">
                        <button
                            onClick={() => setIsSidebarOpen(!isSidebarOpen)}
                            className="p-2 text-slate-500 hover:bg-slate-100 rounded-lg transition-colors"
                            title={isSidebarOpen ? "Hide Sidebar" : "Show Sidebar"}
                        >
                            <Menu size={20} />
                        </button>
                        <div className="h-8 w-1 bg-blue-600 rounded-full"></div>
                        <h2 className="font-semibold text-slate-800 text-lg">
                            {activeChatId
                                ? chatHistory.find(c => c.id === activeChatId)?.title || "Conversation"
                                : "New Conversation"
                            }
                        </h2>
                    </div>
                    <div className="flex items-center gap-2">
                        <button
                            onClick={() => setIsResourceOpen(!isResourceOpen)}
                            className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${isResourceOpen
                                ? 'bg-blue-50 text-blue-600 ring-1 ring-blue-200'
                                : 'text-slate-500 hover:bg-slate-50 hover:text-slate-700'
                                }`}
                        >
                            <BookOpen size={18} />
                            <span>{isResourceOpen ? 'Hide Sources' : 'Show Sources'}</span>
                        </button>
                    </div>
                </header>

                {/* Mobile Topbar */}
                <header className="flex h-16 items-center justify-between border-b border-slate-100 bg-white px-4 lg:hidden z-10">
                    <button
                        onClick={() => setIsSidebarOpen(true)}
                        className="p-2 text-slate-500 hover:bg-slate-100 rounded-lg transition-colors"
                    >
                        <Menu size={24} />
                    </button>
                    <div className="flex items-center gap-2">
                        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-blue-600 font-bold text-white shadow-lg shadow-blue-500/20">
                            AI
                        </div>
                        <span className="font-semibold text-slate-900">AI Legal Assistant</span>
                    </div>
                    <button
                        onClick={() => setIsResourceOpen(!isResourceOpen)}
                        className={`p-2 rounded-lg transition-colors ${isResourceOpen ? 'bg-blue-50 text-blue-600' : 'text-slate-500 hover:bg-slate-100 transition-colors'}`}
                    >
                        <FileText size={24} />
                    </button>
                </header>

                {/* Children Wrapper */}
                <div className="flex flex-1 overflow-hidden relative">
                    {children}
                </div>
            </main>

            {/* --- Logout Modal --- */}
            {isLogoutModalOpen && (
                <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/60 backdrop-blur-sm px-4 animate-in fade-in duration-200">
                    <div className="w-full max-w-sm overflow-hidden rounded-2xl bg-white shadow-2xl animate-in zoom-in-95 duration-200 scale-100">
                        <div className="p-6 text-center">
                            <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-red-100 mb-6">
                                <LogOut className="h-8 w-8 text-red-600" />
                            </div>
                            <h3 className="mb-2 text-xl font-bold text-slate-900">Sign Out</h3>
                            <p className="text-slate-500 mb-8">
                                Are you sure you want to sign out? You will be redirected to the login page.
                            </p>
                            <div className="flex gap-3">
                                <button
                                    onClick={() => setIsLogoutModalOpen(false)}
                                    className="flex-1 rounded-xl border border-slate-200 bg-white py-3 text-sm font-semibold text-slate-700 hover:bg-slate-50 transition-colors"
                                >
                                    Cancel
                                </button>
                                <button
                                    onClick={() => {
                                        setIsLogoutModalOpen(false);
                                        signout();
                                        nav('/login');
                                    }}
                                    className="flex-1 rounded-xl bg-red-600 py-3 text-sm font-semibold text-white shadow-lg shadow-red-200 hover:bg-red-700 transition-colors"
                                >
                                    Log Out
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
