import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import clsx from 'clsx';
import { User, Bot, Copy, Check, Volume2, Square } from 'lucide-react';

// Code Block Component with Copy
const CodeBlock = ({ inline, className, children, ...props }) => {
    const [copied, setCopied] = useState(false);
    const match = /language-(\w+)/.exec(className || '');
    const lang = match ? match[1] : '';

    if (inline) {
        return <code className="bg-slate-100 px-1 py-0.5 rounded text-sm text-pink-600 font-mono" {...props}>{children}</code>;
    }

    const handleCopy = () => {
        navigator.clipboard.writeText(String(children).replace(/\n$/, ''));
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    return (
        <div className="relative group my-4 rounded-lg overflow-hidden border border-slate-200 bg-slate-50">
            <div className="flex items-center justify-between px-4 py-2 bg-slate-100 border-b border-slate-200 text-xs text-slate-500 font-mono">
                <span>{lang}</span>
                <button onClick={handleCopy} className="flex items-center gap-1 hover:text-slate-700">
                    {copied ? <Check size={14} className="text-green-600" /> : <Copy size={14} />}
                    {copied ? "Copied" : "Copy"}
                </button>
            </div>
            <div className="p-4 overflow-x-auto">
                <code className={className} {...props}>
                    {children}
                </code>
            </div>
        </div>
    );
};

// Hook for smooth typewriter effect
const useTypewriter = (text, baseSpeed = 15, isStreaming = false) => {
    const [currentIndex, setCurrentIndex] = useState(0);
    const rafIdRef = useRef(null);
    const lastUpdateRef = useRef(Date.now());

    useEffect(() => {
        // If it's a fresh load (history) and not actively streaming, show immediately.
        if (!isStreaming) {
            if (currentIndex !== text.length) {
                setCurrentIndex(text.length);
            }
            return;
        }

        // If we have text to type
        if (currentIndex < text.length) {
            // Adaptive speed based on how fast text is coming in
            const timeSinceLastUpdate = Date.now() - lastUpdateRef.current;
            const remainingChars = text.length - currentIndex;

            // If large chunk arrived recently, speed up typewriter
            const speed = remainingChars > 50 ? Math.max(baseSpeed / 2, 8) : baseSpeed;

            rafIdRef.current = requestAnimationFrame(() => {
                const timeoutId = setTimeout(() => {
                    setCurrentIndex((prev) => Math.min(prev + 1, text.length));
                    lastUpdateRef.current = Date.now();
                }, speed);

                return () => clearTimeout(timeoutId);
            });

            return () => {
                if (rafIdRef.current) {
                    cancelAnimationFrame(rafIdRef.current);
                }
            };
        }
    }, [text, baseSpeed, isStreaming, currentIndex]);

    // Reset if text abruptly changes to something shorter
    useEffect(() => {
        if (text.length < currentIndex) {
            setCurrentIndex(text.length);
        }
    }, [text, currentIndex]);

    return text.slice(0, currentIndex);
};

export default function MessageBubble({ role, content, status, citations, onCitationClick, userPhoto, isStreaming }) {
    const isUser = role === 'user';
    const isSystem = role === 'system';
    const [isSpeaking, setIsSpeaking] = useState(false);

    // Use typewriter only for loading historical messages, NOT for live streaming
    // This gives instant, professional streaming response
    const safeContent = content || "";
    const displayedContent = useTypewriter(safeContent, 15, false); // Always false for instant display during streaming

    // Show typing indicator only when we have a status message and no content yet
    const isTyping = isStreaming && (!isUser) && safeContent.length === 0 && status;

    // Stop speaking when component unmounts
    useEffect(() => {
        return () => {
            if (isSpeaking) {
                window.speechSynthesis.cancel();
            }
        };
    }, [isSpeaking]);

    if (isSystem) return null;

    const handleSpeak = () => {
        if (!window.speechSynthesis) return;

        if (isSpeaking) {
            window.speechSynthesis.cancel();
            setIsSpeaking(false);
        } else {
            // Cancel any current speech before starting new
            window.speechSynthesis.cancel();

            const utterance = new SpeechSynthesisUtterance(content); // Speak full content, not partial
            utterance.onend = () => setIsSpeaking(false);
            utterance.onerror = () => setIsSpeaking(false);

            window.speechSynthesis.speak(utterance);
            setIsSpeaking(true);
        }
    };

    return (
        <div className={clsx("flex w-full mb-6", isUser ? "justify-end" : "justify-start")}>
            <div className={clsx("flex max-w-[98%] md:max-w-[90%] lg:max-w-[85%]", isUser ? "flex-row-reverse" : "flex-row")}>

                {/* Avatar */}
                <div className={clsx("flex-shrink-0 w-8 h-8 md:w-10 md:h-10 rounded-full flex items-center justify-center mt-1 overflow-hidden shadow-sm",
                    isUser ? "ml-3 bg-slate-200" : "mr-3 bg-gradient-to-br from-blue-600 to-indigo-700 text-white ring-2 ring-white")}>
                    {isUser ? (
                        userPhoto ? (
                            <img src={userPhoto} alt="User" className="w-full h-full object-cover" />
                        ) : (
                            <User size={20} className="text-slate-500" />
                        )
                    ) : (
                        <Bot size={20} />
                    )}
                </div>

                {/* Bubble */}
                <div className={clsx("flex flex-col relative min-w-0")}>
                    <div className={clsx(
                        "px-5 py-4 text-[15px] md:text-base leading-relaxed shadow-sm",
                        isUser
                            ? "bg-gradient-to-br from-slate-900 to-slate-800 text-white rounded-2xl rounded-tr-sm shadow-md" // User: Premium Dark Gradient
                            : "bg-white border border-slate-100 text-slate-800 rounded-2xl rounded-tl-sm shadow-sm" // AI: Clean White
                    )}>

                        {/* Status Indicator (Thinking...) */}
                        {!isUser && status && displayedContent.length === 0 && (
                            <div className="flex items-center gap-2 text-sm text-blue-600 font-medium animate-pulse mb-2">
                                <span className="w-2 h-2 rounded-full bg-blue-500"></span>
                                {status}
                            </div>
                        )}

                        {/* Content */}
                        <div className={clsx("markdown-body", isUser ? "text-slate-50" : "text-slate-800")}>
                            <ReactMarkdown
                                remarkPlugins={[remarkGfm]}
                                components={{
                                    code: CodeBlock,
                                    a: ({ node, ...props }) => <a target="_blank" rel="noopener noreferrer" className="text-blue-500 underline decoration-blue-300 underline-offset-2 hover:text-blue-700" {...props} />,
                                    ul: ({ node, ...props }) => <ul className="list-disc pl-5 my-2 space-y-1" {...props} />,
                                    ol: ({ node, ...props }) => <ol className="list-decimal pl-5 my-2 space-y-1" {...props} />,
                                    p: ({ node, ...props }) => <p className="mb-2 last:mb-0" {...props} />,
                                    h1: ({ node, ...props }) => <h1 className="text-2xl font-bold mb-4 mt-6 border-b pb-2" {...props} />,
                                    h2: ({ node, ...props }) => <h2 className="text-xl font-bold mb-3 mt-5" {...props} />,
                                    h3: ({ node, ...props }) => <h3 className="text-lg font-semibold mb-2 mt-4" {...props} />,
                                }}
                            >
                                {displayedContent}
                            </ReactMarkdown>

                            {/* Cursor - show during active streaming */}
                            {isStreaming && !isUser && safeContent.length > 0 && (
                                <span className="inline-block w-2 h-5 ml-1 align-sub bg-blue-500 rounded-sm animate-pulse" />
                            )}
                        </div>

                        {/* Citations */}
                        {!isUser && citations && citations.length > 0 && (
                            <div className="mt-4 pt-3 border-t border-slate-100 flex flex-wrap gap-2">
                                {citations.map((cite, i) => (
                                    <button
                                        key={i}
                                        onClick={() => onCitationClick && onCitationClick(cite)}
                                        className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-blue-50 hover:bg-blue-100 border border-blue-100 text-xs font-semibold text-blue-700 rounded-full transition-colors"
                                    >
                                        <span className="w-5 h-5 rounded-full bg-blue-200 flex items-center justify-center text-[10px] text-blue-800">{i + 1}</span>
                                        <span className="truncate max-w-[120px]">{cite.source?.split(/[\\/]/).pop()}</span>
                                    </button>
                                ))}
                            </div>
                        )}

                        {/* Actions Footer - show only when streaming is complete */}
                        {!isUser && !isStreaming && (
                            <div className="mt-3 flex items-center justify-end gap-2 pt-2 border-t border-slate-50 border-opacity-50">
                                <button
                                    onClick={handleSpeak}
                                    className={clsx(
                                        "p-1.5 rounded-lg transition-colors",
                                        isSpeaking
                                            ? "text-blue-600 bg-blue-50 animate-pulse"
                                            : "text-slate-400 hover:text-blue-600 hover:bg-blue-50"
                                    )}
                                    title={isSpeaking ? "Stop Speaking" : "Read Aloud"}
                                >
                                    {isSpeaking ? <Square size={16} fill="currentColor" /> : <Volume2 size={16} />}
                                </button>
                                <button
                                    onClick={() => {
                                        navigator.clipboard.writeText(content);
                                    }}
                                    className="p-1.5 text-slate-400 hover:text-blue-600 hover:bg-blue-50 rounded-lg transition-colors group"
                                    title="Copy Text"
                                >
                                    <Copy size={16} className="group-active:hidden" />
                                    <Check size={16} className="hidden group-active:block text-green-600" />
                                </button>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}
