import React, { useRef, useEffect } from 'react';
import { Send, StopCircle } from 'lucide-react';
import clsx from 'clsx';

export default function ChatInput({ input, setInput, onSend, loading, onStop }) {
    const textareaRef = useRef(null);

    // Auto-resize
    useEffect(() => {
        if (textareaRef.current) {
            textareaRef.current.style.height = 'auto';
            textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 150)}px`;
        }
    }, [input]);

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (input.trim() && !loading) onSend();
        }
    };

    return (
        <div className="w-full px-4 pb-6 pt-2 bg-gradient-to-t from-white via-white to-transparent z-10 transition-all">
            <div className={clsx(
                "max-w-4xl mx-auto flex items-end gap-3 p-3 bg-white border border-slate-200/80 rounded-2xl focus-within:border-blue-400 focus-within:ring-4 focus-within:ring-blue-500/10 transition-all shadow-2xl shadow-slate-200/60",
                loading && "opacity-80 grayscale"
            )}>

                <textarea
                    ref={textareaRef}
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder="Ask a legal question..."
                    rows={1}
                    disabled={loading}
                    className="flex-1 bg-transparent border-none focus:ring-0 outline-none resize-none py-3 px-3 text-base text-slate-800 placeholder:text-slate-400 max-h-[150px] overflow-y-auto"
                />

                <div className="flex items-center gap-2 shrink-0 pb-1.5 pr-1.5">
                    {loading ? (
                        <button
                            onClick={onStop}
                            className="p-3 bg-red-50 text-red-500 rounded-xl hover:bg-red-100 transition-colors"
                            title="Stop Generation"
                        >
                            <StopCircle size={22} />
                        </button>
                    ) : (
                        <button
                            onClick={onSend}
                            disabled={!input.trim()}
                            className="rounded-xl px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white shadow-md shadow-blue-600/20 disabled:opacity-50 disabled:cursor-not-allowed disabled:shadow-none transition-all active:scale-95 flex items-center justify-center font-medium text-sm gap-2"
                        >
                            <span className="hidden sm:inline">Send</span>
                            <Send size={18} />
                        </button>
                    )}
                </div>
            </div>
            <div className="text-center mt-3 text-[11px] text-slate-400 font-medium">
                AI-generated insights are for informational purposes only and do not constitute legal advice.
            </div>
        </div>
    );
}
