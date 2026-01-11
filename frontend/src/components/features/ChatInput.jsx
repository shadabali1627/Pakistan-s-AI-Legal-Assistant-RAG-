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
        <div className="w-full px-3 sm:px-4 pb-4 sm:pb-6 pt-2 bg-gradient-to-t from-white via-white to-transparent z-10 transition-all">
            <div className={clsx(
                "max-w-4xl mx-auto flex items-end gap-2 sm:gap-3 p-2.5 sm:p-3 bg-white border-2 border-slate-300 rounded-2xl focus-within:border-blue-500 focus-within:ring-4 focus-within:ring-blue-500/20 transition-all shadow-lg hover:shadow-xl shadow-slate-300/50",
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
                    className="flex-1 bg-transparent border-none focus:ring-0 outline-none resize-none py-2 sm:py-3 px-2 sm:px-3 text-[16px] sm:text-base text-slate-800 placeholder:text-slate-500 font-normal max-h-[150px] overflow-y-auto"
                    style={{ fontSize: '16px' }}
                />

                <div className="flex items-center gap-2 shrink-0 pb-0.5 sm:pb-1.5 pr-0.5 sm:pr-1.5">
                    {loading ? (
                        <button
                            onClick={onStop}
                            className="p-2.5 sm:p-3 bg-red-50 text-red-500 rounded-xl hover:bg-red-100 transition-colors min-w-[44px] min-h-[44px] flex items-center justify-center"
                            title="Stop Generation"
                        >
                            <StopCircle size={20} className="sm:w-[22px] sm:h-[22px]" />
                        </button>
                    ) : (
                        <button
                            onClick={onSend}
                            disabled={!input.trim()}
                            className="rounded-xl px-3 sm:px-4 py-2.5 sm:py-3 bg-blue-600 hover:bg-blue-700 text-white shadow-lg shadow-blue-600/30 disabled:opacity-50 disabled:cursor-not-allowed disabled:shadow-none transition-all active:scale-95 flex items-center justify-center font-semibold text-sm gap-2 min-w-[44px] min-h-[44px]"
                        >
                            <span className="hidden sm:inline">Send</span>
                            <Send size={18} className="sm:w-[18px] sm:h-[18px]" />
                        </button>
                    )}
                </div>
            </div>
            <div className="text-center mt-3 sm:mt-4 text-xs sm:text-sm text-slate-500 font-medium px-2">
                AI-generated insights are for informational purposes only and do not constitute legal advice.
            </div>
        </div>
    );
}
