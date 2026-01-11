import React from 'react';
import { X, BookOpen, FileText } from 'lucide-react';
import clsx from 'clsx';

export default function ResourcePanel({ isOpen, citations, onClose, isMobile }) {
    if (isMobile && !isOpen) return null;

    return (
        <div className={clsx(
            "flex flex-col h-full bg-white border-l border-slate-200 shadow-sm transition-all overflow-hidden",
            isMobile ? "w-full h-full" : "w-full"
        )}>
            {/* Header */}
            <div className="flex items-center justify-between p-4 border-b border-slate-100 bg-slate-50/50">
                <div className="flex items-center gap-2 text-slate-700 font-semibold">
                    <BookOpen size={18} className="text-blue-600" />
                    <span>Sources</span>
                </div>
                {isMobile && (
                    <button
                        onClick={onClose}
                        className="p-2 text-slate-400 hover:text-slate-600 hover:bg-slate-100 rounded-lg transition-colors"
                    >
                        <X size={20} />
                    </button>
                )}
            </div>

            {/* Content */}
            <div className="flex-1 overflow-y-auto p-4 space-y-3 custom-scrollbar">
                {citations && citations.length > 0 ? (
                    citations.map((cite, idx) => (
                        <div
                            key={idx}
                            className="group p-3 bg-white border border-slate-200 rounded-xl hover:border-blue-300 hover:shadow-md transition-all cursor-pointer"
                            title={cite.text || "View Source"}
                        >
                            <div className="flex items-start gap-3">
                                <div className="mt-1 min-w-[24px] w-8 h-8 flex items-center justify-center bg-blue-50 text-blue-600 rounded-lg group-hover:bg-blue-600 group-hover:text-white transition-colors">
                                    <FileText size={16} />
                                </div>
                                <div>
                                    <h4 className="text-sm font-semibold text-slate-800 leading-snug mb-1 group-hover:text-blue-700">
                                        {cite.source ? cite.source.split(/[\\\/]/).pop().replace('.pdf', '') : "Legal Document"}
                                    </h4>
                                    {cite.page && (
                                        <p className="text-xs text-slate-500 font-medium">
                                            Page {cite.page}
                                        </p>
                                    )}
                                    {cite.score && (
                                        <div className="mt-2 text-[10px] font-semibold uppercase tracking-wider text-emerald-600 bg-emerald-50 inline-block px-1.5 py-0.5 rounded border border-emerald-100">
                                            {Math.round(cite.score * 100)}% Match
                                        </div>
                                    )}
                                </div>
                            </div>
                        </div>
                    ))
                ) : (
                    <div className="flex flex-col items-center justify-center h-48 text-center px-4">
                        <div className="w-12 h-12 bg-slate-50 rounded-full flex items-center justify-center mb-3">
                            <SearchIcon className="text-slate-300" />
                        </div>
                        <p className="text-slate-400 text-sm italic">
                            Relevant case law, statutes, and citations will appear here after your query.
                        </p>
                    </div>
                )}
            </div>
        </div>
    );
}

function SearchIcon({ className }) {
    return (
        <svg
            className={`w-6 h-6 ${className}`}
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={2}
        >
            <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
        </svg>
    );
}
