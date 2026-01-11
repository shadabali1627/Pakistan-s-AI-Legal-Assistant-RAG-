import { useState, useRef, useCallback } from 'react';
import { streamAnswer } from '../api';

export function useChatStream() {
    const [messages, setMessages] = useState([]);
    const [loading, setLoading] = useState(false);
    const abortControllerRef = useRef(null);

    const stopGeneration = useCallback(() => {
        if (abortControllerRef.current) {
            abortControllerRef.current.abort();
            abortControllerRef.current = null;
        }
        setLoading(false);
    }, []);

    const sendMessage = useCallback(async (text, history = [], userEmail = null, chatId = null) => {
        if (!text.trim()) return;

        const userMsg = { role: "user", content: text, id: crypto.randomUUID() };
        const botMsg = {
            role: "assistant",
            content: "",
            citations: [],
            status: "Initializing...",
            id: crypto.randomUUID()
        };

        setMessages(prev => [...prev, userMsg, botMsg]);
        setLoading(true);

        abortControllerRef.current = new AbortController();

        try {
            // Prepare history for API
            // content is now natively in the message object, no mapping needed if it matches API expectation
            const apiHistory = history.map(m => ({ role: m.role, content: m.content }));

            let acc = "";
            let currentCitations = [];
            let currentStatus = "Analyzing...";

            for await (const event of streamAnswer(text, apiHistory, abortControllerRef.current.signal, userEmail, chatId)) {
                if (event.type === 'content') {
                    acc += event.data;
                    currentStatus = ""; // Clear status when content flows
                } else if (event.type === 'citations') {
                    currentCitations = event.data;
                } else if (event.type === 'status') {
                    currentStatus = event.data;
                } else if (event.type === 'error') {
                    acc += `\n\n**Error:** ${event.data}`;
                }

                // Update state immediately - React will batch automatically
                setMessages(prev => {
                    const newMsgs = [...prev];
                    const lastIdx = newMsgs.length - 1;
                    newMsgs[lastIdx] = {
                        ...newMsgs[lastIdx],
                        content: acc,
                        citations: currentCitations,
                        status: currentStatus
                    };
                    return newMsgs;
                });
            }
        } catch (e) {
            if (e.name !== 'AbortError') {
                console.error("Stream error", e);
                setMessages(prev => {
                    const newMsgs = [...prev];
                    const lastIdx = newMsgs.length - 1;
                    newMsgs[lastIdx] = { ...newMsgs[lastIdx], content: newMsgs[lastIdx].content + "\n\n[Connection Error]" };
                    return newMsgs;
                });
            }
        } finally {
            setLoading(false);
            abortControllerRef.current = null;
        }
    }, []);

    const resetChat = useCallback(() => {
        setMessages([]);
        setLoading(false);
    }, []);

    return {
        messages,
        loading,
        sendMessage,
        stopGeneration,
        resetChat,
        setMessages
    };
}
