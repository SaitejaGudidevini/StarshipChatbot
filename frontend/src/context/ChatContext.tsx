import { createContext, useContext, useState, useCallback, ReactNode } from 'react';
import { ChatMessage, ChatResponse } from '../types';
import { apiClient } from '../api/client';

interface ChatContextType {
  // State
  messages: ChatMessage[];
  loading: boolean;

  // Actions
  sendMessage: (text: string) => Promise<void>;
  clearChat: () => void;
}

const ChatContext = createContext<ChatContextType | null>(null);

const INITIAL_MESSAGE: ChatMessage = {
  id: '1',
  type: 'bot',
  text: 'Hello! Ask me anything about the Q&A database.',
};

export function ChatProvider({ children }: { children: ReactNode }) {
  const [messages, setMessages] = useState<ChatMessage[]>([INITIAL_MESSAGE]);
  const [loading, setLoading] = useState(false);

  const sendMessage = useCallback(async (text: string) => {
    if (!text.trim() || loading) return;

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      type: 'user',
      text: text,
    };
    setMessages((prev) => [...prev, userMessage]);
    setLoading(true);

    try {
      const response = await apiClient.post<ChatResponse>('/api/chat', {
        question: text,
      });

      const botMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        type: 'bot',
        text: response.answer,
        confidence: response.confidence,
        topic: response.source_topic,
        matched_by: response.matched_by,
        pipeline_info: response.pipeline_info,
      };
      setMessages((prev) => [...prev, botMessage]);
    } catch (err) {
      const errorMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        type: 'bot',
        text: 'Sorry, an error occurred. Please try again.',
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  }, [loading]);

  const clearChat = useCallback(() => {
    setMessages([INITIAL_MESSAGE]);
  }, []);

  return (
    <ChatContext.Provider
      value={{
        messages,
        loading,
        sendMessage,
        clearChat,
      }}
    >
      {children}
    </ChatContext.Provider>
  );
}

export function useChat() {
  const context = useContext(ChatContext);
  if (!context) {
    throw new Error('useChat must be used within a ChatProvider');
  }
  return context;
}
