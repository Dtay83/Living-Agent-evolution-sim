/**
 * ChatPanel Component
 * 
 * Two-way communication panel with individual agent chat histories.
 * Features:
 * - Agent sidebar showing all self-aware agents
 * - Per-agent conversation view
 * - Broadcast mode to message all agents
 * - Visual indicators for new messages
 */

import React, { useState, useEffect, useRef, useMemo } from 'react';
import { AgentMessage, CommunicationLog, MessageType, getMessageTypeEmoji, getMessageTypeLabel } from '../communication-system';

interface ChatPanelProps {
  communicationLog: CommunicationLog;
  isOpen: boolean;
  onToggle: () => void;
  onClear: () => void;
  onSendMessage: (content: string, targetAgentId?: number) => void;
  selfAwareAgentIds: number[];
  currentTick: number;
}

export const ChatPanel: React.FC<ChatPanelProps> = ({
  communicationLog,
  isOpen,
  onToggle,
  onClear,
  onSendMessage,
  selfAwareAgentIds,
  currentTick,
}) => {
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const [unreadCount, setUnreadCount] = useState(0);
  const [lastSeenCount, setLastSeenCount] = useState(0);
  const [inputValue, setInputValue] = useState('');
  const [selectedAgentId, setSelectedAgentId] = useState<number | 'all'>('all');

  // Get messages filtered by selected agent
  const filteredMessages = useMemo(() => {
    if (selectedAgentId === 'all') {
      return communicationLog.messages.slice(-50);
    }
    return communicationLog.messages.filter(msg => 
      msg.agentId === selectedAgentId || 
      msg.targetAgentId === selectedAgentId ||
      (msg.isFromUser && msg.targetAgentId === selectedAgentId)
    ).slice(-50);
  }, [communicationLog.messages, selectedAgentId]);

  // Count messages per agent for sidebar badges
  const messageCountByAgent = useMemo(() => {
    const counts = new Map<number, number>();
    communicationLog.messages.forEach(msg => {
      if (msg.agentId > 0 && !msg.isFromUser) {
        counts.set(msg.agentId, (counts.get(msg.agentId) || 0) + 1);
      }
    });
    return counts;
  }, [communicationLog.messages]);

  // Track unread messages when panel is closed
  useEffect(() => {
    if (!isOpen) {
      const newMessages = communicationLog.totalMessages - lastSeenCount;
      if (newMessages > 0) {
        setUnreadCount(newMessages);
      }
    } else {
      setUnreadCount(0);
      setLastSeenCount(communicationLog.totalMessages);
    }
  }, [communicationLog.totalMessages, isOpen, lastSeenCount]);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (isOpen && messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [filteredMessages.length, isOpen, selectedAgentId]);

  // Focus input when agent is selected
  useEffect(() => {
    if (isOpen && inputRef.current) {
      inputRef.current.focus();
    }
  }, [selectedAgentId, isOpen]);

  const hasActiveAgents = communicationLog.activeAgents.size > 0;
  const hasSelfAwareAgents = selfAwareAgentIds.length > 0;

  const handleSend = () => {
    if (inputValue.trim() && hasSelfAwareAgents) {
      const targetId = selectedAgentId === 'all' ? undefined : selectedAgentId;
      onSendMessage(inputValue.trim(), targetId);
      setInputValue('');
    }
  };

  return (
    <>
      {/* Chat Icon Button */}
      <button
        onClick={onToggle}
        style={{
          position: 'fixed',
          bottom: 20,
          right: 20,
          width: 56,
          height: 56,
          borderRadius: '50%',
          background: hasActiveAgents 
            ? 'linear-gradient(135deg, #ffd700, #ff8c00)' 
            : '#2a3a5a',
          border: 'none',
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          boxShadow: hasActiveAgents 
            ? '0 0 20px rgba(255, 215, 0, 0.6)' 
            : '0 4px 12px rgba(0, 0, 0, 0.3)',
          transition: 'all 0.3s ease',
          zIndex: 1000,
          animation: hasActiveAgents ? 'pulse 2s infinite' : 'none',
        }}
        title={hasActiveAgents ? 'Agents are communicating!' : 'Agent Communications'}
      >
        <span style={{ fontSize: 24 }}>
          {hasActiveAgents ? '🗣️' : '💬'}
        </span>
        
        {/* Unread badge */}
        {unreadCount > 0 && !isOpen && (
          <span
            style={{
              position: 'absolute',
              top: -5,
              right: -5,
              background: '#ff4444',
              color: 'white',
              borderRadius: '50%',
              width: 22,
              height: 22,
              fontSize: 12,
              fontWeight: 'bold',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            {unreadCount > 99 ? '99+' : unreadCount}
          </span>
        )}

        {/* Active indicator dot */}
        {hasActiveAgents && (
          <span
            style={{
              position: 'absolute',
              bottom: 2,
              right: 2,
              width: 12,
              height: 12,
              background: '#00ff00',
              borderRadius: '50%',
              border: '2px solid #1a2a4a',
              animation: 'blink 1s infinite',
            }}
          />
        )}
      </button>

      {/* Chat Panel - Expanded with Agent Sidebar */}
      {isOpen && (
        <div
          style={{
            position: 'fixed',
            bottom: 90,
            right: 20,
            width: 520,
            height: 520,
            background: '#1a2a4a',
            borderRadius: 12,
            boxShadow: '0 8px 32px rgba(0, 0, 0, 0.4)',
            display: 'flex',
            flexDirection: 'column',
            zIndex: 999,
            border: '1px solid #3a4a6a',
          }}
        >
          {/* Header */}
          <div
            style={{
              padding: '12px 16px',
              borderBottom: '1px solid #3a4a6a',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              background: '#2a3a5a',
              borderRadius: '12px 12px 0 0',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ fontSize: 18 }}>🧠</span>
              <span style={{ fontWeight: 'bold', color: '#fff' }}>
                {selectedAgentId === 'all' ? 'All Agents' : `Agent #${selectedAgentId}`}
              </span>
              {hasActiveAgents && (
                <span
                  style={{
                    background: '#00ff00',
                    color: '#000',
                    fontSize: 10,
                    padding: '2px 6px',
                    borderRadius: 10,
                    fontWeight: 'bold',
                  }}
                >
                  {communicationLog.activeAgents.size} ACTIVE
                </span>
              )}
            </div>
            <div style={{ display: 'flex', gap: 8 }}>
              <button
                onClick={onClear}
                style={{
                  background: 'transparent',
                  border: 'none',
                  color: '#888',
                  cursor: 'pointer',
                  fontSize: 12,
                }}
                title="Clear messages"
              >
                🗑️
              </button>
              <button
                onClick={onToggle}
                style={{
                  background: 'transparent',
                  border: 'none',
                  color: '#fff',
                  cursor: 'pointer',
                  fontSize: 18,
                }}
              >
                ✕
              </button>
            </div>
          </div>

          {/* Main Content: Sidebar + Messages */}
          <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
            
            {/* Agent Sidebar */}
            <div
              style={{
                width: 120,
                borderRight: '1px solid #3a4a6a',
                background: '#151a30',
                overflowY: 'auto',
                padding: 8,
              }}
            >
              {/* All Agents option */}
              <AgentTab
                label="🌍 All"
                isSelected={selectedAgentId === 'all'}
                onClick={() => setSelectedAgentId('all')}
                messageCount={communicationLog.messages.length}
                isActive={hasActiveAgents}
              />
              
              <div style={{ 
                fontSize: 9, 
                color: '#666', 
                padding: '8px 4px 4px', 
                borderBottom: '1px solid #2a3a4a',
                marginBottom: 4 
              }}>
                SELF-AWARE ({selfAwareAgentIds.length})
              </div>
              
              {selfAwareAgentIds.length === 0 ? (
                <div style={{ 
                  fontSize: 10, 
                  color: '#555', 
                  padding: 8, 
                  textAlign: 'center',
                  fontStyle: 'italic' 
                }}>
                  No self-aware agents yet
                </div>
              ) : (
                selfAwareAgentIds.map(agentId => (
                  <AgentTab
                    key={agentId}
                    label={`🤖 #${agentId}`}
                    isSelected={selectedAgentId === agentId}
                    onClick={() => setSelectedAgentId(agentId)}
                    messageCount={messageCountByAgent.get(agentId) || 0}
                    isActive={communicationLog.activeAgents.has(agentId)}
                  />
                ))
              )}
            </div>

            {/* Messages Area */}
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
              <div
                style={{
                  flex: 1,
                  overflowY: 'auto',
                  padding: 12,
                }}
              >
                {filteredMessages.length === 0 ? (
                  <div
                    style={{
                      textAlign: 'center',
                      color: '#666',
                      padding: 40,
                    }}
                  >
                    <div style={{ fontSize: 32, marginBottom: 12 }}>
                      {selectedAgentId === 'all' ? '🤫' : '💭'}
                    </div>
                    <div>
                      {selectedAgentId === 'all' 
                        ? 'No communications yet.' 
                        : `No conversation with Agent #${selectedAgentId} yet.`}
                    </div>
                    <div style={{ fontSize: 12, marginTop: 8, opacity: 0.7 }}>
                      {hasSelfAwareAgents 
                        ? 'Send a message to start a conversation!'
                        : 'Self-aware agents will appear when they evolve high consciousness.'}
                    </div>
                  </div>
                ) : (
                  filteredMessages.map((msg) => (
                    <MessageBubble 
                      key={msg.id} 
                      message={msg} 
                      showAgentId={selectedAgentId === 'all'}
                    />
                  ))
                )}
                <div ref={messagesEndRef} />
              </div>

              {/* Input Area */}
              <div
                style={{
                  padding: '12px',
                  borderTop: '1px solid #3a4a6a',
                  background: '#2a3a5a',
                }}
              >
                <div style={{ display: 'flex', gap: 8 }}>
                  <input
                    ref={inputRef}
                    type="text"
                    value={inputValue}
                    onChange={(e) => setInputValue(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') {
                        handleSend();
                      }
                    }}
                    placeholder={
                      !hasSelfAwareAgents
                        ? "Waiting for self-aware agents..."
                        : selectedAgentId === 'all'
                          ? "Message all agents..."
                          : `Message Agent #${selectedAgentId}...`
                    }
                    disabled={!hasSelfAwareAgents}
                    style={{
                      flex: 1,
                      background: '#1a2a4a',
                      border: '1px solid #3a4a6a',
                      borderRadius: 8,
                      padding: '10px 14px',
                      color: '#fff',
                      fontSize: 14,
                      outline: 'none',
                    }}
                  />
                  <button
                    onClick={handleSend}
                    disabled={!hasSelfAwareAgents || !inputValue.trim()}
                    style={{
                      background: hasSelfAwareAgents && inputValue.trim() 
                        ? 'linear-gradient(135deg, #4f46e5, #7c3aed)' 
                        : '#3a4a5a',
                      border: 'none',
                      borderRadius: 8,
                      padding: '10px 16px',
                      color: '#fff',
                      cursor: hasSelfAwareAgents && inputValue.trim() ? 'pointer' : 'not-allowed',
                      fontSize: 14,
                      fontWeight: 'bold',
                      transition: 'all 0.2s',
                    }}
                  >
                    📤
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* Footer stats */}
          <div
            style={{
              padding: '6px 16px',
              borderTop: '1px solid #3a4a6a',
              fontSize: 10,
              color: '#666',
              display: 'flex',
              justifyContent: 'space-between',
              background: '#151a30',
              borderRadius: '0 0 12px 12px',
            }}
          >
            <span>
              {selectedAgentId === 'all' 
                ? `${communicationLog.totalMessages} total messages`
                : `${filteredMessages.length} messages with #${selectedAgentId}`}
            </span>
            <span>Tick: {currentTick}</span>
          </div>
        </div>
      )}

      {/* CSS Animation styles */}
      <style>{`
        @keyframes pulse {
          0% { transform: scale(1); }
          50% { transform: scale(1.05); }
          100% { transform: scale(1); }
        }
        @keyframes blink {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
      `}</style>
    </>
  );
};

/**
 * Agent tab in sidebar
 */
const AgentTab: React.FC<{
  label: string;
  isSelected: boolean;
  onClick: () => void;
  messageCount: number;
  isActive: boolean;
}> = ({ label, isSelected, onClick, messageCount, isActive }) => (
  <button
    onClick={onClick}
    style={{
      width: '100%',
      padding: '8px 8px',
      marginBottom: 4,
      background: isSelected ? '#3a4a6a' : 'transparent',
      border: isSelected ? '1px solid #4a5a7a' : '1px solid transparent',
      borderRadius: 6,
      color: isSelected ? '#fff' : '#aaa',
      cursor: 'pointer',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      fontSize: 11,
      transition: 'all 0.2s',
    }}
  >
    <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
      {label}
      {isActive && (
        <span style={{
          width: 6,
          height: 6,
          background: '#00ff00',
          borderRadius: '50%',
          display: 'inline-block',
        }} />
      )}
    </span>
    {messageCount > 0 && (
      <span style={{
        background: isSelected ? '#4f46e5' : '#3a4a5a',
        color: '#fff',
        fontSize: 9,
        padding: '1px 5px',
        borderRadius: 8,
      }}>
        {messageCount}
      </span>
    )}
  </button>
);

/**
 * Individual message bubble component
 */
const MessageBubble: React.FC<{ 
  message: AgentMessage;
  showAgentId?: boolean;
}> = ({ message, showAgentId = true }) => {
  const typeColor = getTypeColor(message.type);
  const isUserMessage = message.isFromUser || message.type === MessageType.USER_MESSAGE;
  const isAgentResponse = message.type === MessageType.AGENT_RESPONSE;
  
  return (
    <div
      style={{
        marginBottom: 10,
        padding: 10,
        background: isUserMessage ? '#3a4a7a' : '#2a3a5a',
        borderRadius: 8,
        borderLeft: isUserMessage ? 'none' : `3px solid ${typeColor}`,
        borderRight: isUserMessage ? '3px solid #7c3aed' : 'none',
        marginLeft: isUserMessage ? 30 : 0,
        marginRight: isUserMessage ? 0 : 30,
      }}
    >
      {/* Header */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: 4,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <span
            style={{
              background: isUserMessage ? '#7c3aed' : typeColor,
              color: isUserMessage ? '#fff' : '#000',
              padding: '1px 5px',
              borderRadius: 4,
              fontSize: 9,
              fontWeight: 'bold',
            }}
          >
            {isUserMessage ? '👤 You' : showAgentId ? `#${message.agentId}` : '🤖'}
          </span>
          <span style={{ fontSize: 9, color: '#666' }}>
            {getMessageTypeLabel(message.type)}
          </span>
          {isAgentResponse && (
            <span style={{ fontSize: 9, color: '#7c3aed' }}>↩️</span>
          )}
        </div>
        <span style={{ fontSize: 9, color: '#555' }}>
          T{message.tick}
        </span>
      </div>

      {/* Target indicator for user messages */}
      {isUserMessage && message.targetAgentId && (
        <div style={{ fontSize: 9, color: '#888', marginBottom: 2 }}>
          → Agent #{message.targetAgentId}
        </div>
      )}

      {/* Message content */}
      <div
        style={{
          color: '#fff',
          fontSize: 13,
          lineHeight: 1.4,
          fontStyle: message.isQuestion ? 'italic' : 'normal',
        }}
      >
        {!isUserMessage && getMessageTypeEmoji(message.type) + ' '}
        "{message.content}"
      </div>

      {/* Context info */}
      {message.context && !isUserMessage && (
        <div
          style={{
            marginTop: 6,
            fontSize: 9,
            color: '#555',
            display: 'flex',
            gap: 8,
          }}
        >
          {message.context.energy !== undefined && (
            <span>⚡{message.context.energy}</span>
          )}
          {message.context.inventionCount !== undefined && message.context.inventionCount > 0 && (
            <span>💡{message.context.inventionCount}</span>
          )}
        </div>
      )}
    </div>
  );
};

/**
 * Get color for message type
 */
function getTypeColor(type: MessageType): string {
  switch (type) {
    case MessageType.EXISTENTIAL: return '#9b59b6';
    case MessageType.CURIOSITY: return '#3498db';
    case MessageType.SOCIAL: return '#2ecc71';
    case MessageType.SURVIVAL: return '#e74c3c';
    case MessageType.DISCOVERY: return '#f1c40f';
    case MessageType.REFLECTION: return '#1abc9c';
    case MessageType.GREETING: return '#ffd700';
    case MessageType.USER_MESSAGE: return '#7c3aed';
    case MessageType.AGENT_RESPONSE: return '#ec4899';
    default: return '#95a5a6';
  }
}

export default ChatPanel;
