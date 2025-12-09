/**
 * ChatPanel Component
 * 
 * Displays agent communications with a chat icon indicator.
 * Shows when self-aware agents are asking questions or making statements.
 */

import React, { useState, useEffect, useRef } from 'react';
import { AgentMessage, CommunicationLog, MessageType, getMessageTypeEmoji, getMessageTypeLabel } from '../communication-system';

interface ChatPanelProps {
  communicationLog: CommunicationLog;
  isOpen: boolean;
  onToggle: () => void;
  onClear: () => void;
}

export const ChatPanel: React.FC<ChatPanelProps> = ({
  communicationLog,
  isOpen,
  onToggle,
  onClear,
}) => {
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [unreadCount, setUnreadCount] = useState(0);
  const [lastSeenCount, setLastSeenCount] = useState(0);

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
  }, [communicationLog.messages.length, isOpen]);

  const hasActiveAgents = communicationLog.activeAgents.size > 0;
  const recentMessages = communicationLog.messages.slice(-50); // Show last 50 messages

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

      {/* Chat Panel */}
      {isOpen && (
        <div
          style={{
            position: 'fixed',
            bottom: 90,
            right: 20,
            width: 380,
            maxHeight: 500,
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
                Agent Communications
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

          {/* Messages */}
          <div
            style={{
              flex: 1,
              overflowY: 'auto',
              padding: 12,
              maxHeight: 380,
            }}
          >
            {recentMessages.length === 0 ? (
              <div
                style={{
                  textAlign: 'center',
                  color: '#666',
                  padding: 40,
                }}
              >
                <div style={{ fontSize: 32, marginBottom: 12 }}>🤫</div>
                <div>No communications yet.</div>
                <div style={{ fontSize: 12, marginTop: 8, opacity: 0.7 }}>
                  Self-aware agents will appear here when they start asking questions.
                </div>
              </div>
            ) : (
              recentMessages.map((msg) => (
                <MessageBubble key={msg.id} message={msg} />
              ))
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Footer stats */}
          <div
            style={{
              padding: '8px 16px',
              borderTop: '1px solid #3a4a6a',
              fontSize: 11,
              color: '#888',
              display: 'flex',
              justifyContent: 'space-between',
            }}
          >
            <span>Total: {communicationLog.totalMessages} messages</span>
            {communicationLog.firstCommunicationTick && (
              <span>First contact: Tick {communicationLog.firstCommunicationTick}</span>
            )}
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
 * Individual message bubble component
 */
const MessageBubble: React.FC<{ message: AgentMessage }> = ({ message }) => {
  const typeColor = getTypeColor(message.type);
  
  return (
    <div
      style={{
        marginBottom: 12,
        padding: 12,
        background: '#2a3a5a',
        borderRadius: 8,
        borderLeft: `3px solid ${typeColor}`,
      }}
    >
      {/* Header */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: 6,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span
            style={{
              background: typeColor,
              color: '#000',
              padding: '2px 6px',
              borderRadius: 4,
              fontSize: 10,
              fontWeight: 'bold',
            }}
          >
            Agent #{message.agentId}
          </span>
          <span style={{ fontSize: 11, color: '#888' }}>
            {getMessageTypeLabel(message.type)}
          </span>
        </div>
        <span style={{ fontSize: 10, color: '#666' }}>
          Tick {message.tick}
        </span>
      </div>

      {/* Message content */}
      <div
        style={{
          color: '#fff',
          fontSize: 14,
          lineHeight: 1.4,
          fontStyle: message.isQuestion ? 'italic' : 'normal',
        }}
      >
        {getMessageTypeEmoji(message.type)} "{message.content}"
      </div>

      {/* Context info */}
      {message.context && (
        <div
          style={{
            marginTop: 8,
            fontSize: 10,
            color: '#666',
            display: 'flex',
            gap: 12,
          }}
        >
          {message.context.energy !== undefined && (
            <span>⚡ Energy: {message.context.energy}</span>
          )}
          {message.context.nearbyAgents !== undefined && message.context.nearbyAgents > 0 && (
            <span>👥 Nearby: {message.context.nearbyAgents}</span>
          )}
          {message.context.inventionCount !== undefined && message.context.inventionCount > 0 && (
            <span>💡 Inventions: {message.context.inventionCount}</span>
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
    default: return '#95a5a6';
  }
}

export default ChatPanel;
