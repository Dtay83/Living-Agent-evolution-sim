/**
 * Speech & Communication Panel
 * Displays language evolution, vocabulary growth, and active dialogues
 */

import React from 'react';
import type { CivilizationSpeechState, AgentLanguageState, CommunicationStyle } from '../speech-integration';
import type { Agent } from '../types';

interface SpeechPanelProps {
  speechState: CivilizationSpeechState;
  agents: Agent[];
  agentLanguages: Map<number, AgentLanguageState>;
  currentTick: number;
}

const STYLE_ICONS: Record<CommunicationStyle, string> = {
  primitive: '🗿',
  curious: '🔍',
  analytical: '🔬',
  philosophical: '🎭',
  social: '👥',
  eloquent: '📜'
};

const STYLE_COLORS: Record<CommunicationStyle, string> = {
  primitive: '#8b7355',
  curious: '#4caf50',
  analytical: '#2196f3',
  philosophical: '#9c27b0',
  social: '#ff9800',
  eloquent: '#ffd700'
};

const EVOLUTION_DESCRIPTIONS: Record<number, string> = {
  1: 'Primitive grunts and gestures',
  2: 'Basic words and simple sentences',
  3: 'Questions and conversations emerge',
  4: 'Abstract concepts being discussed',
  5: 'Scientific terminology developing',
  6: 'Complex philosophical discourse',
  7: 'Rich metaphorical language',
  8: 'Poetic and artistic expression',
  9: 'Transcendent communication',
  10: 'Universal language achieved'
};

export const SpeechPanel: React.FC<SpeechPanelProps> = ({
  speechState,
  agents,
  agentLanguages,
  currentTick
}) => {
  // Calculate summary stats
  let totalVocab = 0;
  let totalComplexity = 0;
  let count = 0;
  let mostEloquent: { id: number; vocabSize: number; style: CommunicationStyle } | null = null;
  const styleDistribution: Record<CommunicationStyle, number> = {
    primitive: 0,
    curious: 0,
    analytical: 0,
    philosophical: 0,
    social: 0,
    eloquent: 0
  };

  for (const agent of agents) {
    const lang = agentLanguages.get(agent.id);
    if (lang) {
      totalVocab += lang.vocabularySize;
      totalComplexity += lang.languageComplexity;
      count++;
      styleDistribution[lang.communicationStyle]++;

      if (!mostEloquent || lang.vocabularySize > mostEloquent.vocabSize) {
        mostEloquent = { id: agent.id, vocabSize: lang.vocabularySize, style: lang.communicationStyle };
      }
    }
  }

  const avgVocab = count > 0 ? (totalVocab / count).toFixed(1) : '0';
  const avgComplexity = count > 0 ? (totalComplexity / count).toFixed(2) : '0';

  // Get recent dialogues
  const recentDialogues = speechState.activeDialogues.slice(-3);

  // Calculate evolution progress
  const evolutionProgress = (speechState.languageEvolutionLevel / 10) * 100;

  return (
    <div style={{
      marginBottom: '12px',
      padding: '10px',
      background: '#151a30',
      borderRadius: '8px',
      border: '1px solid #333'
    }}>
      <h3>💬 Language & Communication</h3>

      {/* Language Evolution Level */}
      <div style={{ marginBottom: '12px' }}>
        <div style={{ 
          display: 'flex', 
          justifyContent: 'space-between', 
          alignItems: 'center',
          marginBottom: '4px'
        }}>
          <span style={{ fontSize: '0.9em', fontWeight: 'bold' }}>
            Language Evolution: Level {speechState.languageEvolutionLevel}
          </span>
          <span style={{ fontSize: '0.8em', opacity: 0.8 }}>
            {EVOLUTION_DESCRIPTIONS[speechState.languageEvolutionLevel] || 'Unknown'}
          </span>
        </div>
        <div style={{
          height: '8px',
          background: '#333',
          borderRadius: '4px',
          overflow: 'hidden'
        }}>
          <div style={{
            height: '100%',
            width: `${evolutionProgress}%`,
            background: 'linear-gradient(90deg, #4caf50, #2196f3, #9c27b0, #ffd700)',
            borderRadius: '4px',
            transition: 'width 0.3s ease'
          }} />
        </div>
      </div>

      {/* Overview Stats */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(2, 1fr)',
        gap: '8px',
        marginBottom: '12px',
        fontSize: '0.85em'
      }}>
        <div style={{ padding: '6px', background: '#1a2035', borderRadius: '4px' }}>
          <strong>Unique Words:</strong> {speechState.totalWordsKnown}
        </div>
        <div style={{ padding: '6px', background: '#1a2035', borderRadius: '4px' }}>
          <strong>Avg Vocabulary:</strong> {avgVocab} words
        </div>
        <div style={{ padding: '6px', background: '#1a2035', borderRadius: '4px' }}>
          <strong>Complexity:</strong> {avgComplexity}/10
        </div>
        <div style={{ padding: '6px', background: '#1a2035', borderRadius: '4px' }}>
          <strong>Knowledge Transfers:</strong> {speechState.knowledgeTransferCount}
        </div>
      </div>

      {/* Communication Style Distribution */}
      <div style={{ marginBottom: '12px' }}>
        <div style={{ fontSize: '0.85em', marginBottom: '6px', opacity: 0.9 }}>
          <strong>Communication Styles:</strong>
        </div>
        <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
          {Object.entries(styleDistribution)
            .filter(([_, count]) => count > 0)
            .map(([style, count]) => (
              <div
                key={style}
                style={{
                  padding: '4px 8px',
                  background: STYLE_COLORS[style as CommunicationStyle],
                  borderRadius: '4px',
                  fontSize: '0.8em',
                  color: '#fff',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px'
                }}
              >
                {STYLE_ICONS[style as CommunicationStyle]} {style}: {count}
              </div>
            ))}
        </div>
      </div>

      {/* Most Eloquent Agent */}
      {mostEloquent && (
        <div style={{
          marginBottom: '12px',
          padding: '8px',
          background: '#1a2035',
          borderRadius: '4px',
          border: `1px solid ${STYLE_COLORS[mostEloquent.style]}`
        }}>
          <div style={{ fontSize: '0.85em' }}>
            <strong>🏆 Most Eloquent:</strong> Agent #{mostEloquent.id}
          </div>
          <div style={{ fontSize: '0.8em', opacity: 0.8, marginTop: '2px' }}>
            {mostEloquent.vocabSize} words • {STYLE_ICONS[mostEloquent.style]} {mostEloquent.style} style
          </div>
        </div>
      )}

      {/* Active Dialogues */}
      {recentDialogues.length > 0 && (
        <div style={{ marginBottom: '12px' }}>
          <div style={{ fontSize: '0.85em', marginBottom: '6px', opacity: 0.9 }}>
            <strong>Recent Dialogues:</strong>
          </div>
          <ul style={{
            margin: 0,
            paddingLeft: '18px',
            fontSize: '0.8em',
            listStyle: 'none'
          }}>
            {recentDialogues.map((dialogue) => {
              const lastExchange = dialogue.exchanges[dialogue.exchanges.length - 1];
              return (
                <li key={dialogue.id} style={{ marginBottom: '4px' }}>
                  <span style={{ opacity: 0.7 }}>Agent {dialogue.initiatorId}</span>
                  {' ↔ '}
                  <span style={{ opacity: 0.7 }}>Agent {dialogue.responderId}</span>
                  {dialogue.knowledgeTransferred.length > 0 && (
                    <span style={{ color: '#4caf50' }}> 📚</span>
                  )}
                  <br />
                  <em style={{ opacity: 0.8 }}>"{lastExchange?.content.slice(0, 40)}..."</em>
                </li>
              );
            })}
          </ul>
        </div>
      )}

      {/* Neologisms (new words created) */}
      {speechState.neologisms.length > 0 && (
        <div>
          <div style={{ fontSize: '0.85em', marginBottom: '6px', opacity: 0.9 }}>
            <strong>New Words Created:</strong> {speechState.neologisms.length}
          </div>
          <div style={{ display: 'flex', gap: '4px', flexWrap: 'wrap' }}>
            {speechState.neologisms.slice(-5).map((word, idx) => (
              <span
                key={idx}
                style={{
                  padding: '2px 6px',
                  background: '#2a3a5a',
                  borderRadius: '3px',
                  fontSize: '0.75em',
                  fontStyle: 'italic'
                }}
              >
                "{word.word}"
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Shared Vocabulary Preview */}
      <div style={{ marginTop: '12px' }}>
        <div style={{ fontSize: '0.85em', marginBottom: '6px', opacity: 0.9 }}>
          <strong>Shared Vocabulary Sample:</strong>
        </div>        <div style={{ 
          display: 'flex', 
          gap: '4px', 
          flexWrap: 'wrap',
          maxHeight: '100px',
          overflowY: 'auto'
        }}>
          {speechState.sharedVocabulary.slice(0, 15).map((word, idx) => (
            <span
              key={idx}
              style={{
                padding: '2px 6px',
                background: word.category === 'scientific' ? '#2196f3' : 
                           word.category === 'philosophical' ? '#9c27b0' :
                           word.category === 'emotional' ? '#ff9800' : '#555',
                borderRadius: '3px',
                fontSize: '0.7em',
                color: '#fff'
              }}
              title={`${word.category} • complexity: ${word.complexity}`}
            >
              {word.word}
            </span>
          ))}
          {speechState.sharedVocabulary.length > 15 && (
            <span style={{ fontSize: '0.7em', opacity: 0.6 }}>
              +{speechState.sharedVocabulary.length - 15} more
            </span>
          )}
        </div>
      </div>
    </div>
  );
};
