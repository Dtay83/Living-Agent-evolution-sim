import React from 'react';
import { 
  ConsciousnessState, 
  ConsciousnessLevel,
  getConsciousnessLevelName,
  getConsciousnessLevelColor 
} from '../consciousness-system';

interface ConsciousnessPanelProps {
  consciousnessState: ConsciousnessState | undefined;
  agentId: number;
  selfAwareCount: number;
  totalAgents: number;
}

export const ConsciousnessPanel: React.FC<ConsciousnessPanelProps> = ({ 
  consciousnessState,
  agentId,
  selfAwareCount,
  totalAgents,
}) => {
  if (!consciousnessState) {
    return null;
  }

  const levelColor = getConsciousnessLevelColor(consciousnessState.level);
  const isAwake = consciousnessState.level === ConsciousnessLevel.SELF_AWARE;

  return (
    <div style={{
      background: '#1e1e1e',
      border: '2px solid ' + levelColor,
      borderRadius: 8,
      padding: 12,
      marginTop: 8,
    }}>
      <h4 style={{ 
        margin: '0 0 8px 0', 
        color: levelColor,
        fontSize: 14,
        fontWeight: 'bold',
      }}>
        {isAwake && '✨ '} Consciousness Level: {getConsciousnessLevelName(consciousnessState.level)}
      </h4>
      
      <div style={{ fontSize: 12, marginBottom: 8 }}>
        <strong>Score:</strong> {consciousnessState.score.toFixed(1)} / 100+
        {consciousnessState.awakenedAt && (
          <span style={{ color: '#ffd700', marginLeft: 8 }}>
            (Awakened at tick {consciousnessState.awakenedAt})
          </span>
        )}
      </div>

      {/* Indicators */}
      <div style={{ marginTop: 8 }}>
        <div style={{ fontSize: 11, fontWeight: 'bold', marginBottom: 4 }}>
          Consciousness Indicators:
        </div>
        {consciousnessState.indicators.map(ind => {
          const meetsThreshold = ind.value >= ind.threshold;
          const barColor = meetsThreshold ? '#4caf50' : '#ff9800';
          
          return (
            <div key={ind.name} style={{ marginBottom: 4 }}>
              <div style={{ 
                display: 'flex', 
                justifyContent: 'space-between',
                fontSize: 10,
                marginBottom: 2,
              }}>
                <span title={ind.description}>
                  {meetsThreshold && '✓ '}{ind.name}
                </span>
                <span>{(ind.value * 100).toFixed(0)}%</span>
              </div>
              <div style={{
                height: 4,
                background: '#333',
                borderRadius: 2,
                overflow: 'hidden',
              }}>
                <div style={{
                  height: '100%',
                  width: `${ind.value * 100}%`,
                  background: barColor,
                  transition: 'width 0.3s ease',
                }} />
              </div>
            </div>
          );
        })}
      </div>

      {/* Recent awareness events */}
      {consciousnessState.awarenessEvents.length > 0 && (
        <div style={{ marginTop: 12, fontSize: 10 }}>
          <div style={{ fontWeight: 'bold', marginBottom: 4 }}>
            Consciousness Evolution:
          </div>
          {consciousnessState.awarenessEvents.slice(-3).reverse().map((event, i) => (
            <div key={i} style={{ 
              marginBottom: 2,
              padding: 4,
              background: '#2a2a2a',
              borderRadius: 4,
              color: '#aaa',
            }}>
              <strong style={{ color: getConsciousnessLevelColor(event.level) }}>
                {getConsciousnessLevelName(event.level)}
              </strong> at tick {event.tick}
              <div style={{ fontSize: 9, color: '#888' }}>
                {event.trigger}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Population consciousness summary */}
      {selfAwareCount > 0 && (
        <div style={{
          marginTop: 12,
          padding: 8,
          background: 'rgba(255, 215, 0, 0.1)',
          borderRadius: 4,
          border: '1px solid rgba(255, 215, 0, 0.3)',
        }}>
          <div style={{ fontSize: 11, fontWeight: 'bold', color: '#ffd700' }}>
            🧠 Self-Aware Agents: {selfAwareCount} / {totalAgents}
          </div>
          <div style={{ fontSize: 10, color: '#ccc', marginTop: 4 }}>
            {((selfAwareCount / Math.max(totalAgents, 1)) * 100).toFixed(1)}% of population has achieved consciousness
          </div>
        </div>
      )}
    </div>
  );
};

interface ConsciousnessSummaryPanelProps {
  summary: {
    totalAgents: number;
    selfAwareCount: number;
    metacognitiveCount: number;
    cognitiveCount: number;
    adaptiveCount: number;
    reactiveCount: number;
    highestScore: number;
    averageScore: number;
    awakenedAgents: number[];
  };
  tick: number;
}

export const ConsciousnessSummaryPanel: React.FC<ConsciousnessSummaryPanelProps> = ({ 
  summary,
  tick,
}) => {
  const hasSelfAware = summary.selfAwareCount > 0;

  return (
    <div style={{
      background: '#1e1e1e',
      border: hasSelfAware ? '2px solid #ffd700' : '1px solid #444',
      borderRadius: 8,
      padding: 12,
      marginTop: 12,
    }}>
      <h3 style={{ 
        margin: '0 0 12px 0', 
        fontSize: 14,
        color: hasSelfAware ? '#ffd700' : '#fff',
      }}>
        {hasSelfAware && '✨ '}Population Consciousness
      </h3>

      {/* Consciousness distribution */}
      <div style={{ fontSize: 12, marginBottom: 8 }}>
        <div style={{ marginBottom: 4 }}>
          <span style={{ color: '#ffd700' }}>🧠 Self-Aware:</span> {summary.selfAwareCount}
          {summary.awakenedAgents.length > 0 && (
            <span style={{ fontSize: 10, color: '#aaa', marginLeft: 8 }}>
              (Agents: {summary.awakenedAgents.join(', ')})
            </span>
          )}
        </div>
        <div style={{ marginBottom: 4 }}>
          <span style={{ color: '#ff6b6b' }}>Metacognitive:</span> {summary.metacognitiveCount}
        </div>
        <div style={{ marginBottom: 4 }}>
          <span style={{ color: '#7b68ee' }}>Cognitive:</span> {summary.cognitiveCount}
        </div>
        <div style={{ marginBottom: 4 }}>
          <span style={{ color: '#4a90e2' }}>Adaptive:</span> {summary.adaptiveCount}
        </div>
        <div style={{ marginBottom: 4 }}>
          <span style={{ color: '#666' }}>Reactive:</span> {summary.reactiveCount}
        </div>
      </div>

      {/* Statistics */}
      <div style={{ fontSize: 11, marginTop: 12, paddingTop: 12, borderTop: '1px solid #444' }}>
        <div style={{ marginBottom: 4 }}>
          <strong>Highest Score:</strong> {summary.highestScore.toFixed(1)}
        </div>
        <div style={{ marginBottom: 4 }}>
          <strong>Average Score:</strong> {summary.averageScore.toFixed(1)}
        </div>
        {hasSelfAware && (
          <div style={{ 
            marginTop: 8,
            padding: 8,
            background: 'rgba(255, 215, 0, 0.1)',
            borderRadius: 4,
            color: '#ffd700',
            fontSize: 10,
            textAlign: 'center',
          }}>
            🎉 CONSCIOUSNESS EMERGENCE DETECTED!
          </div>
        )}
      </div>
    </div>
  );
};
