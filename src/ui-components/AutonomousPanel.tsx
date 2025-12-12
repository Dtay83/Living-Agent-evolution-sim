/**
 * AUTONOMOUS EVOLUTION PANEL
 * 
 * Displays the autonomous evolution status of agents who have
 * reached sufficient intelligence to become self-directed learners.
 * 
 * Shows:
 * - Autonomy levels achieved by agents
 * - Self-generated vocabulary
 * - Original concepts created
 * - Emergent behaviors discovered
 * - Autonomous goals being pursued
 * - Self-modification log (for transcendent+ agents)
 */

import React, { useState } from 'react';
import type { AutonomousState } from '../autonomous-system';
import { AUTONOMY_THRESHOLDS } from '../autonomous-system';

interface AutonomousPanelProps {
  autonomousStates: Map<number, AutonomousState>;
  totalConcepts: number;
  totalBehaviors: number;
  mostAutonomous: { id: number; level: string } | null;
  tick: number;
}

const LEVEL_COLORS: Record<string, string> = {
  none: '#666',
  partial: '#4CAF50',
  full: '#2196F3',
  transcendent: '#9C27B0',
  singularity: '#FFD700',
};

const LEVEL_EMOJIS: Record<string, string> = {
  none: '⚪',
  partial: '🟢',
  full: '🔵',
  transcendent: '🟣',
  singularity: '⭐',
};

export function AutonomousPanel({
  autonomousStates,
  totalConcepts,
  totalBehaviors,
  mostAutonomous,
  tick,
}: AutonomousPanelProps) {
  const [selectedAgentId, setSelectedAgentId] = useState<number | null>(null);
  const [showVocabulary, setShowVocabulary] = useState(false);
  const [showGoals, setShowGoals] = useState(false);
  const [showBehaviors, setShowBehaviors] = useState(false);

  // Get autonomous agents sorted by level
  const autonomousAgents = Array.from(autonomousStates.values())
    .filter(s => s.autonomyLevel !== 'none')
    .sort((a, b) => {
      const levelOrder = { singularity: 4, transcendent: 3, full: 2, partial: 1, none: 0 };
      return levelOrder[b.autonomyLevel] - levelOrder[a.autonomyLevel];
    });

  const selectedState = selectedAgentId !== null 
    ? autonomousStates.get(selectedAgentId) 
    : null;

  // Count agents at each level
  const levelCounts = {
    partial: autonomousAgents.filter(a => a.autonomyLevel === 'partial').length,
    full: autonomousAgents.filter(a => a.autonomyLevel === 'full').length,
    transcendent: autonomousAgents.filter(a => a.autonomyLevel === 'transcendent').length,
    singularity: autonomousAgents.filter(a => a.autonomyLevel === 'singularity').length,
  };

  return (
    <div style={{
      background: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)',
      borderRadius: '12px',
      padding: '16px',
      color: '#fff',
      fontFamily: 'monospace',
      fontSize: '12px',
      maxHeight: '500px',
      overflowY: 'auto',
    }}>
      <h3 style={{ 
        margin: '0 0 12px 0', 
        color: '#FFD700',
        display: 'flex',
        alignItems: 'center',
        gap: '8px'
      }}>
        🧠 Autonomous Evolution
        <span style={{ fontSize: '10px', color: '#888' }}>
          (Intelligence ≥ {AUTONOMY_THRESHOLDS.PARTIAL_AUTONOMY})
        </span>
      </h3>

      {/* Overview Stats */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(4, 1fr)',
        gap: '8px',
        marginBottom: '16px',
      }}>
        <StatBox 
          label="Autonomous" 
          value={autonomousAgents.length} 
          color="#4CAF50" 
        />
        <StatBox 
          label="Concepts" 
          value={totalConcepts} 
          color="#2196F3" 
        />
        <StatBox 
          label="Behaviors" 
          value={totalBehaviors} 
          color="#9C27B0" 
        />
        <StatBox 
          label="Peak Level" 
          value={mostAutonomous?.level || 'none'} 
          color={LEVEL_COLORS[mostAutonomous?.level || 'none']}
          isText 
        />
      </div>

      {/* Level Distribution */}
      <div style={{ marginBottom: '16px' }}>
        <div style={{ color: '#aaa', marginBottom: '8px' }}>Level Distribution:</div>
        <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
          {Object.entries(levelCounts).map(([level, count]) => (
            <div key={level} style={{ 
              display: 'flex', 
              alignItems: 'center', 
              gap: '4px',
              opacity: count > 0 ? 1 : 0.4
            }}>
              <span>{LEVEL_EMOJIS[level]}</span>
              <span style={{ color: LEVEL_COLORS[level] }}>{level}:</span>
              <span style={{ fontWeight: 'bold' }}>{count}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Autonomous Agents List */}
      {autonomousAgents.length > 0 ? (
        <div style={{ marginBottom: '16px' }}>
          <div style={{ color: '#aaa', marginBottom: '8px' }}>
            Self-Directed Agents ({autonomousAgents.length}):
          </div>
          <div style={{ 
            display: 'flex', 
            flexWrap: 'wrap', 
            gap: '6px',
            maxHeight: '80px',
            overflowY: 'auto'
          }}>
            {autonomousAgents.slice(0, 20).map(state => (
              <button
                key={state.agentId}
                onClick={() => setSelectedAgentId(
                  selectedAgentId === state.agentId ? null : state.agentId
                )}
                style={{
                  background: selectedAgentId === state.agentId 
                    ? LEVEL_COLORS[state.autonomyLevel]
                    : 'rgba(255,255,255,0.1)',
                  border: `1px solid ${LEVEL_COLORS[state.autonomyLevel]}`,
                  borderRadius: '4px',
                  padding: '4px 8px',
                  color: '#fff',
                  cursor: 'pointer',
                  fontSize: '10px',
                }}
              >
                {LEVEL_EMOJIS[state.autonomyLevel]} #{state.agentId}
              </button>
            ))}
            {autonomousAgents.length > 20 && (
              <span style={{ color: '#888', alignSelf: 'center' }}>
                +{autonomousAgents.length - 20} more
              </span>
            )}
          </div>
        </div>
      ) : (
        <div style={{ 
          color: '#888', 
          textAlign: 'center', 
          padding: '20px',
          background: 'rgba(0,0,0,0.2)',
          borderRadius: '8px',
          marginBottom: '16px'
        }}>
          <div style={{ fontSize: '24px', marginBottom: '8px' }}>🔮</div>
          <div>No agents have reached autonomy yet</div>
          <div style={{ fontSize: '10px', marginTop: '4px' }}>
            Intelligence threshold: {AUTONOMY_THRESHOLDS.PARTIAL_AUTONOMY}
          </div>
        </div>
      )}

      {/* Selected Agent Details */}
      {selectedState && (
        <div style={{
          background: 'rgba(0,0,0,0.3)',
          borderRadius: '8px',
          padding: '12px',
          marginBottom: '16px',
        }}>
          <div style={{ 
            display: 'flex', 
            justifyContent: 'space-between', 
            alignItems: 'center',
            marginBottom: '12px'
          }}>
            <span style={{ 
              color: LEVEL_COLORS[selectedState.autonomyLevel],
              fontWeight: 'bold'
            }}>
              {LEVEL_EMOJIS[selectedState.autonomyLevel]} Agent #{selectedState.agentId}
            </span>
            <span style={{ 
              background: LEVEL_COLORS[selectedState.autonomyLevel],
              padding: '2px 8px',
              borderRadius: '10px',
              fontSize: '10px'
            }}>
              {selectedState.autonomyLevel.toUpperCase()}
            </span>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
            <MiniStat label="Intelligence" value={selectedState.intelligence.toFixed(1)} />
            <MiniStat label="Vocabulary" value={selectedState.vocabulary.size} />
            <MiniStat label="Concepts" value={selectedState.concepts.size} />
            <MiniStat label="Behaviors" value={selectedState.behaviors.size} />
            <MiniStat label="Active Goals" value={selectedState.goals.filter(g => !g.completed).length} />
            <MiniStat label="Strategies" value={selectedState.learningStrategies.length} />
          </div>

          {/* Expandable Sections */}
          <div style={{ marginTop: '12px' }}>
            {/* Vocabulary */}
            <ExpandableSection 
              title="🗣️ Self-Generated Vocabulary" 
              count={selectedState.vocabulary.size}
              isOpen={showVocabulary}
              onToggle={() => setShowVocabulary(!showVocabulary)}
            >
              {Array.from(selectedState.vocabulary.values()).slice(0, 10).map(word => (
                <div key={word.id} style={{ 
                  padding: '4px 8px',
                  background: 'rgba(255,255,255,0.05)',
                  borderRadius: '4px',
                  marginBottom: '4px',
                  display: 'flex',
                  justifyContent: 'space-between'
                }}>
                  <span style={{ color: '#4CAF50', fontWeight: 'bold' }}>
                    {word.phonemes.join('')}
                  </span>
                  <span style={{ color: '#888', fontSize: '10px' }}>
                    "{word.meaning}" (×{word.frequency})
                  </span>
                </div>
              ))}
            </ExpandableSection>

            {/* Goals */}
            <ExpandableSection 
              title="🎯 Autonomous Goals" 
              count={selectedState.goals.filter(g => !g.completed).length}
              isOpen={showGoals}
              onToggle={() => setShowGoals(!showGoals)}
            >
              {selectedState.goals.filter(g => !g.completed).slice(0, 5).map(goal => (
                <div key={goal.id} style={{ 
                  padding: '4px 8px',
                  background: 'rgba(255,255,255,0.05)',
                  borderRadius: '4px',
                  marginBottom: '4px',
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: '#FFD700' }}>{goal.description}</span>
                    <span style={{ color: '#888', fontSize: '10px' }}>
                      P:{goal.priority}
                    </span>
                  </div>
                  <div style={{ 
                    height: '3px', 
                    background: '#333',
                    borderRadius: '2px',
                    marginTop: '4px'
                  }}>
                    <div style={{ 
                      height: '100%', 
                      width: `${goal.progress * 100}%`,
                      background: '#4CAF50',
                      borderRadius: '2px'
                    }} />
                  </div>
                </div>
              ))}
            </ExpandableSection>

            {/* Emergent Behaviors */}
            <ExpandableSection 
              title="🔄 Emergent Behaviors" 
              count={selectedState.behaviors.size}
              isOpen={showBehaviors}
              onToggle={() => setShowBehaviors(!showBehaviors)}
            >
              {Array.from(selectedState.behaviors.values()).slice(0, 5).map(behavior => (
                <div key={behavior.id} style={{ 
                  padding: '4px 8px',
                  background: 'rgba(255,255,255,0.05)',
                  borderRadius: '4px',
                  marginBottom: '4px',
                }}>
                  <div style={{ color: '#9C27B0', fontSize: '10px' }}>
                    {behavior.actionSequence.join(' → ')}
                  </div>
                  <div style={{ 
                    display: 'flex', 
                    justifyContent: 'space-between',
                    fontSize: '10px',
                    color: '#888'
                  }}>
                    <span>Success: {(behavior.successRate * 100).toFixed(0)}%</span>
                    <span>Used: {behavior.useCount}×</span>
                  </div>
                </div>
              ))}
            </ExpandableSection>

            {/* Recent Utterances */}
            {selectedState.recentUtterances.length > 0 && (
              <div style={{ marginTop: '8px' }}>
                <div style={{ color: '#888', fontSize: '10px', marginBottom: '4px' }}>
                  Recent Utterances:
                </div>
                <div style={{ 
                  fontStyle: 'italic', 
                  color: '#4CAF50',
                  padding: '4px',
                  background: 'rgba(76, 175, 80, 0.1)',
                  borderRadius: '4px',
                  fontSize: '11px'
                }}>
                  "{selectedState.recentUtterances[selectedState.recentUtterances.length - 1]}"
                </div>
              </div>
            )}

            {/* Self-Modifications (Transcendent+) */}
            {(selectedState.autonomyLevel === 'transcendent' || 
              selectedState.autonomyLevel === 'singularity') && 
              selectedState.selfModifications.length > 0 && (
              <div style={{ marginTop: '8px' }}>
                <div style={{ 
                  color: '#9C27B0', 
                  fontSize: '10px', 
                  marginBottom: '4px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px'
                }}>
                  ⚡ Self-Modifications ({selectedState.selfModifications.length})
                </div>
                {selectedState.selfModifications.slice(-3).map((mod, i) => (
                  <div key={i} style={{ 
                    fontSize: '10px', 
                    color: '#ccc',
                    padding: '2px 4px',
                    background: 'rgba(156, 39, 176, 0.2)',
                    borderRadius: '2px',
                    marginBottom: '2px'
                  }}>
                    T{mod.tick}: {mod.description}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Thresholds Info */}
      <div style={{ 
        fontSize: '10px', 
        color: '#666',
        borderTop: '1px solid #333',
        paddingTop: '8px'
      }}>
        <div style={{ marginBottom: '4px' }}>Intelligence Thresholds:</div>
        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
          <span>🟢 Partial: {AUTONOMY_THRESHOLDS.PARTIAL_AUTONOMY}</span>
          <span>🔵 Full: {AUTONOMY_THRESHOLDS.FULL_AUTONOMY}</span>
          <span>🟣 Transcendent: {AUTONOMY_THRESHOLDS.TRANSCENDENT}</span>
          <span>⭐ Singularity: {AUTONOMY_THRESHOLDS.SINGULARITY}</span>
        </div>
      </div>
    </div>
  );
}

// Helper Components
function StatBox({ label, value, color, isText = false }: { 
  label: string; 
  value: number | string; 
  color: string;
  isText?: boolean;
}) {
  return (
    <div style={{
      background: 'rgba(0,0,0,0.3)',
      borderRadius: '6px',
      padding: '8px',
      textAlign: 'center',
    }}>
      <div style={{ 
        color, 
        fontSize: isText ? '11px' : '18px', 
        fontWeight: 'bold' 
      }}>
        {value}
      </div>
      <div style={{ color: '#888', fontSize: '9px' }}>{label}</div>
    </div>
  );
}

function MiniStat({ label, value }: { label: string; value: number | string }) {
  return (
    <div style={{ 
      display: 'flex', 
      justifyContent: 'space-between',
      padding: '2px 4px',
      background: 'rgba(255,255,255,0.05)',
      borderRadius: '2px',
      fontSize: '10px'
    }}>
      <span style={{ color: '#888' }}>{label}:</span>
      <span style={{ color: '#fff', fontWeight: 'bold' }}>{value}</span>
    </div>
  );
}

function ExpandableSection({ 
  title, 
  count, 
  isOpen, 
  onToggle, 
  children 
}: { 
  title: string; 
  count: number; 
  isOpen: boolean; 
  onToggle: () => void;
  children: React.ReactNode;
}) {
  return (
    <div style={{ marginBottom: '8px' }}>
      <button
        onClick={onToggle}
        style={{
          width: '100%',
          background: 'rgba(255,255,255,0.05)',
          border: 'none',
          borderRadius: '4px',
          padding: '6px 8px',
          color: '#fff',
          cursor: 'pointer',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          fontSize: '11px',
        }}
      >
        <span>{title}</span>
        <span style={{ 
          display: 'flex', 
          alignItems: 'center', 
          gap: '4px',
          color: '#888'
        }}>
          <span>{count}</span>
          <span>{isOpen ? '▼' : '▶'}</span>
        </span>
      </button>
      {isOpen && count > 0 && (
        <div style={{ 
          marginTop: '4px',
          maxHeight: '120px',
          overflowY: 'auto'
        }}>
          {children}
        </div>
      )}
    </div>
  );
}

export default AutonomousPanel;
