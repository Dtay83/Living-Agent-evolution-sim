/**
 * Era Progress Panel
 * Displays civilization's current era, milestones, and progression
 */

import React from 'react';
import type { CivilizationEraState, EpochMilestone, EraAgentBonuses } from '../era-integration';
import { getFullEraSummary } from '../era-integration';

interface EraPanelProps {
  eraState: CivilizationEraState;
  physicsCount: number;
  mathCount: number;
  currentTick: number;
}

const MILESTONE_ICONS: Record<string, string> = {
  discovery: '🔬',
  population: '👥',
  invention: '💡',
  collaboration: '🤝',
  evolution: '🏛️'
};

const ERA_ICONS: Record<number, string> = {
  0: '🪨',  // Stone Age
  1: '🥉',  // Bronze Age
  2: '⚔️',  // Iron Age
  3: '🏛️',  // Classical Age
  4: '🎨',  // Renaissance
  5: '🏭',  // Industrial Age
  6: '💻',  // Information Age
  7: '⚛️',  // Quantum Age
  8: '🌌',  // Singularity Age
};

const ERA_COLORS: Record<number, string> = {
  0: '#8d6e63',  // Brown - Stone
  1: '#cd7f32',  // Bronze
  2: '#607d8b',  // Steel gray - Iron
  3: '#d4af37',  // Gold - Classical
  4: '#9c27b0',  // Purple - Renaissance
  5: '#795548',  // Dark brown - Industrial
  6: '#2196f3',  // Blue - Information
  7: '#00bcd4',  // Cyan - Quantum
  8: '#673ab7',  // Deep purple - Singularity
};

export const EraPanel: React.FC<EraPanelProps> = ({
  eraState,
  physicsCount,
  mathCount,
  currentTick
}) => {
  const summary = getFullEraSummary(eraState, physicsCount, mathCount);
  const ticksSinceEra = eraState.lastEraAdvanceTick > 0 
    ? currentTick - eraState.lastEraAdvanceTick 
    : currentTick;
  
  const eraColor = ERA_COLORS[summary.currentEraLevel] || '#666';
  const eraIcon = ERA_ICONS[summary.currentEraLevel] || '🌟';
  const progressPercent = Math.round(summary.progressToNextEra * 100);

  return (
    <div style={{
      marginBottom: '12px',
      padding: '10px',
      background: '#151a30',
      borderRadius: '8px',
      border: '1px solid #333'
    }}>
      {/* Era Header */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '10px',
        marginBottom: '12px'
      }}>
        <span style={{ fontSize: '1.5em' }}>{eraIcon}</span>
        <div>
          <h3 style={{ 
            margin: 0, 
            color: eraColor,
            textShadow: `0 0 10px ${eraColor}40`
          }}>
            {summary.currentEraName}
          </h3>
          <div style={{ fontSize: '0.75em', opacity: 0.7 }}>
            Era Level {summary.currentEraLevel} • {ticksSinceEra} ticks in this era
          </div>
        </div>
      </div>

      {/* Era Description */}
      <div style={{
        padding: '8px',
        background: `${eraColor}15`,
        borderRadius: '4px',
        marginBottom: '12px',
        fontSize: '0.85em',
        fontStyle: 'italic',
        borderLeft: `3px solid ${eraColor}`
      }}>
        {summary.currentEraDescription}
      </div>

      {/* Progress to Next Era */}
      <div style={{ marginBottom: '12px' }}>
        <div style={{ 
          fontSize: '0.85em', 
          marginBottom: '6px', 
          display: 'flex',
          justifyContent: 'space-between'
        }}>
          <strong>Progress to Next Era:</strong>
          <span>{progressPercent}%</span>
        </div>
        
        {/* Progress Bar */}
        <div style={{
          height: '8px',
          background: '#333',
          borderRadius: '4px',
          overflow: 'hidden'
        }}>
          <div style={{
            height: '100%',
            width: `${progressPercent}%`,
            background: `linear-gradient(90deg, ${eraColor}, ${eraColor}dd)`,
            borderRadius: '4px',
            transition: 'width 0.3s ease'
          }} />
        </div>
        
        {/* Requirements Breakdown */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(3, 1fr)',
          gap: '8px',
          marginTop: '8px',
          fontSize: '0.75em'
        }}>
          <div style={{
            padding: '4px 6px',
            background: '#1a1f35',
            borderRadius: '4px',
            textAlign: 'center'
          }}>
            <div style={{ opacity: 0.7 }}>Discoveries</div>
            <div style={{ 
              color: summary.nextEraRequirements.discoveries.current >= summary.nextEraRequirements.discoveries.required 
                ? '#4caf50' : '#fff'
            }}>
              {summary.nextEraRequirements.discoveries.current}/{summary.nextEraRequirements.discoveries.required}
            </div>
          </div>
          <div style={{
            padding: '4px 6px',
            background: '#1a1f35',
            borderRadius: '4px',
            textAlign: 'center'
          }}>
            <div style={{ opacity: 0.7 }}>Physics</div>
            <div style={{ 
              color: summary.nextEraRequirements.physics.current >= summary.nextEraRequirements.physics.required 
                ? '#4caf50' : '#fff'
            }}>
              {summary.nextEraRequirements.physics.current}/{summary.nextEraRequirements.physics.required}
            </div>
          </div>
          <div style={{
            padding: '4px 6px',
            background: '#1a1f35',
            borderRadius: '4px',
            textAlign: 'center'
          }}>
            <div style={{ opacity: 0.7 }}>Math</div>
            <div style={{ 
              color: summary.nextEraRequirements.math.current >= summary.nextEraRequirements.math.required 
                ? '#4caf50' : '#fff'
            }}>
              {summary.nextEraRequirements.math.current}/{summary.nextEraRequirements.math.required}
            </div>
          </div>
        </div>
      </div>

      {/* Era Bonuses */}
      <div style={{ marginBottom: '12px' }}>
        <div style={{ fontSize: '0.85em', marginBottom: '6px', opacity: 0.9 }}>
          <strong>Era Bonuses:</strong>
        </div>
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(2, 1fr)',
          gap: '6px',
          fontSize: '0.75em'
        }}>
          <BonusItem 
            label="Learning" 
            value={`+${Math.round((summary.bonuses.learningMultiplier - 1) * 100)}%`}
            positive={summary.bonuses.learningMultiplier > 1}
          />
          <BonusItem 
            label="Discovery" 
            value={`+${(summary.bonuses.discoveryChance * 100).toFixed(1)}%`}
            positive={summary.bonuses.discoveryChance > 0}
          />
          <BonusItem 
            label="Energy Cost" 
            value={`${Math.round((1 - summary.bonuses.energyEfficiency) * 100)}% reduced`}
            positive={summary.bonuses.energyEfficiency < 1}
          />
          <BonusItem 
            label="Knowledge Retention" 
            value={`${Math.round(summary.bonuses.knowledgeRetention * 100)}%`}
            positive={summary.bonuses.knowledgeRetention > 0.1}
          />
        </div>
      </div>

      {/* Milestones */}
      <div>
        <div style={{ 
          fontSize: '0.85em', 
          marginBottom: '6px', 
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center'
        }}>
          <strong>Milestones:</strong>
          <span style={{ opacity: 0.7 }}>
            🏆 {summary.totalMilestones} achieved
          </span>
        </div>
        
        {summary.recentMilestones.length > 0 ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
            {summary.recentMilestones.map((milestone, idx) => (
              <MilestoneItem key={milestone.id} milestone={milestone} />
            ))}
          </div>
        ) : (
          <div style={{ 
            fontSize: '0.8em', 
            opacity: 0.6, 
            textAlign: 'center',
            padding: '8px'
          }}>
            No milestones achieved yet. Keep exploring!
          </div>
        )}
      </div>
    </div>
  );
};

// Helper component for bonus display
const BonusItem: React.FC<{ label: string; value: string; positive: boolean }> = ({ 
  label, value, positive 
}) => (
  <div style={{
    padding: '4px 6px',
    background: positive ? '#1a2f1a' : '#1a1f35',
    borderRadius: '4px',
    display: 'flex',
    justifyContent: 'space-between'
  }}>
    <span style={{ opacity: 0.8 }}>{label}:</span>
    <span style={{ color: positive ? '#4caf50' : '#aaa' }}>{value}</span>
  </div>
);

// Helper component for milestone display
const MilestoneItem: React.FC<{ milestone: EpochMilestone }> = ({ milestone }) => (
  <div style={{
    padding: '6px 8px',
    background: '#1a1f35',
    borderRadius: '4px',
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    fontSize: '0.8em'
  }}>
    <span>{MILESTONE_ICONS[milestone.type] || '🌟'}</span>
    <div style={{ flex: 1 }}>
      <div style={{ fontWeight: 'bold' }}>{milestone.name}</div>
      <div style={{ opacity: 0.7, fontSize: '0.9em' }}>{milestone.description}</div>
    </div>
    <div style={{ opacity: 0.5, fontSize: '0.85em' }}>
      T{milestone.achievedAt}
    </div>
  </div>
);
