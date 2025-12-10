/**
 * Physics Progress Panel
 * Displays civilization's physics discoveries and bonuses
 */

import React from 'react';
import type { PhysicsConcept } from '../science-system/physics';
import { getPhysicsSummary } from '../physics-integration';

interface PhysicsPanelProps {
  unlockedConcepts: PhysicsConcept[];
  lastDiscoveryTick: number;
  currentTick: number;
}

const CATEGORY_ICONS: Record<string, string> = {
  mechanics: '⚙️',
  thermodynamics: '🔥',
  electromagnetism: '⚡',
  quantum: '🔮',
  relativity: '🌌',
  unified: '🌟',
  cosmology: '🪐',
  particle: '⚛️',
  plasma: '☀️',
  condensed_matter: '💎',
  exotic: '🌀',
  transcendent: '✨'
};

const CATEGORY_COLORS: Record<string, string> = {
  mechanics: '#4caf50',
  thermodynamics: '#ff5722',
  electromagnetism: '#ffeb3b',
  quantum: '#9c27b0',
  relativity: '#2196f3',
  unified: '#e91e63',
  cosmology: '#673ab7',
  particle: '#00bcd4',
  plasma: '#ff9800',
  condensed_matter: '#795548',
  exotic: '#00e676',
  transcendent: '#d500f9'
};

export const PhysicsPanel: React.FC<PhysicsPanelProps> = ({
  unlockedConcepts,
  lastDiscoveryTick,
  currentTick
}) => {
  const summary = getPhysicsSummary(unlockedConcepts);
  const ticksSinceDiscovery = lastDiscoveryTick > 0 ? currentTick - lastDiscoveryTick : -1;
  
  // Get recent discoveries (last 5)
  const recentDiscoveries = [...unlockedConcepts]
    .sort((a, b) => (b.discoveredAt || 0) - (a.discoveredAt || 0))
    .slice(0, 5);

  return (
    <div style={{
      marginBottom: '12px',
      padding: '10px',
      background: '#151a30',
      borderRadius: '8px',
      border: '1px solid #333'
    }}>
      <h3>⚛️ Physics Discoveries</h3>
        {/* Overview Stats */}
      <div style={{ 
        display: 'flex', 
        gap: '16px', 
        marginBottom: '12px',
        fontSize: '0.85em'
      }}>
        <div>
          <strong>Total Concepts:</strong> {summary.totalConcepts}
        </div>
        {summary.proceduralLevel > 0 && (
          <div style={{ color: '#d500f9' }}>
            <strong>🌟 Transcendent Level:</strong> {summary.proceduralLevel}
          </div>
        )}
        {ticksSinceDiscovery >= 0 && (
          <div style={{ opacity: 0.8 }}>
            Last discovery: {ticksSinceDiscovery} ticks ago
          </div>
        )}
      </div>

      {/* Category Breakdown */}
      {Object.keys(summary.byCategory).length > 0 && (
        <div style={{ marginBottom: '12px' }}>
          <div style={{ fontSize: '0.85em', marginBottom: '6px', opacity: 0.9 }}>
            <strong>By Category:</strong>
          </div>
          <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
            {Object.entries(summary.byCategory).map(([category, count]) => (
              <div
                key={category}
                style={{
                  padding: '4px 8px',
                  background: CATEGORY_COLORS[category] || '#666',
                  borderRadius: '4px',
                  fontSize: '0.8em',
                  color: category === 'electromagnetism' ? '#000' : '#fff'
                }}
              >
                {CATEGORY_ICONS[category] || '📚'} {category}: {count}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Active Bonuses */}
      {summary.topBonuses.length > 0 && (
        <div style={{ marginBottom: '12px' }}>
          <div style={{ fontSize: '0.85em', marginBottom: '6px', opacity: 0.9 }}>
            <strong>Active Bonuses:</strong>
          </div>
          <ul style={{ 
            margin: 0, 
            paddingLeft: '18px', 
            fontSize: '0.8em',
            listStyle: 'none'
          }}>
            {summary.topBonuses.map((bonus, idx) => (
              <li key={idx} style={{ marginBottom: '2px' }}>
                ✨ <strong>{bonus.name}:</strong> {bonus.value}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Recent Discoveries */}
      {recentDiscoveries.length > 0 ? (
        <div>
          <div style={{ fontSize: '0.85em', marginBottom: '6px', opacity: 0.9 }}>
            <strong>Recent Discoveries:</strong>
          </div>
          <ul style={{ 
            margin: 0, 
            paddingLeft: '18px', 
            fontSize: '0.8em',
            maxHeight: '120px',
            overflowY: 'auto'
          }}>
            {recentDiscoveries.map(concept => (
              <li key={concept.id} style={{ marginBottom: '4px' }}>
                {CATEGORY_ICONS[concept.category] || '📚'}{' '}
                <strong style={{ color: CATEGORY_COLORS[concept.category] || '#fff' }}>
                  {concept.name}
                </strong>
                <br />
                <span style={{ opacity: 0.7, fontSize: '0.9em' }}>
                  {concept.description}
                  {concept.discoveredAt && concept.discoveredBy && (
                    <> (Tick {concept.discoveredAt}, Agent {concept.discoveredBy})</>
                  )}
                </span>
              </li>
            ))}
          </ul>
        </div>
      ) : (
        <p style={{ fontSize: '0.85em', opacity: 0.7, margin: 0 }}>
          No physics discoveries yet. Agents with high curiosity and creativity 
          can discover physics concepts that benefit the entire civilization!
        </p>
      )}
    </div>
  );
};

export default PhysicsPanel;
