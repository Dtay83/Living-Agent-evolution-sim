/**
 * Science Progress Panel
 * Displays current scientific era, progress, and recent discoveries
 */

import React from 'react';
import type { ScienceState } from '../science-system';

interface ScienceProgressPanelProps {
  scienceState: ScienceState | null;
  tick: number;
}

export const ScienceProgressPanel: React.FC<ScienceProgressPanelProps> = ({ scienceState, tick }) => {
  if (!scienceState) {
    return (
      <div style={{
        padding: 16,
        background: '#1a2332',
        borderRadius: 8,
        border: '1px solid #3d4f6a',
        marginTop: 12,
      }}>
        <h3 style={{ margin: '0 0 12px 0', color: '#e0e6ed' }}>🔬 Scientific Progress</h3>
        <p style={{ color: '#8b99ab', fontSize: 14 }}>
          Science system inactive. Agents must achieve sufficient knowledge to begin scientific discovery.
        </p>
      </div>
    );
  }

  const { currentEra, progressionMetrics, unlockedPhysics, unlockedMath } = scienceState;
  
  // Calculate progress to next era
  const nextEra = currentEra.level < 8 
    ? currentEra.requirements
    : {
        minDiscoveries: currentEra.requirements.minDiscoveries * 2,
        minPhysicsConcepts: Math.ceil(currentEra.requirements.minPhysicsConcepts * 1.5),
        minMathConcepts: Math.ceil(currentEra.requirements.minMathConcepts * 1.5),
      };
  
  const progressPercent = Math.min(100, 
    (progressionMetrics.totalDiscoveries / nextEra.minDiscoveries) * 100
  );
  
  // Get recent discoveries (last 5)
  const recentDiscoveries = scienceState.allDiscoveries.slice(-5).reverse();
  
  return (
    <div style={{
      padding: 16,
      background: '#1a2332',
      borderRadius: 8,
      border: '1px solid #3d4f6a',
      marginTop: 12,
    }}>
      <h3 style={{ margin: '0 0 12px 0', color: '#e0e6ed' }}>
        🔬 Scientific Era: {currentEra.name}
      </h3>
      
      {/* Era Progress Bar */}
      <div style={{ marginBottom: 12 }}>
        <div style={{ 
          display: 'flex', 
          justifyContent: 'space-between', 
          fontSize: 12, 
          color: '#8b99ab',
          marginBottom: 4
        }}>
          <span>Progress to Next Era</span>
          <span>{progressionMetrics.totalDiscoveries} / {nextEra.minDiscoveries} discoveries</span>
        </div>
        <div style={{
          width: '100%',
          height: 20,
          background: '#0d1621',
          borderRadius: 4,
          overflow: 'hidden',
          border: '1px solid #2a3b52'
        }}>
          <div style={{
            width: `${progressPercent}%`,
            height: '100%',
            background: `linear-gradient(90deg, #2563eb, #3b82f6)`,
            transition: 'width 0.3s ease'
          }} />
        </div>
      </div>
      
      {/* Statistics */}
      <div style={{ 
        display: 'grid', 
        gridTemplateColumns: '1fr 1fr', 
        gap: 8,
        marginBottom: 12,
        fontSize: 13
      }}>
        <div style={{ color: '#60a5fa' }}>
          <strong>Physics:</strong> {unlockedPhysics.length} concepts
        </div>
        <div style={{ color: '#a78bfa' }}>
          <strong>Mathematics:</strong> {unlockedMath.length} concepts
        </div>
        <div style={{ color: '#34d399' }}>
          <strong>Discovery Rate:</strong> {progressionMetrics.averageDiscoveryRate.toFixed(3)}/tick
        </div>
        <div style={{ color: '#fbbf24' }}>
          <strong>Era Duration:</strong> {tick - currentEra.startTick} ticks
        </div>
      </div>
      
      {/* Recent Discoveries */}
      {recentDiscoveries.length > 0 && (
        <div>
          <div style={{ 
            fontSize: 12, 
            fontWeight: 'bold', 
            color: '#8b99ab',
            marginBottom: 6
          }}>
            Recent Scientific Discoveries:
          </div>          <div style={{ fontSize: 12 }}>
            {recentDiscoveries.map((discovery, i) => (
              <div 
                key={i}
                style={{
                  padding: '4px 8px',
                  background: discovery.category === 'physics' ? '#1e3a5f' : '#2e1f47',
                  borderRadius: 4,
                  marginBottom: 4,
                  borderLeft: `3px solid ${discovery.category === 'physics' ? '#60a5fa' : '#a78bfa'}`
                }}
              >
                <span style={{ color: '#e0e6ed' }}>
                  {discovery.name}
                </span>
                <span style={{ color: '#8b99ab', marginLeft: 8 }}>
                  (Tick {discovery.discoveredAt})
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
      
      {recentDiscoveries.length === 0 && (
        <p style={{ color: '#8b99ab', fontSize: 13, fontStyle: 'italic' }}>
          No scientific discoveries yet. Agents with high curiosity may unlock physics and mathematics concepts!
        </p>
      )}
    </div>
  );
};
