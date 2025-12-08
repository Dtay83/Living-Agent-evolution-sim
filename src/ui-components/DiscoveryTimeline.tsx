/**
 * Discovery Timeline Component
 * Shows recent invention discoveries with details
 */

import React from 'react';
import type { DiscoveryEvent } from '../types';

interface DiscoveryTimelineProps {
  discoveries: DiscoveryEvent[];
}

export const DiscoveryTimeline: React.FC<DiscoveryTimelineProps> = ({ discoveries }) => {
  const recent = discoveries.slice(-15).reverse();
  
  return (
    <div style={{
      marginBottom: "12px",
      padding: "10px",
      background: "#151a30",
      borderRadius: "8px",
      border: "1px solid #333"
    }}>
      <h3>🔬 Discovery Timeline (Limitless Creativity)</h3>
      <p style={{ fontSize: "0.85em", opacity: 0.8, marginBottom: 8 }}>
        Total Discoveries: <strong>{discoveries.length}</strong> | No cap on inventions!
      </p>
      {recent.length === 0 ? (
        <p style={{ fontSize: "0.9em", opacity: 0.8 }}>
          No discoveries yet. Agents accumulate invention points through exploration and curiosity. 
          High creativity allows for more powerful inventions!
        </p>
      ) : (
        <ul style={{ paddingLeft: "18px", fontSize: "0.8em", maxHeight: 250, overflowY: "auto" }}>
          {recent.map((d) => (
            <li key={`${d.tick}-${d.agentId}-${d.invention.id}`} style={{ marginBottom: 6 }}>
              <strong>Tick {d.tick}</strong>: Agent {d.agentId} discovered{" "}
              <strong style={{ color: "#4caf50" }}>{d.invention.name}</strong>
              <br />
              <span style={{ opacity: 0.8, fontSize: "0.9em" }}>
                ({d.invention.type}) - {d.invention.description}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};
