/**
 * Invention Statistics Component
 * Displays aggregate invention and creativity metrics
 */

import React from 'react';
import type { Agent, DiscoveryEvent } from '../types';

interface InventionStatsProps {
  agents: Agent[];
  discoveries: DiscoveryEvent[];
}

export const InventionStats: React.FC<InventionStatsProps> = ({ agents, discoveries }) => {
  const totalInventions = discoveries.length;
  const avgInventionsPerAgent = agents.length > 0 
    ? agents.reduce((sum, a) => sum + a.inventions.length, 0) / agents.length 
    : 0;
  const mostInventive = agents.length > 0
    ? agents.reduce((max, a) => a.inventions.length > max.inventions.length ? a : max, agents[0])
    : null;
  const avgCreativity = agents.length > 0
    ? agents.reduce((sum, a) => sum + a.genes.creativity, 0) / agents.length
    : 0;
  
  return (
    <div style={{
      marginBottom: "12px",
      padding: "10px",
      background: "#151a30",
      borderRadius: "8px",
      border: "1px solid #333"
    }}>
      <h3>💡 Invention Statistics</h3>
      <div style={{ fontSize: "0.85em" }}>
        <p>
          <strong>Total Unique Inventions:</strong> {totalInventions} (Unlimited!)
        </p>
        <p>
          <strong>Avg Inventions per Agent:</strong> {avgInventionsPerAgent.toFixed(1)}
        </p>
        <p>
          <strong>Avg Creativity:</strong> {avgCreativity.toFixed(2)}
        </p>
        {mostInventive && (
          <p>
            <strong>Most Inventive Agent:</strong> #{mostInventive.id} with {mostInventive.inventions.length} inventions
          </p>
        )}
      </div>
    </div>
  );
};
