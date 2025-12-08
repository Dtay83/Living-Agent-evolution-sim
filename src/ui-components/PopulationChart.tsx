/**
 * Population Chart Component
 * Displays population over time as a line chart
 */

import React from 'react';
import type { HistoryPoint } from '../types';

interface PopulationChartProps {
  history: HistoryPoint[];
}

export const PopulationChart: React.FC<PopulationChartProps> = ({ history }) => {
  if (history.length < 2) return <p>Not enough data for chart yet.</p>;

  const width = 360;
  const height = 120;
  const maxPop = Math.max(...history.map(h => h.totalAgents), 1);
  const points = history.map((h, idx) => {
    const x = (idx / Math.max(history.length - 1, 1)) * (width - 20) + 10;
    const y = height - 10 - (h.totalAgents / maxPop) * (height - 20);
    return { x, y };
  });

  return (
    <svg width={width} height={height} style={{ background: "#0e1528", borderRadius: 8 }}>
      <polyline
        fill="none"
        stroke="#4caf50"
        strokeWidth={2}
        points={points.map(p => `${p.x},${p.y}`).join(" ")}
      />
      <line
        x1={10}
        y1={height - 10}
        x2={width - 10}
        y2={height - 10}
        stroke="#555"
        strokeWidth={1}
      />
      <line x1={10} y1={10} x2={10} y2={height - 10} stroke="#555" strokeWidth={1} />
      <text x={width - 60} y={height - 16} fontSize={10} fill="#ccc">
        ticks
      </text>
      <text x={14} y={18} fontSize={10} fill="#ccc">
        pop
      </text>
    </svg>
  );
};
