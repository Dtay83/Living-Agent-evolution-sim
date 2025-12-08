/**
 * Trait Distribution Chart Component
 * Displays bar chart of most common genetic traits
 */

import React from 'react';
import type { HistoryPoint } from '../types';
import { colorForTrait } from '../utils';

interface TraitChartProps {
  last: HistoryPoint | null;
  watchedTraitId: number | null;
  onSelectTrait: (traitId: number) => void;
}

export const TraitChart: React.FC<TraitChartProps> = ({ last, watchedTraitId, onSelectTrait }) => {
  if (!last) return <p>No trait data yet.</p>;
  const entries = Object.entries(last.byTrait);
  if (entries.length === 0) return <p>No agents alive.</p>;

  const sorted = entries.sort((a, b) => b[1] - a[1]).slice(0, 5);
  const maxCount = Math.max(...sorted.map(([, c]) => c), 1);

  return (
    <div style={{ display: "flex", alignItems: "flex-end", gap: 8, height: 120 }}>
      {sorted.map(([traitStr, count]) => {
        const traitId = Number(traitStr);
        const heightRatio = count / maxCount;
        const isWatched = watchedTraitId === traitId;
        return (
          <div
            key={traitStr}
            style={{ textAlign: "center", cursor: "pointer" }}
            onClick={() => onSelectTrait(traitId)}
            title={`Click to watch traitId ${traitId}`}
          >
            <div
              style={{
                width: 30,
                height: heightRatio * 90,
                background: colorForTrait(traitId),
                borderRadius: 4,
                marginBottom: 4,
                outline: isWatched ? "2px solid #ffeb3b" : "none"
              }}
            />
            <div style={{ fontSize: 9 }}>id {traitStr.slice(0, 4)}</div>
            <div style={{ fontSize: 9 }}>{count}</div>
          </div>
        );
      })}
    </div>
  );
};
