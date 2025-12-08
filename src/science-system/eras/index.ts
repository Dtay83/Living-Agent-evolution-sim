/**
 * Scientific Era System
 * Tracks civilization progress through unlimited scientific ages
 */

export interface ScientificEra {
  id: string;
  name: string;
  level: number;              // Era progression level (0 = Stone Age, 1+ = unlimited)
  startTick: number;          // When this era began
  discoveries: string[];      // IDs of discoveries made in this era
  physicsUnlocked: string[];  // Physics concepts unlocked
  mathUnlocked: string[];     // Math concepts unlocked
  description: string;
  requirements: {
    minDiscoveries: number;   // Total discoveries needed to advance
    minPhysicsConcepts: number;
    minMathConcepts: number;
  };
}

export interface ScientificDiscovery {
  id: string;
  name: string;
  category: 'physics' | 'mathematics' | 'invention' | 'evolution';
  discoveredAt: number;       // Tick when discovered
  discoveredBy: number;       // Agent ID
  eraLevel: number;           // Era when discovered
  significance: number;       // 0-1: How important this discovery is
  description: string;
  prerequisiteIds: string[];  // Other discoveries needed first
  enablesIds: string[];       // What this discovery unlocks
}

export interface ProgressionMetrics {
  totalDiscoveries: number;
  discoveriesPerEra: Record<number, number>;
  averageDiscoveryRate: number;  // Discoveries per tick
  currentEraLevel: number;
  ticksSinceLastEra: number;
  scientificAcceleration: number; // Rate of acceleration in discoveries
}

/**
 * Base eras that unlock progressively
 * System supports unlimited era generation beyond these
 */
export const BASE_ERAS: Omit<ScientificEra, 'startTick' | 'discoveries' | 'physicsUnlocked' | 'mathUnlocked'>[] = [
  {
    id: 'stone_age',
    name: 'Stone Age',
    level: 0,
    description: 'Basic survival and tool use',
    requirements: { minDiscoveries: 0, minPhysicsConcepts: 0, minMathConcepts: 0 }
  },
  {
    id: 'bronze_age',
    name: 'Bronze Age',
    level: 1,
    description: 'Material manipulation and basic mechanics',
    requirements: { minDiscoveries: 10, minPhysicsConcepts: 3, minMathConcepts: 2 }
  },
  {
    id: 'iron_age',
    name: 'Iron Age',
    level: 2,
    description: 'Advanced materials and early mathematics',
    requirements: { minDiscoveries: 25, minPhysicsConcepts: 6, minMathConcepts: 5 }
  },
  {
    id: 'classical_age',
    name: 'Classical Age',
    level: 3,
    description: 'Scientific method and systematic thinking',
    requirements: { minDiscoveries: 50, minPhysicsConcepts: 10, minMathConcepts: 10 }
  },
  {
    id: 'renaissance',
    name: 'Renaissance',
    level: 4,
    description: 'Empirical science and mathematical formalization',
    requirements: { minDiscoveries: 100, minPhysicsConcepts: 15, minMathConcepts: 15 }
  },
  {
    id: 'industrial_age',
    name: 'Industrial Age',
    level: 5,
    description: 'Applied physics and engineering mathematics',
    requirements: { minDiscoveries: 200, minPhysicsConcepts: 25, minMathConcepts: 25 }
  },
  {
    id: 'information_age',
    name: 'Information Age',
    level: 6,
    description: 'Computational theory and quantum mechanics',
    requirements: { minDiscoveries: 400, minPhysicsConcepts: 40, minMathConcepts: 40 }
  },
  {
    id: 'quantum_age',
    name: 'Quantum Age',
    level: 7,
    description: 'Quantum physics and abstract mathematics',
    requirements: { minDiscoveries: 800, minPhysicsConcepts: 60, minMathConcepts: 60 }
  },
  {
    id: 'singularity_age',
    name: 'Singularity Age',
    level: 8,
    description: 'Post-scarcity science and theoretical breakthroughs',
    requirements: { minDiscoveries: 1600, minPhysicsConcepts: 100, minMathConcepts: 100 }
  }
];

/**
 * Generate a new era when civilization advances beyond base eras
 * Supports unlimited progression
 */
export function generateNextEra(currentLevel: number): Omit<ScientificEra, 'startTick' | 'discoveries' | 'physicsUnlocked' | 'mathUnlocked'> {
  const eraNames = [
    'Transcendence Age', 'Cosmic Age', 'Multiversal Age', 'Dimensional Age',
    'Entropy Age', 'Hyperdimensional Age', 'Omniscient Age', 'Ultimate Age'
  ];
  
  const baseIndex = currentLevel - BASE_ERAS.length;
  const cycleName = baseIndex < eraNames.length 
    ? eraNames[baseIndex]
    : `Advanced Era ${currentLevel}`;
  
  // Exponential scaling for requirements
  const scaleFactor = Math.pow(2, currentLevel - 8);
  
  return {
    id: `era_${currentLevel}`,
    name: cycleName,
    level: currentLevel,
    description: `Advanced civilization level ${currentLevel} - Unlimited potential`,
    requirements: {
      minDiscoveries: Math.floor(1600 * scaleFactor),
      minPhysicsConcepts: Math.floor(100 * scaleFactor),
      minMathConcepts: Math.floor(100 * scaleFactor)
    }
  };
}

/**
 * Check if civilization should advance to next era
 */
export function shouldAdvanceEra(
  currentEra: ScientificEra,
  metrics: ProgressionMetrics,
  physicsCount: number,
  mathCount: number
): boolean {
  const nextEraRequirements = currentEra.requirements;
  
  return (
    metrics.totalDiscoveries >= nextEraRequirements.minDiscoveries &&
    physicsCount >= nextEraRequirements.minPhysicsConcepts &&
    mathCount >= nextEraRequirements.minMathConcepts
  );
}
