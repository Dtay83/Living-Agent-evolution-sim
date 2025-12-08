/**
 * Physics Concepts System
 * Provides physics knowledge that aids agent learning and invention
 */

export interface PhysicsConcept {
  id: string;
  name: string;
  category: 'mechanics' | 'thermodynamics' | 'electromagnetism' | 'quantum' | 'relativity' | 'unified';
  complexity: number;         // 1-10: How advanced this concept is
  discoveredAt?: number;      // Tick when discovered
  discoveredBy?: number;      // Agent ID who discovered
  prerequisiteIds: string[];  // Required prior knowledge
  description: string;
  aiBonus: {
    energyEfficiency?: number;  // Multiplier for energy usage
    movementSpeed?: number;     // Multiplier for movement
    learningRate?: number;      // Multiplier for Q-learning alpha
    inventionChance?: number;   // Additive bonus to invention discovery
  };
}

/**
 * Progressive physics concepts from basic to advanced
 * Unlimited system - new concepts can be generated procedurally
 */
export const PHYSICS_CONCEPTS: PhysicsConcept[] = [
  // MECHANICS (Level 1-3)
  {
    id: 'basic_motion',
    name: 'Basic Motion',
    category: 'mechanics',
    complexity: 1,
    prerequisiteIds: [],
    description: 'Objects move in predictable paths',
    aiBonus: { movementSpeed: 1.05 }
  },
  {
    id: 'leverage',
    name: 'Leverage & Simple Machines',
    category: 'mechanics',
    complexity: 2,
    prerequisiteIds: ['basic_motion'],
    description: 'Force multiplication through mechanical advantage',
    aiBonus: { energyEfficiency: 0.95 }
  },
  {
    id: 'inertia',
    name: 'Inertia',
    category: 'mechanics',
    complexity: 2,
    prerequisiteIds: ['basic_motion'],
    description: 'Objects resist changes in motion',
    aiBonus: { movementSpeed: 1.08 }
  },
  {
    id: 'momentum',
    name: 'Momentum Conservation',
    category: 'mechanics',
    complexity: 3,
    prerequisiteIds: ['inertia'],
    description: 'Momentum is conserved in interactions',
    aiBonus: { energyEfficiency: 0.93, movementSpeed: 1.10 }
  },
  {
    id: 'energy_conservation',
    name: 'Energy Conservation',
    category: 'mechanics',
    complexity: 3,
    prerequisiteIds: ['leverage'],
    description: 'Energy cannot be created or destroyed',
    aiBonus: { energyEfficiency: 0.90, inventionChance: 0.005 }
  },
  
  // THERMODYNAMICS (Level 3-5)
  {
    id: 'heat_transfer',
    name: 'Heat Transfer',
    category: 'thermodynamics',
    complexity: 3,
    prerequisiteIds: ['energy_conservation'],
    description: 'Energy flows from hot to cold',
    aiBonus: { energyEfficiency: 0.92 }
  },
  {
    id: 'entropy',
    name: 'Entropy',
    category: 'thermodynamics',
    complexity: 5,
    prerequisiteIds: ['heat_transfer'],
    description: 'Disorder tends to increase',
    aiBonus: { learningRate: 1.10, inventionChance: 0.008 }
  },
  {
    id: 'thermodynamic_efficiency',
    name: 'Thermodynamic Efficiency',
    category: 'thermodynamics',
    complexity: 5,
    prerequisiteIds: ['entropy'],
    description: 'Maximum efficiency limits',
    aiBonus: { energyEfficiency: 0.85, inventionChance: 0.010 }
  },
  
  // ELECTROMAGNETISM (Level 5-7)
  {
    id: 'electricity',
    name: 'Electricity',
    category: 'electromagnetism',
    complexity: 5,
    prerequisiteIds: ['energy_conservation'],
    description: 'Charge and electric forces',
    aiBonus: { learningRate: 1.15, inventionChance: 0.012 }
  },
  {
    id: 'magnetism',
    name: 'Magnetism',
    category: 'electromagnetism',
    complexity: 5,
    prerequisiteIds: ['basic_motion'],
    description: 'Magnetic forces and fields',
    aiBonus: { movementSpeed: 1.15 }
  },
  {
    id: 'electromagnetic_induction',
    name: 'Electromagnetic Induction',
    category: 'electromagnetism',
    complexity: 6,
    prerequisiteIds: ['electricity', 'magnetism'],
    description: 'Changing magnetic fields create electricity',
    aiBonus: { energyEfficiency: 0.80, inventionChance: 0.015 }
  },
  {
    id: 'maxwells_equations',
    name: "Maxwell's Equations",
    category: 'electromagnetism',
    complexity: 7,
    prerequisiteIds: ['electromagnetic_induction'],
    description: 'Unified electromagnetic theory',
    aiBonus: { learningRate: 1.25, inventionChance: 0.020 }
  },
  
  // QUANTUM MECHANICS (Level 7-9)
  {
    id: 'wave_particle_duality',
    name: 'Wave-Particle Duality',
    category: 'quantum',
    complexity: 7,
    prerequisiteIds: ['maxwells_equations'],
    description: 'Matter exhibits wave and particle properties',
    aiBonus: { learningRate: 1.30, inventionChance: 0.025 }
  },
  {
    id: 'uncertainty_principle',
    name: 'Uncertainty Principle',
    category: 'quantum',
    complexity: 8,
    prerequisiteIds: ['wave_particle_duality'],
    description: 'Fundamental limits to measurement',
    aiBonus: { learningRate: 1.40, inventionChance: 0.030 }
  },
  {
    id: 'quantum_entanglement',
    name: 'Quantum Entanglement',
    category: 'quantum',
    complexity: 9,
    prerequisiteIds: ['uncertainty_principle'],
    description: 'Non-local quantum correlations',
    aiBonus: { learningRate: 1.50, movementSpeed: 1.30, inventionChance: 0.040 }
  },
  
  // RELATIVITY (Level 8-10)
  {
    id: 'special_relativity',
    name: 'Special Relativity',
    category: 'relativity',
    complexity: 8,
    prerequisiteIds: ['maxwells_equations', 'momentum'],
    description: 'Space and time are relative',
    aiBonus: { movementSpeed: 1.40, learningRate: 1.35, inventionChance: 0.035 }
  },
  {
    id: 'general_relativity',
    name: 'General Relativity',
    category: 'relativity',
    complexity: 9,
    prerequisiteIds: ['special_relativity'],
    description: 'Gravity is curved spacetime',
    aiBonus: { energyEfficiency: 0.70, movementSpeed: 1.50, inventionChance: 0.045 }
  },
  
  // UNIFIED THEORIES (Level 10+)
  {
    id: 'quantum_field_theory',
    name: 'Quantum Field Theory',
    category: 'unified',
    complexity: 10,
    prerequisiteIds: ['quantum_entanglement', 'special_relativity'],
    description: 'Quantum mechanics meets special relativity',
    aiBonus: { energyEfficiency: 0.65, learningRate: 1.60, inventionChance: 0.050 }
  },
  {
    id: 'string_theory',
    name: 'String Theory',
    category: 'unified',
    complexity: 10,
    prerequisiteIds: ['quantum_field_theory', 'general_relativity'],
    description: 'Fundamental strings in higher dimensions',
    aiBonus: { energyEfficiency: 0.60, learningRate: 1.70, movementSpeed: 1.60, inventionChance: 0.060 }
  }
];

/**
 * Generate advanced physics concepts procedurally
 * Enables unlimited scientific progression
 */
export function generateAdvancedPhysicsConcept(level: number): PhysicsConcept {
  const categories: PhysicsConcept['category'][] = ['mechanics', 'thermodynamics', 'electromagnetism', 'quantum', 'relativity', 'unified'];
  const category = categories[level % categories.length];
  
  const complexity = Math.min(10 + Math.floor(level / 10), 100);
  
  return {
    id: `advanced_physics_${level}`,
    name: `Advanced ${category.charAt(0).toUpperCase() + category.slice(1)} ${level}`,
    category,
    complexity,
    prerequisiteIds: level > 0 ? [`advanced_physics_${level - 1}`] : ['string_theory'],
    description: `Cutting-edge ${category} research level ${level}`,
    aiBonus: {
      energyEfficiency: Math.max(0.1, 0.60 - (level * 0.02)),
      learningRate: 1.70 + (level * 0.05),
      movementSpeed: 1.60 + (level * 0.03),
      inventionChance: 0.060 + (level * 0.005)
    }
  };
}

/**
 * Calculate total physics bonuses for an agent
 */
export function calculatePhysicsBonuses(unlockedPhysics: PhysicsConcept[]): {
  energyEfficiency: number;
  movementSpeed: number;
  learningRate: number;
  inventionChance: number;
} {
  let energyMult = 1.0;
  let movementMult = 1.0;
  let learningMult = 1.0;
  let inventionBonus = 0.0;
  
  for (const concept of unlockedPhysics) {
    if (concept.aiBonus.energyEfficiency) {
      energyMult *= concept.aiBonus.energyEfficiency;
    }
    if (concept.aiBonus.movementSpeed) {
      movementMult *= concept.aiBonus.movementSpeed;
    }
    if (concept.aiBonus.learningRate) {
      learningMult *= concept.aiBonus.learningRate;
    }
    if (concept.aiBonus.inventionChance) {
      inventionBonus += concept.aiBonus.inventionChance;
    }
  }
  
  return {
    energyEfficiency: energyMult,
    movementSpeed: movementMult,
    learningRate: learningMult,
    inventionChance: Math.min(inventionBonus, 0.5) // Cap at 50% bonus
  };
}
