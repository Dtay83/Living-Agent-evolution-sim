/**
 * Physics Concepts System
 * Provides physics knowledge that aids agent learning and invention
 * UNLIMITED EVOLUTION - No caps on discovery or bonuses
 */

export interface PhysicsConcept {
  id: string;
  name: string;
  category: 'mechanics' | 'thermodynamics' | 'electromagnetism' | 'quantum' | 'relativity' | 'unified' | 'cosmology' | 'particle' | 'plasma' | 'condensed_matter' | 'exotic' | 'transcendent';
  complexity: number;         // 1-100+: How advanced this concept is (NO CAP)
  discoveredAt?: number;      // Tick when discovered
  discoveredBy?: number;      // Agent ID who discovered
  prerequisiteIds: string[];  // Required prior knowledge
  description: string;
  aiBonus: {
    energyEfficiency?: number;  // Multiplier for energy usage (lower is better)
    movementSpeed?: number;     // Multiplier for movement
    learningRate?: number;      // Multiplier for Q-learning alpha
    inventionChance?: number;   // Additive bonus to invention discovery
    spaceManipulation?: number; // NEW: Ability to affect grid space
    timePerception?: number;    // NEW: Extra ticks or preview ability
  };
}

/**
 * Progressive physics concepts from basic to transcendent
 * UNLIMITED system - new concepts generated procedurally beyond base set
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
  {
    id: 'rotational_dynamics',
    name: 'Rotational Dynamics',
    category: 'mechanics',
    complexity: 4,
    prerequisiteIds: ['momentum'],
    description: 'Angular momentum and torque',
    aiBonus: { movementSpeed: 1.15, energyEfficiency: 0.92 }
  },
  {
    id: 'fluid_dynamics',
    name: 'Fluid Dynamics',
    category: 'mechanics',
    complexity: 5,
    prerequisiteIds: ['rotational_dynamics'],
    description: 'Motion of liquids and gases',
    aiBonus: { movementSpeed: 1.20, energyEfficiency: 0.88 }
  },
  
  // THERMODYNAMICS (Level 3-6)
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
  {
    id: 'statistical_mechanics',
    name: 'Statistical Mechanics',
    category: 'thermodynamics',
    complexity: 6,
    prerequisiteIds: ['thermodynamic_efficiency'],
    description: 'Microscopic basis of thermodynamics',
    aiBonus: { learningRate: 1.18, energyEfficiency: 0.82 }
  },
  {
    id: 'phase_transitions',
    name: 'Phase Transitions',
    category: 'thermodynamics',
    complexity: 6,
    prerequisiteIds: ['statistical_mechanics'],
    description: 'Critical points and symmetry breaking',
    aiBonus: { inventionChance: 0.015, learningRate: 1.22 }
  },
  
  // ELECTROMAGNETISM (Level 5-8)
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
  {
    id: 'electromagnetic_waves',
    name: 'Electromagnetic Waves',
    category: 'electromagnetism',
    complexity: 7,
    prerequisiteIds: ['maxwells_equations'],
    description: 'Light, radio, and all EM radiation',
    aiBonus: { movementSpeed: 1.25, learningRate: 1.20, inventionChance: 0.018 }
  },
  {
    id: 'photonics',
    name: 'Photonics',
    category: 'electromagnetism',
    complexity: 8,
    prerequisiteIds: ['electromagnetic_waves'],
    description: 'Generation and control of photons',
    aiBonus: { learningRate: 1.30, inventionChance: 0.025, movementSpeed: 1.30 }
  },
  
  // QUANTUM MECHANICS (Level 7-10)
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
  {
    id: 'quantum_superposition',
    name: 'Quantum Superposition',
    category: 'quantum',
    complexity: 8,
    prerequisiteIds: ['wave_particle_duality'],
    description: 'Multiple states simultaneously',
    aiBonus: { learningRate: 1.45, inventionChance: 0.035, timePerception: 1.10 }
  },
  {
    id: 'quantum_tunneling',
    name: 'Quantum Tunneling',
    category: 'quantum',
    complexity: 8,
    prerequisiteIds: ['uncertainty_principle'],
    description: 'Particles pass through barriers',
    aiBonus: { movementSpeed: 1.40, spaceManipulation: 1.10, inventionChance: 0.032 }
  },
  {
    id: 'quantum_computing_theory',
    name: 'Quantum Computing',
    category: 'quantum',
    complexity: 10,
    prerequisiteIds: ['quantum_superposition', 'quantum_entanglement'],
    description: 'Computation using quantum states',
    aiBonus: { learningRate: 1.60, inventionChance: 0.055, timePerception: 1.20 }
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
    aiBonus: { energyEfficiency: 0.70, movementSpeed: 1.50, inventionChance: 0.045, spaceManipulation: 1.15 }
  },
  {
    id: 'gravitational_waves',
    name: 'Gravitational Waves',
    category: 'relativity',
    complexity: 9,
    prerequisiteIds: ['general_relativity'],
    description: 'Ripples in spacetime fabric',
    aiBonus: { spaceManipulation: 1.20, inventionChance: 0.048, learningRate: 1.45 }
  },
  {
    id: 'frame_dragging',
    name: 'Frame Dragging',
    category: 'relativity',
    complexity: 10,
    prerequisiteIds: ['gravitational_waves'],
    description: 'Rotating masses drag spacetime',
    aiBonus: { spaceManipulation: 1.25, timePerception: 1.15, movementSpeed: 1.55 }
  },
  
  // PARTICLE PHYSICS (Level 8-10) - NEW CATEGORY
  {
    id: 'standard_model',
    name: 'Standard Model',
    category: 'particle',
    complexity: 8,
    prerequisiteIds: ['quantum_entanglement'],
    description: 'Fundamental particles and forces',
    aiBonus: { learningRate: 1.50, inventionChance: 0.042 }
  },
  {
    id: 'higgs_mechanism',
    name: 'Higgs Mechanism',
    category: 'particle',
    complexity: 9,
    prerequisiteIds: ['standard_model'],
    description: 'Origin of mass',
    aiBonus: { energyEfficiency: 0.68, inventionChance: 0.050, learningRate: 1.55 }
  },
  {
    id: 'antimatter',
    name: 'Antimatter Physics',
    category: 'particle',
    complexity: 9,
    prerequisiteIds: ['standard_model'],
    description: 'Particles with opposite charge',
    aiBonus: { energyEfficiency: 0.50, inventionChance: 0.055 }
  },
  {
    id: 'neutrino_physics',
    name: 'Neutrino Physics',
    category: 'particle',
    complexity: 10,
    prerequisiteIds: ['higgs_mechanism'],
    description: 'Ghost particles that rarely interact',
    aiBonus: { learningRate: 1.60, spaceManipulation: 1.18, inventionChance: 0.052 }
  },
  
  // PLASMA PHYSICS (Level 7-9) - NEW CATEGORY
  {
    id: 'plasma_basics',
    name: 'Plasma Physics',
    category: 'plasma',
    complexity: 7,
    prerequisiteIds: ['electromagnetic_induction', 'statistical_mechanics'],
    description: 'Fourth state of matter',
    aiBonus: { energyEfficiency: 0.78, inventionChance: 0.022 }
  },
  {
    id: 'fusion',
    name: 'Nuclear Fusion',
    category: 'plasma',
    complexity: 9,
    prerequisiteIds: ['plasma_basics'],
    description: 'Combining atomic nuclei',
    aiBonus: { energyEfficiency: 0.40, inventionChance: 0.060, learningRate: 1.55 }
  },
  {
    id: 'magnetohydrodynamics',
    name: 'Magnetohydrodynamics',
    category: 'plasma',
    complexity: 8,
    prerequisiteIds: ['plasma_basics', 'fluid_dynamics'],
    description: 'Magnetic fluid dynamics',
    aiBonus: { movementSpeed: 1.35, energyEfficiency: 0.72, inventionChance: 0.028 }
  },
  
  // CONDENSED MATTER (Level 7-10) - NEW CATEGORY
  {
    id: 'solid_state',
    name: 'Solid State Physics',
    category: 'condensed_matter',
    complexity: 7,
    prerequisiteIds: ['quantum_superposition'],
    description: 'Physics of solids',
    aiBonus: { inventionChance: 0.025, learningRate: 1.28 }
  },
  {
    id: 'superconductivity',
    name: 'Superconductivity',
    category: 'condensed_matter',
    complexity: 9,
    prerequisiteIds: ['solid_state'],
    description: 'Zero electrical resistance',
    aiBonus: { energyEfficiency: 0.35, movementSpeed: 1.50, inventionChance: 0.058 }
  },
  {
    id: 'topological_matter',
    name: 'Topological Matter',
    category: 'condensed_matter',
    complexity: 10,
    prerequisiteIds: ['superconductivity'],
    description: 'Matter with topological properties',
    aiBonus: { inventionChance: 0.065, learningRate: 1.65, spaceManipulation: 1.22 }
  },
  
  // COSMOLOGY (Level 9-11) - NEW CATEGORY
  {
    id: 'cosmological_model',
    name: 'Cosmological Model',
    category: 'cosmology',
    complexity: 9,
    prerequisiteIds: ['general_relativity'],
    description: 'Structure and evolution of universe',
    aiBonus: { spaceManipulation: 1.25, timePerception: 1.20, inventionChance: 0.048 }
  },
  {
    id: 'dark_matter',
    name: 'Dark Matter',
    category: 'cosmology',
    complexity: 10,
    prerequisiteIds: ['cosmological_model'],
    description: 'Invisible mass pervading the cosmos',
    aiBonus: { spaceManipulation: 1.30, inventionChance: 0.058 }
  },
  {
    id: 'dark_energy',
    name: 'Dark Energy',
    category: 'cosmology',
    complexity: 10,
    prerequisiteIds: ['dark_matter'],
    description: 'Accelerating expansion of universe',
    aiBonus: { energyEfficiency: 0.30, spaceManipulation: 1.35, inventionChance: 0.062 }
  },
  {
    id: 'multiverse_theory',
    name: 'Multiverse Theory',
    category: 'cosmology',
    complexity: 11,
    prerequisiteIds: ['dark_energy'],
    description: 'Infinite parallel universes',
    aiBonus: { spaceManipulation: 1.50, timePerception: 1.35, inventionChance: 0.070 }
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
  },
  {
    id: 'm_theory',
    name: 'M-Theory',
    category: 'unified',
    complexity: 11,
    prerequisiteIds: ['string_theory'],
    description: 'Unification of string theories in 11 dimensions',
    aiBonus: { spaceManipulation: 1.40, timePerception: 1.30, inventionChance: 0.072, learningRate: 1.75 }
  },
  {
    id: 'loop_quantum_gravity',
    name: 'Loop Quantum Gravity',
    category: 'unified',
    complexity: 11,
    prerequisiteIds: ['general_relativity', 'quantum_field_theory'],
    description: 'Quantized spacetime',
    aiBonus: { spaceManipulation: 1.45, inventionChance: 0.075, energyEfficiency: 0.50 }
  },
  
  // EXOTIC PHYSICS (Level 11-12) - NEW CATEGORY
  {
    id: 'wormhole_physics',
    name: 'Wormhole Physics',
    category: 'exotic',
    complexity: 11,
    prerequisiteIds: ['loop_quantum_gravity'],
    description: 'Shortcuts through spacetime',
    aiBonus: { spaceManipulation: 1.60, movementSpeed: 2.0, inventionChance: 0.080 }
  },
  {
    id: 'negative_energy',
    name: 'Negative Energy States',
    category: 'exotic',
    complexity: 12,
    prerequisiteIds: ['wormhole_physics', 'dark_energy'],
    description: 'Exotic matter with negative mass-energy',
    aiBonus: { energyEfficiency: 0.20, spaceManipulation: 1.70, inventionChance: 0.085 }
  },
  {
    id: 'time_manipulation',
    name: 'Temporal Mechanics',
    category: 'exotic',
    complexity: 12,
    prerequisiteIds: ['wormhole_physics'],
    description: 'Theoretical time control',
    aiBonus: { timePerception: 1.80, spaceManipulation: 1.65, inventionChance: 0.088 }
  },
  
  // TRANSCENDENT PHYSICS (Level 12+) - NEW CATEGORY
  {
    id: 'reality_engineering',
    name: 'Reality Engineering',
    category: 'transcendent',
    complexity: 13,
    prerequisiteIds: ['negative_energy', 'time_manipulation'],
    description: 'Direct manipulation of physical law',
    aiBonus: { spaceManipulation: 2.0, timePerception: 2.0, energyEfficiency: 0.10, inventionChance: 0.10 }
  },
  {
    id: 'omega_physics',
    name: 'Omega Point Physics',
    category: 'transcendent',
    complexity: 15,
    prerequisiteIds: ['reality_engineering', 'multiverse_theory'],
    description: 'Physics at the end of time',
    aiBonus: { 
      energyEfficiency: 0.05, 
      movementSpeed: 2.5, 
      learningRate: 2.5, 
      inventionChance: 0.15,
      spaceManipulation: 2.5,
      timePerception: 2.5
    }
  }
];

// Track procedurally generated concepts for unlimited evolution
let generatedPhysicsLevel = 0;

/**
 * Generate advanced physics concepts procedurally
 * Enables UNLIMITED scientific progression - NO CAPS
 */
export function generateAdvancedPhysicsConcept(level: number): PhysicsConcept {
  const categories: PhysicsConcept['category'][] = ['mechanics', 'thermodynamics', 'electromagnetism', 'quantum', 'relativity', 'unified', 'cosmology', 'particle', 'plasma', 'condensed_matter', 'exotic', 'transcendent'];
  const category = categories[level % categories.length];
  
  // Complexity scales infinitely
  const complexity = 15 + level;
  
  generatedPhysicsLevel = Math.max(generatedPhysicsLevel, level);
  
  // Bonuses grow without limit
  const baseEfficiency = Math.max(0.01, 0.05 - (level * 0.002)); // Approaches but never reaches 0
  const baseMultiplier = 2.5 + (level * 0.15);
  
  return {
    id: `advanced_physics_${level}`,
    name: `${getCategoryPrefix(category)} ${getRomanNumeral(level)}`,
    category,
    complexity,
    prerequisiteIds: level > 0 ? [`advanced_physics_${level - 1}`] : ['omega_physics'],
    description: `${category.charAt(0).toUpperCase() + category.slice(1).replace('_', ' ')} beyond current understanding - Level ${level}`,
    aiBonus: {
      energyEfficiency: baseEfficiency,
      learningRate: baseMultiplier,
      movementSpeed: baseMultiplier,
      inventionChance: 0.15 + (level * 0.01),
      spaceManipulation: baseMultiplier,
      timePerception: baseMultiplier
    }
  };
}

function getCategoryPrefix(category: string): string {
  const prefixes: Record<string, string> = {
    'mechanics': 'Hypermechanics',
    'thermodynamics': 'Entropic Mastery',
    'electromagnetism': 'Field Manipulation',
    'quantum': 'Quantum Singularity',
    'relativity': 'Spacetime Mastery',
    'unified': 'Grand Unification',
    'cosmology': 'Cosmic Engineering',
    'particle': 'Subatomic Control',
    'plasma': 'Plasma Dominion',
    'condensed_matter': 'Matter Mastery',
    'exotic': 'Exotic Manipulation',
    'transcendent': 'Transcendent Physics'
  };
  return prefixes[category] || 'Advanced Physics';
}

function getRomanNumeral(num: number): string {
  if (num <= 0) return 'I';
  if (num > 100) return `${num}`;
  const numerals = ['', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X',
    'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI', 'XVII', 'XVIII', 'XIX', 'XX'];
  return num < numerals.length ? numerals[num] : `${num}`;
}

/**
 * Calculate total physics bonuses for an agent
 * NO CAPS - bonuses can grow infinitely
 */
export function calculatePhysicsBonuses(unlockedPhysics: PhysicsConcept[]): {
  energyEfficiency: number;
  movementSpeed: number;
  learningRate: number;
  inventionChance: number;
  spaceManipulation: number;
  timePerception: number;
} {
  let energyMult = 1.0;
  let movementMult = 1.0;
  let learningMult = 1.0;
  let inventionBonus = 0.0;
  let spaceMult = 1.0;
  let timeMult = 1.0;
  
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
    if (concept.aiBonus.spaceManipulation) {
      spaceMult *= concept.aiBonus.spaceManipulation;
    }
    if (concept.aiBonus.timePerception) {
      timeMult *= concept.aiBonus.timePerception;
    }
  }
  
  // NO CAPS - unlimited evolution!
  return {
    energyEfficiency: energyMult,
    movementSpeed: movementMult,
    learningRate: learningMult,
    inventionChance: inventionBonus, // REMOVED CAP
    spaceManipulation: spaceMult,
    timePerception: timeMult
  };
}

/**
 * Get all available physics concepts including procedurally generated ones
 */
export function getAllPhysicsConcepts(generatedLevels: number = 0): PhysicsConcept[] {
  const baseConcepts = [...PHYSICS_CONCEPTS];
  for (let i = 0; i < generatedLevels; i++) {
    baseConcepts.push(generateAdvancedPhysicsConcept(i));
  }
  return baseConcepts;
}

/**
 * Get current generated physics level for saving state
 */
export function getGeneratedPhysicsLevel(): number {
  return generatedPhysicsLevel;
}

/**
 * Set generated physics level when loading state
 */
export function setGeneratedPhysicsLevel(level: number): void {
  generatedPhysicsLevel = level;
}
