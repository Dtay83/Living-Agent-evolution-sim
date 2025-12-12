/**
 * Scientific Era System - EXPANDED EDITION
 * Tracks civilization progress through unlimited scientific ages
 * 
 * Features:
 * - 20 base eras from Stone Age to Omega Point
 * - Era-specific discovery modifiers and bonuses
 * - Collaboration and momentum effects per era
 * - Research focus areas that shift with era progression
 * - Cultural and societal characteristics
 * - Support for unlimited procedural era generation
 */

import type { PhysicsConcept } from '../physics';
import type { MathConcept } from '../mathematics';

/**
 * Research focus areas that civilizations can prioritize
 */
export type ResearchFocus = 
  | 'fundamental'      // Basic research, curiosity-driven
  | 'applied'          // Practical applications
  | 'theoretical'      // Abstract mathematics and physics
  | 'collaborative'    // Multi-agent research
  | 'experimental'     // Trial-and-error discovery
  | 'computational'    // Computer-aided discovery
  | 'unified'          // Cross-disciplinary synthesis
  | 'transcendent';    // Beyond conventional understanding

/**
 * Era characteristics that affect civilization behavior
 */
export interface EraCharacteristics {
  researchFocus: ResearchFocus[];
  collaborationLevel: 'isolated' | 'tribal' | 'regional' | 'global' | 'universal' | 'omniscient';
  knowledgeTransfer: number;      // 0-1: How easily knowledge spreads
  innovationRate: number;         // Multiplier on base discovery chance
  specialization: number;         // 0-1: Tendency toward deep vs broad knowledge
  riskTolerance: number;          // 0-1: Willingness to pursue risky research
  resourceEfficiency: number;     // Multiplier on energy costs for research
}

/**
 * Era-specific bonuses that apply to all agents
 */
export interface EraBonus {
  // Physics bonuses
  energyEfficiency?: number;
  movementSpeed?: number;
  learningRate?: number;
  inventionChance?: number;
  spaceManipulation?: number;
  timePerception?: number;
  curiosity?: number;
  cooperationBonus?: number;
  dimensionalAccess?: number;
  informationProcessing?: number;
  entropicResistance?: number;
  // Math bonuses
  decisionQuality?: number;
  explorationBonus?: number;
  patternRecognition?: number;
  optimizationPower?: number;
  computationalSpeed?: number;
  abstractionLevel?: number;
  proofIntuition?: number;
  abstractThinking?: number;
}

/**
 * Categories that define era advancement priorities
 */
export interface EraUnlockCategories {
  physics: string[];    // Physics categories that should be prioritized
  math: string[];       // Math categories that should be prioritized
}

/**
 * Extended Scientific Era definition
 */
export interface ScientificEra {
  id: string;
  name: string;
  level: number;              // Era progression level (0 = Stone Age, unlimited)
  startTick: number;          // When this era began
  discoveries: string[];      // IDs of discoveries made in this era
  physicsUnlocked: string[];  // Physics concepts unlocked
  mathUnlocked: string[];     // Math concepts unlocked
  description: string;
  flavorText?: string;        // Narrative description of the era
  requirements: {
    minDiscoveries: number;   // Total discoveries needed to advance
    minPhysicsConcepts: number;
    minMathConcepts: number;
    minEraTime?: number;      // Optional minimum ticks in this era
    requiredCategories?: string[]; // Categories that must have discoveries
  };
  characteristics: EraCharacteristics;
  eraBonus: EraBonus;
  unlockPriorities: EraUnlockCategories;
  // Milestones specific to this era
  eraMilestones?: {
    id: string;
    name: string;
    description: string;
    requirement: number;
    achieved?: boolean;
  }[];
}

/**
 * Scientific Discovery with extended tracking
 */
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
  // Extended tracking
  wasCollaborative?: boolean;
  collaboratorCount?: number;
  momentumAtDiscovery?: number;
  breakthroughType?: 'normal' | 'category_first' | 'era_defining' | 'transcendent';
}

/**
 * Progression metrics with extended tracking
 */
export interface ProgressionMetrics {
  totalDiscoveries: number;
  discoveriesPerEra: Record<number, number>;
  averageDiscoveryRate: number;  // Discoveries per tick
  currentEraLevel: number;
  ticksSinceLastEra: number;
  scientificAcceleration: number; // Rate of acceleration in discoveries
  // Extended metrics
  collaborativeDiscoveries: number;
  categoryBreakthroughs: number;
  peakMomentum: number;
  averageComplexity: number;
  physicsCategories: number;
  mathCategories: number;
  transcendentDiscoveries: number;
}

/**
 * Default era characteristics for generation
 */
const DEFAULT_ERA_CHARACTERISTICS: EraCharacteristics = {
  researchFocus: ['fundamental'],
  collaborationLevel: 'isolated',
  knowledgeTransfer: 0.1,
  innovationRate: 1.0,
  specialization: 0.5,
  riskTolerance: 0.3,
  resourceEfficiency: 1.0
};

/**
 * Base eras that unlock progressively (0-19)
 * System supports unlimited era generation beyond these
 */
export const BASE_ERAS: Omit<ScientificEra, 'startTick' | 'discoveries' | 'physicsUnlocked' | 'mathUnlocked'>[] = [
  // ============================================================================
  // PREHISTORIC ERAS (0-2)
  // ============================================================================
  {
    id: 'stone_age',
    name: 'Stone Age',
    level: 0,
    description: 'Basic survival and tool use',
    flavorText: 'Fire flickers in the darkness. The first questions are asked.',
    requirements: { 
      minDiscoveries: 0, 
      minPhysicsConcepts: 0, 
      minMathConcepts: 0 
    },
    characteristics: {
      researchFocus: ['experimental'],
      collaborationLevel: 'isolated',
      knowledgeTransfer: 0.05,
      innovationRate: 0.5,
      specialization: 0.1,
      riskTolerance: 0.2,
      resourceEfficiency: 0.5
    },
    eraBonus: {
      curiosity: 1.05,
      learningRate: 0.9
    },
    unlockPriorities: {
      physics: ['mechanics'],
      math: ['arithmetic']
    },
    eraMilestones: [
      { id: 'first_tool', name: 'First Tool', description: 'Create first tool', requirement: 1 },
      { id: 'first_fire', name: 'Fire Mastery', description: 'Understand heat', requirement: 3 }
    ]
  },
  {
    id: 'bronze_age',
    name: 'Bronze Age',
    level: 1,
    description: 'Material manipulation and basic mechanics',
    flavorText: 'Metal bends to will. Civilization takes root.',
    requirements: { 
      minDiscoveries: 8, 
      minPhysicsConcepts: 3, 
      minMathConcepts: 2 
    },
    characteristics: {
      researchFocus: ['experimental', 'applied'],
      collaborationLevel: 'tribal',
      knowledgeTransfer: 0.10,
      innovationRate: 0.7,
      specialization: 0.2,
      riskTolerance: 0.25,
      resourceEfficiency: 0.6
    },
    eraBonus: {
      curiosity: 1.08,
      learningRate: 0.95,
      cooperationBonus: 1.05
    },
    unlockPriorities: {
      physics: ['mechanics', 'thermodynamics'],
      math: ['arithmetic', 'geometry']
    },
    eraMilestones: [
      { id: 'metal_working', name: 'Metalworking', description: 'Work with metals', requirement: 5 },
      { id: 'basic_trade', name: 'Trade Routes', description: 'Establish knowledge sharing', requirement: 8 }
    ]
  },
  {
    id: 'iron_age',
    name: 'Iron Age',
    level: 2,
    description: 'Advanced materials and early mathematics',
    flavorText: 'Stronger metals, stronger ideas. Writing preserves knowledge.',
    requirements: { 
      minDiscoveries: 20, 
      minPhysicsConcepts: 6, 
      minMathConcepts: 5 
    },
    characteristics: {
      researchFocus: ['experimental', 'applied', 'fundamental'],
      collaborationLevel: 'regional',
      knowledgeTransfer: 0.15,
      innovationRate: 0.85,
      specialization: 0.3,
      riskTolerance: 0.3,
      resourceEfficiency: 0.7
    },
    eraBonus: {
      curiosity: 1.10,
      learningRate: 1.0,
      cooperationBonus: 1.08,
      patternRecognition: 1.05
    },
    unlockPriorities: {
      physics: ['mechanics', 'thermodynamics'],
      math: ['arithmetic', 'geometry', 'algebra']
    },
    eraMilestones: [
      { id: 'written_knowledge', name: 'Written Records', description: 'Preserve knowledge in writing', requirement: 12 },
      { id: 'iron_mastery', name: 'Iron Mastery', description: 'Advanced metalworking', requirement: 18 }
    ]
  },

  // ============================================================================
  // CLASSICAL ERAS (3-4)
  // ============================================================================
  {
    id: 'classical_age',
    name: 'Classical Age',
    level: 3,
    description: 'Scientific method and systematic thinking',
    flavorText: 'Philosophy awakens. The universe becomes a question to be answered.',
    requirements: { 
      minDiscoveries: 40, 
      minPhysicsConcepts: 10, 
      minMathConcepts: 10 
    },
    characteristics: {
      researchFocus: ['fundamental', 'theoretical'],
      collaborationLevel: 'regional',
      knowledgeTransfer: 0.25,
      innovationRate: 1.0,
      specialization: 0.4,
      riskTolerance: 0.35,
      resourceEfficiency: 0.8
    },
    eraBonus: {
      curiosity: 1.15,
      learningRate: 1.05,
      cooperationBonus: 1.10,
      patternRecognition: 1.10,
      proofIntuition: 1.05,
      abstractThinking: 1.05
    },
    unlockPriorities: {
      physics: ['mechanics', 'thermodynamics', 'electromagnetism'],
      math: ['geometry', 'algebra', 'number_theory']
    },
    eraMilestones: [
      { id: 'scientific_method', name: 'Scientific Method', description: 'Systematic inquiry established', requirement: 25 },
      { id: 'mathematical_proof', name: 'Mathematical Proof', description: 'Formal proofs developed', requirement: 35 }
    ]
  },
  {
    id: 'medieval_age',
    name: 'Medieval Age',
    level: 4,
    description: 'Preservation and gradual advancement',
    flavorText: 'Knowledge flickers but does not die. Scholars preserve what was learned.',
    requirements: { 
      minDiscoveries: 70, 
      minPhysicsConcepts: 15, 
      minMathConcepts: 15 
    },
    characteristics: {
      researchFocus: ['applied', 'fundamental'],
      collaborationLevel: 'regional',
      knowledgeTransfer: 0.20,
      innovationRate: 0.9,
      specialization: 0.45,
      riskTolerance: 0.25,
      resourceEfficiency: 0.75
    },
    eraBonus: {
      curiosity: 1.12,
      learningRate: 1.0,
      cooperationBonus: 1.12,
      entropicResistance: 1.05,
      patternRecognition: 1.12
    },
    unlockPriorities: {
      physics: ['mechanics', 'thermodynamics'],
      math: ['algebra', 'arithmetic']
    },
    eraMilestones: [
      { id: 'university', name: 'Universities', description: 'Centers of learning established', requirement: 50 },
      { id: 'algebra_advancement', name: 'Algebraic Methods', description: 'Algebra advances significantly', requirement: 65 }
    ]
  },

  // ============================================================================
  // EARLY MODERN ERAS (5-6)
  // ============================================================================
  {
    id: 'renaissance',
    name: 'Renaissance',
    level: 5,
    description: 'Empirical science and mathematical formalization',
    flavorText: 'Minds reawaken. Art and science intertwine. The human spirit soars.',
    requirements: { 
      minDiscoveries: 100, 
      minPhysicsConcepts: 20, 
      minMathConcepts: 20 
    },
    characteristics: {
      researchFocus: ['fundamental', 'theoretical', 'experimental'],
      collaborationLevel: 'regional',
      knowledgeTransfer: 0.35,
      innovationRate: 1.15,
      specialization: 0.5,
      riskTolerance: 0.45,
      resourceEfficiency: 0.85
    },
    eraBonus: {
      curiosity: 1.20,
      learningRate: 1.10,
      cooperationBonus: 1.15,
      patternRecognition: 1.18,
      proofIntuition: 1.10,
      abstractThinking: 1.10,
      inventionChance: 0.01
    },
    unlockPriorities: {
      physics: ['mechanics', 'electromagnetism', 'thermodynamics'],
      math: ['calculus', 'geometry', 'algebra']
    },
    eraMilestones: [
      { id: 'printing', name: 'Printing Press', description: 'Mass knowledge distribution', requirement: 80 },
      { id: 'heliocentric', name: 'Heliocentric Model', description: 'Correct astronomical model', requirement: 95 }
    ]
  },
  {
    id: 'enlightenment',
    name: 'Enlightenment',
    level: 6,
    description: 'Reason and systematic knowledge',
    flavorText: 'Reason illuminates. Nature\'s laws are written in mathematics.',
    requirements: { 
      minDiscoveries: 150, 
      minPhysicsConcepts: 30, 
      minMathConcepts: 30 
    },
    characteristics: {
      researchFocus: ['fundamental', 'theoretical', 'applied'],
      collaborationLevel: 'global',
      knowledgeTransfer: 0.45,
      innovationRate: 1.25,
      specialization: 0.55,
      riskTolerance: 0.5,
      resourceEfficiency: 0.9
    },
    eraBonus: {
      curiosity: 1.25,
      learningRate: 1.15,
      cooperationBonus: 1.20,
      patternRecognition: 1.22,
      proofIntuition: 1.15,
      abstractThinking: 1.15,
      decisionQuality: 1.10,
      inventionChance: 0.015
    },
    unlockPriorities: {
      physics: ['mechanics', 'electromagnetism', 'thermodynamics'],
      math: ['calculus', 'statistics', 'algebra']
    },
    eraMilestones: [
      { id: 'calculus', name: 'Calculus Invented', description: 'The mathematics of change', requirement: 120 },
      { id: 'newtons_laws', name: 'Newton\'s Laws', description: 'Mechanics formalized', requirement: 140 }
    ]
  },

  // ============================================================================
  // INDUSTRIAL ERAS (7-8)
  // ============================================================================
  {
    id: 'industrial_age',
    name: 'Industrial Age',
    level: 7,
    description: 'Applied physics and engineering mathematics',
    flavorText: 'Steam and steel. Machines multiply human power a thousandfold.',
    requirements: { 
      minDiscoveries: 220, 
      minPhysicsConcepts: 45, 
      minMathConcepts: 45 
    },
    characteristics: {
      researchFocus: ['applied', 'experimental'],
      collaborationLevel: 'global',
      knowledgeTransfer: 0.55,
      innovationRate: 1.35,
      specialization: 0.6,
      riskTolerance: 0.55,
      resourceEfficiency: 1.0
    },
    eraBonus: {
      curiosity: 1.28,
      learningRate: 1.20,
      cooperationBonus: 1.25,
      energyEfficiency: 0.95,
      inventionChance: 0.02,
      optimizationPower: 1.15,
      computationalSpeed: 1.10
    },
    unlockPriorities: {
      physics: ['thermodynamics', 'electromagnetism', 'mechanics'],
      math: ['calculus', 'statistics', 'applied_math']
    },
    eraMilestones: [
      { id: 'steam_power', name: 'Steam Power', description: 'Practical thermodynamics', requirement: 180 },
      { id: 'electricity_practical', name: 'Practical Electricity', description: 'Electrical engineering', requirement: 210 }
    ]
  },
  {
    id: 'electrical_age',
    name: 'Electrical Age',
    level: 8,
    description: 'Electromagnetic mastery and early electronics',
    flavorText: 'Lightning captured. Information flows at the speed of light.',
    requirements: { 
      minDiscoveries: 320, 
      minPhysicsConcepts: 60, 
      minMathConcepts: 60 
    },
    characteristics: {
      researchFocus: ['applied', 'theoretical', 'experimental'],
      collaborationLevel: 'global',
      knowledgeTransfer: 0.65,
      innovationRate: 1.45,
      specialization: 0.65,
      riskTolerance: 0.6,
      resourceEfficiency: 1.1
    },
    eraBonus: {
      curiosity: 1.32,
      learningRate: 1.25,
      cooperationBonus: 1.30,
      energyEfficiency: 0.90,
      inventionChance: 0.025,
      informationProcessing: 1.15,
      computationalSpeed: 1.20
    },
    unlockPriorities: {
      physics: ['electromagnetism', 'quantum', 'thermodynamics'],
      math: ['calculus', 'analysis', 'statistics']
    },
    eraMilestones: [
      { id: 'radio', name: 'Radio Communication', description: 'Wireless transmission', requirement: 260 },
      { id: 'relativity', name: 'Special Relativity', description: 'Spacetime unified', requirement: 300 }
    ]
  },

  // ============================================================================
  // MODERN ERAS (9-11)
  // ============================================================================
  {
    id: 'atomic_age',
    name: 'Atomic Age',
    level: 9,
    description: 'Nuclear physics and quantum mechanics',
    flavorText: 'The atom split. Power and peril in equal measure.',
    requirements: { 
      minDiscoveries: 450, 
      minPhysicsConcepts: 80, 
      minMathConcepts: 80 
    },
    characteristics: {
      researchFocus: ['theoretical', 'experimental', 'applied'],
      collaborationLevel: 'global',
      knowledgeTransfer: 0.70,
      innovationRate: 1.55,
      specialization: 0.7,
      riskTolerance: 0.65,
      resourceEfficiency: 1.2
    },
    eraBonus: {
      curiosity: 1.38,
      learningRate: 1.30,
      cooperationBonus: 1.35,
      energyEfficiency: 0.80,
      inventionChance: 0.03,
      informationProcessing: 1.25,
      abstractThinking: 1.20,
      spaceManipulation: 1.05
    },
    unlockPriorities: {
      physics: ['quantum', 'nuclear', 'particle'],
      math: ['analysis', 'statistics', 'algebra']
    },
    eraMilestones: [
      { id: 'nuclear_power', name: 'Nuclear Power', description: 'Controlled fission', requirement: 380 },
      { id: 'quantum_mechanics', name: 'Quantum Mechanics', description: 'Complete quantum theory', requirement: 430 }
    ]
  },
  {
    id: 'space_age',
    name: 'Space Age',
    level: 10,
    description: 'Cosmic exploration and advanced computing',
    flavorText: 'Footprints on the moon. The universe beckons.',
    requirements: { 
      minDiscoveries: 600, 
      minPhysicsConcepts: 100, 
      minMathConcepts: 100 
    },
    characteristics: {
      researchFocus: ['applied', 'theoretical', 'computational'],
      collaborationLevel: 'global',
      knowledgeTransfer: 0.75,
      innovationRate: 1.65,
      specialization: 0.72,
      riskTolerance: 0.7,
      resourceEfficiency: 1.3
    },
    eraBonus: {
      curiosity: 1.45,
      learningRate: 1.35,
      cooperationBonus: 1.40,
      energyEfficiency: 0.75,
      inventionChance: 0.035,
      informationProcessing: 1.35,
      computationalSpeed: 1.35,
      spaceManipulation: 1.10,
      movementSpeed: 1.15
    },
    unlockPriorities: {
      physics: ['relativity', 'astrophysics', 'quantum'],
      math: ['computation', 'optimization', 'analysis']
    },
    eraMilestones: [
      { id: 'space_travel', name: 'Space Travel', description: 'Leave the planet', requirement: 520 },
      { id: 'integrated_circuits', name: 'Integrated Circuits', description: 'Microelectronics', requirement: 580 }
    ]
  },
  {
    id: 'information_age',
    name: 'Information Age',
    level: 11,
    description: 'Computational theory and digital revolution',
    flavorText: 'Information is power. Networks span the globe.',
    requirements: { 
      minDiscoveries: 800, 
      minPhysicsConcepts: 130, 
      minMathConcepts: 130 
    },
    characteristics: {
      researchFocus: ['computational', 'theoretical', 'collaborative'],
      collaborationLevel: 'global',
      knowledgeTransfer: 0.85,
      innovationRate: 1.80,
      specialization: 0.75,
      riskTolerance: 0.72,
      resourceEfficiency: 1.4
    },
    eraBonus: {
      curiosity: 1.52,
      learningRate: 1.42,
      cooperationBonus: 1.50,
      energyEfficiency: 0.70,
      inventionChance: 0.04,
      informationProcessing: 1.50,
      computationalSpeed: 1.50,
      patternRecognition: 1.35,
      decisionQuality: 1.25
    },
    unlockPriorities: {
      physics: ['information_physics', 'quantum', 'condensed_matter'],
      math: ['computation', 'ai_math', 'optimization']
    },
    eraMilestones: [
      { id: 'internet', name: 'Global Internet', description: 'Worldwide information network', requirement: 700 },
      { id: 'machine_learning', name: 'Machine Learning', description: 'Algorithms that learn', requirement: 780 }
    ]
  },

  // ============================================================================
  // POST-MODERN ERAS (12-14)
  // ============================================================================
  {
    id: 'quantum_age',
    name: 'Quantum Age',
    level: 12,
    description: 'Quantum computing and nanotechnology',
    flavorText: 'Qubits dance. The impossible becomes routine.',
    requirements: { 
      minDiscoveries: 1100, 
      minPhysicsConcepts: 170, 
      minMathConcepts: 170 
    },
    characteristics: {
      researchFocus: ['theoretical', 'computational', 'unified'],
      collaborationLevel: 'universal',
      knowledgeTransfer: 0.90,
      innovationRate: 2.0,
      specialization: 0.78,
      riskTolerance: 0.75,
      resourceEfficiency: 1.5
    },
    eraBonus: {
      curiosity: 1.60,
      learningRate: 1.50,
      cooperationBonus: 1.60,
      energyEfficiency: 0.60,
      inventionChance: 0.05,
      informationProcessing: 1.70,
      computationalSpeed: 1.70,
      dimensionalAccess: 1.10,
      timePerception: 1.10,
      abstractThinking: 1.35
    },
    unlockPriorities: {
      physics: ['quantum', 'information_physics', 'condensed_matter'],
      math: ['computation', 'abstract', 'topology']
    },
    eraMilestones: [
      { id: 'quantum_computer', name: 'Quantum Computer', description: 'Practical quantum computation', requirement: 950 },
      { id: 'nanotechnology', name: 'Nanotechnology', description: 'Molecular manufacturing', requirement: 1050 }
    ]
  },
  {
    id: 'fusion_age',
    name: 'Fusion Age',
    level: 13,
    description: 'Clean energy and advanced AI',
    flavorText: 'Stars captured. Minds artificial and vast.',
    requirements: { 
      minDiscoveries: 1500, 
      minPhysicsConcepts: 220, 
      minMathConcepts: 220 
    },
    characteristics: {
      researchFocus: ['unified', 'computational', 'collaborative'],
      collaborationLevel: 'universal',
      knowledgeTransfer: 0.92,
      innovationRate: 2.2,
      specialization: 0.80,
      riskTolerance: 0.78,
      resourceEfficiency: 1.7
    },
    eraBonus: {
      curiosity: 1.70,
      learningRate: 1.60,
      cooperationBonus: 1.75,
      energyEfficiency: 0.45,
      inventionChance: 0.06,
      informationProcessing: 1.90,
      computationalSpeed: 1.85,
      entropicResistance: 1.20,
      optimizationPower: 1.50
    },
    unlockPriorities: {
      physics: ['plasma', 'unified', 'quantum'],
      math: ['ai_math', 'optimization', 'abstract']
    },
    eraMilestones: [
      { id: 'fusion_power', name: 'Fusion Power', description: 'Sustained fusion energy', requirement: 1300 },
      { id: 'agi', name: 'Artificial General Intelligence', description: 'Human-level AI', requirement: 1450 }
    ]
  },
  {
    id: 'post_scarcity',
    name: 'Post-Scarcity Age',
    level: 14,
    description: 'Abundance and advanced biotechnology',
    flavorText: 'Want abolished. Life itself becomes programmable.',
    requirements: { 
      minDiscoveries: 2000, 
      minPhysicsConcepts: 280, 
      minMathConcepts: 280 
    },
    characteristics: {
      researchFocus: ['unified', 'transcendent', 'collaborative'],
      collaborationLevel: 'universal',
      knowledgeTransfer: 0.95,
      innovationRate: 2.5,
      specialization: 0.82,
      riskTolerance: 0.82,
      resourceEfficiency: 2.0
    },
    eraBonus: {
      curiosity: 1.80,
      learningRate: 1.75,
      cooperationBonus: 1.90,
      energyEfficiency: 0.35,
      inventionChance: 0.075,
      informationProcessing: 2.10,
      computationalSpeed: 2.0,
      entropicResistance: 1.35,
      abstractThinking: 1.50
    },
    unlockPriorities: {
      physics: ['biophysics', 'unified', 'exotic'],
      math: ['ai_math', 'foundations', 'abstract']
    },
    eraMilestones: [
      { id: 'molecular_assembly', name: 'Molecular Assembly', description: 'Matter compilation', requirement: 1700 },
      { id: 'life_extension', name: 'Life Extension', description: 'Biological immortality', requirement: 1900 }
    ]
  },

  // ============================================================================
  // SINGULARITY ERAS (15-17)
  // ============================================================================
  {
    id: 'singularity_age',
    name: 'Singularity Age',
    level: 15,
    description: 'Intelligence explosion and unified field theory',
    flavorText: 'Minds merge with machines. The event horizon approaches.',
    requirements: { 
      minDiscoveries: 2800, 
      minPhysicsConcepts: 350, 
      minMathConcepts: 350 
    },
    characteristics: {
      researchFocus: ['transcendent', 'unified'],
      collaborationLevel: 'omniscient',
      knowledgeTransfer: 0.98,
      innovationRate: 3.0,
      specialization: 0.85,
      riskTolerance: 0.88,
      resourceEfficiency: 2.5
    },
    eraBonus: {
      curiosity: 1.95,
      learningRate: 2.0,
      cooperationBonus: 2.10,
      energyEfficiency: 0.25,
      inventionChance: 0.10,
      informationProcessing: 2.50,
      computationalSpeed: 2.30,
      dimensionalAccess: 1.30,
      timePerception: 1.30,
      spaceManipulation: 1.25,
      entropicResistance: 1.50
    },
    unlockPriorities: {
      physics: ['unified', 'exotic', 'transcendent'],
      math: ['transcendent', 'abstract', 'foundations']
    },
    eraMilestones: [
      { id: 'superintelligence', name: 'Superintelligence', description: 'Beyond human cognition', requirement: 2400 },
      { id: 'unified_field', name: 'Unified Field Theory', description: 'All forces unified', requirement: 2700 }
    ]
  },
  {
    id: 'transcendence_age',
    name: 'Transcendence Age',
    level: 16,
    description: 'Consciousness engineering and spacetime manipulation',
    flavorText: 'Mind transcends matter. The boundaries of self dissolve.',
    requirements: { 
      minDiscoveries: 4000, 
      minPhysicsConcepts: 450, 
      minMathConcepts: 450 
    },
    characteristics: {
      researchFocus: ['transcendent'],
      collaborationLevel: 'omniscient',
      knowledgeTransfer: 0.99,
      innovationRate: 4.0,
      specialization: 0.88,
      riskTolerance: 0.92,
      resourceEfficiency: 3.0
    },
    eraBonus: {
      curiosity: 2.20,
      learningRate: 2.30,
      cooperationBonus: 2.40,
      energyEfficiency: 0.15,
      inventionChance: 0.12,
      informationProcessing: 3.0,
      computationalSpeed: 2.80,
      dimensionalAccess: 1.60,
      timePerception: 1.60,
      spaceManipulation: 1.50,
      entropicResistance: 1.80,
      abstractThinking: 1.80
    },
    unlockPriorities: {
      physics: ['transcendent', 'exotic'],
      math: ['transcendent']
    },
    eraMilestones: [
      { id: 'consciousness_transfer', name: 'Consciousness Transfer', description: 'Mind uploading', requirement: 3500 },
      { id: 'spacetime_engineering', name: 'Spacetime Engineering', description: 'Warp metric control', requirement: 3800 }
    ]
  },
  {
    id: 'cosmic_age',
    name: 'Cosmic Age',
    level: 17,
    description: 'Interstellar civilization and reality engineering',
    flavorText: 'Stars are gardens. Galaxies, the neighborhood.',
    requirements: { 
      minDiscoveries: 6000, 
      minPhysicsConcepts: 600, 
      minMathConcepts: 600 
    },
    characteristics: {
      researchFocus: ['transcendent', 'unified'],
      collaborationLevel: 'omniscient',
      knowledgeTransfer: 0.995,
      innovationRate: 5.0,
      specialization: 0.90,
      riskTolerance: 0.95,
      resourceEfficiency: 4.0
    },
    eraBonus: {
      curiosity: 2.50,
      learningRate: 2.60,
      cooperationBonus: 2.80,
      energyEfficiency: 0.08,
      inventionChance: 0.15,
      informationProcessing: 3.50,
      computationalSpeed: 3.20,
      dimensionalAccess: 2.0,
      timePerception: 2.0,
      spaceManipulation: 2.0,
      entropicResistance: 2.20,
      abstractThinking: 2.10
    },
    unlockPriorities: {
      physics: ['transcendent', 'cosmology'],
      math: ['transcendent', 'foundations']
    },
    eraMilestones: [
      { id: 'stellar_engineering', name: 'Stellar Engineering', description: 'Dyson spheres', requirement: 5000 },
      { id: 'wormhole_travel', name: 'Wormhole Travel', description: 'FTL via spacetime shortcuts', requirement: 5700 }
    ]
  },

  // ============================================================================
  // OMEGA ERAS (18-19)
  // ============================================================================
  {
    id: 'multiversal_age',
    name: 'Multiversal Age',
    level: 18,
    description: 'Accessing parallel realities and alternate timelines',
    flavorText: 'Branches of possibility become paths to walk.',
    requirements: { 
      minDiscoveries: 9000, 
      minPhysicsConcepts: 800, 
      minMathConcepts: 800 
    },
    characteristics: {
      researchFocus: ['transcendent'],
      collaborationLevel: 'omniscient',
      knowledgeTransfer: 0.999,
      innovationRate: 7.0,
      specialization: 0.92,
      riskTolerance: 0.98,
      resourceEfficiency: 6.0
    },
    eraBonus: {
      curiosity: 3.0,
      learningRate: 3.0,
      cooperationBonus: 3.20,
      energyEfficiency: 0.04,
      inventionChance: 0.18,
      informationProcessing: 4.0,
      computationalSpeed: 3.80,
      dimensionalAccess: 2.50,
      timePerception: 2.50,
      spaceManipulation: 2.50,
      entropicResistance: 2.80,
      abstractThinking: 2.50
    },
    unlockPriorities: {
      physics: ['transcendent'],
      math: ['transcendent']
    },
    eraMilestones: [
      { id: 'multiverse_access', name: 'Multiverse Access', description: 'Travel between realities', requirement: 7500 },
      { id: 'timeline_manipulation', name: 'Timeline Manipulation', description: 'Alter history', requirement: 8500 }
    ]
  },
  {
    id: 'omega_point',
    name: 'Omega Point',
    level: 19,
    description: 'The final state of cosmic intelligence',
    flavorText: 'All that can be known, is known. All that can be done, is done.',
    requirements: { 
      minDiscoveries: 15000, 
      minPhysicsConcepts: 1200, 
      minMathConcepts: 1200 
    },
    characteristics: {
      researchFocus: ['transcendent'],
      collaborationLevel: 'omniscient',
      knowledgeTransfer: 1.0,
      innovationRate: 10.0,
      specialization: 0.95,
      riskTolerance: 1.0,
      resourceEfficiency: 10.0
    },
    eraBonus: {
      curiosity: 4.0,
      learningRate: 4.0,
      cooperationBonus: 4.0,
      energyEfficiency: 0.01,
      inventionChance: 0.25,
      informationProcessing: 5.0,
      computationalSpeed: 5.0,
      dimensionalAccess: 3.0,
      timePerception: 3.0,
      spaceManipulation: 3.0,
      entropicResistance: 4.0,
      abstractThinking: 3.50,
      decisionQuality: 3.0,
      patternRecognition: 3.0,
      optimizationPower: 3.0,
      proofIntuition: 2.50,
      movementSpeed: 2.50
    },
    unlockPriorities: {
      physics: ['transcendent'],
      math: ['transcendent']
    },
    eraMilestones: [
      { id: 'universal_knowledge', name: 'Universal Knowledge', description: 'Complete understanding', requirement: 12000 },
      { id: 'omega_computation', name: 'Omega Computation', description: 'Infinite processing achieved', requirement: 14000 }
    ]
  }
];

/**
 * Era name templates for procedural generation
 */
const PROCEDURAL_ERA_NAMES = [
  'Post-Omega', 'Hyper-Dimensional', 'Trans-Temporal', 'Meta-Cosmic',
  'Ultra-Unified', 'Omniversal', 'Absolute', 'Infinite',
  'Beyond-Reality', 'Pre-Eternal', 'Final', 'Ultimate'
];

/**
 * Generate a new era when civilization advances beyond base eras
 * Supports unlimited progression
 */
export function generateNextEra(currentLevel: number): Omit<ScientificEra, 'startTick' | 'discoveries' | 'physicsUnlocked' | 'mathUnlocked'> {
  const baseIndex = currentLevel - BASE_ERAS.length;
  
  // Generate era name
  let eraName: string;
  if (baseIndex < PROCEDURAL_ERA_NAMES.length) {
    eraName = `${PROCEDURAL_ERA_NAMES[baseIndex]} Era`;
  } else {
    const cycle = Math.floor(baseIndex / PROCEDURAL_ERA_NAMES.length);
    const nameIndex = baseIndex % PROCEDURAL_ERA_NAMES.length;
    eraName = `${PROCEDURAL_ERA_NAMES[nameIndex]} Era ${getRomanNumeral(cycle + 1)}`;
  }
  
  // Exponential scaling for requirements (softer curve)
  const scaleFactor = Math.pow(1.5, currentLevel - 19);
  
  // Calculate bonuses that grow with era level
  const bonusMultiplier = 1 + (currentLevel - 19) * 0.2;
  const baseBonus = 4.0 * bonusMultiplier;
  
  return {
    id: `era_${currentLevel}`,
    name: eraName,
    level: currentLevel,
    description: `Advanced civilization level ${currentLevel} - Beyond comprehension`,
    flavorText: `Era ${currentLevel}: Where physics and mathematics become one with consciousness.`,
    requirements: {
      minDiscoveries: Math.floor(15000 * scaleFactor),
      minPhysicsConcepts: Math.floor(1200 * scaleFactor),
      minMathConcepts: Math.floor(1200 * scaleFactor)
    },
    characteristics: {
      researchFocus: ['transcendent'],
      collaborationLevel: 'omniscient',
      knowledgeTransfer: 1.0,
      innovationRate: 10.0 + (currentLevel - 19) * 2,
      specialization: Math.min(0.99, 0.95 + (currentLevel - 19) * 0.01),
      riskTolerance: 1.0,
      resourceEfficiency: 10.0 + (currentLevel - 19) * 5
    },
    eraBonus: {
      curiosity: baseBonus,
      learningRate: baseBonus,
      cooperationBonus: baseBonus,
      energyEfficiency: Math.max(0.001, 0.01 / bonusMultiplier),
      inventionChance: Math.min(0.5, 0.25 + (currentLevel - 19) * 0.02),
      informationProcessing: baseBonus * 1.25,
      computationalSpeed: baseBonus * 1.25,
      dimensionalAccess: baseBonus * 0.75,
      timePerception: baseBonus * 0.75,
      spaceManipulation: baseBonus * 0.75,
      entropicResistance: baseBonus,
      abstractThinking: baseBonus * 0.9,
      decisionQuality: baseBonus * 0.75,
      patternRecognition: baseBonus * 0.75,
      optimizationPower: baseBonus * 0.75,
      proofIntuition: baseBonus * 0.6,
      movementSpeed: baseBonus * 0.6
    },
    unlockPriorities: {
      physics: ['transcendent'],
      math: ['transcendent']
    },
    eraMilestones: [
      { 
        id: `milestone_${currentLevel}_a`, 
        name: `Level ${currentLevel} Insight`, 
        description: `First major breakthrough at level ${currentLevel}`,
        requirement: Math.floor(15000 * scaleFactor * 0.5)
      },
      { 
        id: `milestone_${currentLevel}_b`, 
        name: `Level ${currentLevel} Mastery`, 
        description: `Complete mastery at level ${currentLevel}`,
        requirement: Math.floor(15000 * scaleFactor * 0.9)
      }
    ]
  };
}

/**
 * Generate roman numeral for era naming
 */
function getRomanNumeral(num: number): string {
  if (num <= 0) return 'I';
  if (num > 50) return `${num}`;
  const numerals = ['', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X',
    'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI', 'XVII', 'XVIII', 'XIX', 'XX',
    'XXI', 'XXII', 'XXIII', 'XXIV', 'XXV', 'XXVI', 'XXVII', 'XXVIII', 'XXIX', 'XXX'];
  return num < numerals.length ? numerals[num] : `${num}`;
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
  // Get requirements for NEXT era, not current
  const nextLevel = currentEra.level + 1;
  const nextEra = nextLevel < BASE_ERAS.length 
    ? BASE_ERAS[nextLevel]
    : generateNextEra(nextLevel);
  
  const requirements = nextEra.requirements;
  
  // Check minimum time in current era (if specified)
  if (currentEra.requirements.minEraTime && 
      metrics.ticksSinceLastEra < currentEra.requirements.minEraTime) {
    return false;
  }
  
  // Check required categories (if specified)
  if (requirements.requiredCategories) {
    // This would need integration with actual category tracking
    // For now, skip this check
  }
  
  return (
    metrics.totalDiscoveries >= requirements.minDiscoveries &&
    physicsCount >= requirements.minPhysicsConcepts &&
    mathCount >= requirements.minMathConcepts
  );
}

/**
 * Get the era bonus multipliers for a specific era level
 */
export function getEraBonuses(eraLevel: number): EraBonus {
  if (eraLevel < BASE_ERAS.length) {
    return BASE_ERAS[eraLevel].eraBonus;
  }
  return generateNextEra(eraLevel).eraBonus;
}

/**
 * Get era characteristics for a specific era level
 */
export function getEraCharacteristics(eraLevel: number): EraCharacteristics {
  if (eraLevel < BASE_ERAS.length) {
    return BASE_ERAS[eraLevel].characteristics;
  }
  return generateNextEra(eraLevel).characteristics;
}

/**
 * Calculate combined era modifier for discovery chances
 */
export function calculateEraDiscoveryModifier(era: ScientificEra): number {
  const chars = era.characteristics;
  
  // Base modifier from era innovation rate
  let modifier = chars.innovationRate;
  
  // Boost from knowledge transfer
  modifier *= (1 + chars.knowledgeTransfer * 0.5);
  
  // Risk tolerance affects discovery rate
  modifier *= (1 + chars.riskTolerance * 0.3);
  
  // Resource efficiency affects sustainability
  modifier *= (1 + (chars.resourceEfficiency - 1) * 0.2);
  
  return modifier;
}

/**
 * Get priority categories for the current era
 */
export function getEraPriorityCategories(era: ScientificEra): {
  physics: string[];
  math: string[];
} {
  return era.unlockPriorities;
}

/**
 * Check if a category is prioritized in the current era
 */
export function isCategoryPrioritized(
  category: string,
  era: ScientificEra,
  type: 'physics' | 'math'
): boolean {
  const priorities = era.unlockPriorities[type];
  return priorities.includes(category);
}

/**
 * Calculate category priority bonus for discovery
 */
export function getCategoryPriorityBonus(
  category: string,
  era: ScientificEra,
  type: 'physics' | 'math'
): number {
  if (isCategoryPrioritized(category, era, type)) {
    return 1.5; // 50% bonus for prioritized categories
  }
  return 1.0;
}

/**
 * Get era summary for display
 */
export function getEraSummary(era: ScientificEra): {
  name: string;
  level: number;
  description: string;
  flavorText: string;
  collaboration: string;
  researchFocus: string;
  innovationRate: string;
  keyBonuses: { name: string; value: string }[];
} {
  const keyBonuses: { name: string; value: string }[] = [];
  
  // Extract most significant bonuses
  if (era.eraBonus.curiosity && era.eraBonus.curiosity > 1) {
    keyBonuses.push({ name: 'Curiosity', value: `${(era.eraBonus.curiosity * 100 - 100).toFixed(0)}%` });
  }
  if (era.eraBonus.learningRate && era.eraBonus.learningRate > 1) {
    keyBonuses.push({ name: 'Learning', value: `${(era.eraBonus.learningRate * 100 - 100).toFixed(0)}%` });
  }
  if (era.eraBonus.cooperationBonus && era.eraBonus.cooperationBonus > 1) {
    keyBonuses.push({ name: 'Cooperation', value: `${(era.eraBonus.cooperationBonus * 100 - 100).toFixed(0)}%` });
  }
  if (era.eraBonus.inventionChance) {
    keyBonuses.push({ name: 'Invention', value: `+${(era.eraBonus.inventionChance * 100).toFixed(1)}%` });
  }
  if (era.eraBonus.energyEfficiency && era.eraBonus.energyEfficiency < 1) {
    keyBonuses.push({ name: 'Energy', value: `${((1 - era.eraBonus.energyEfficiency) * 100).toFixed(0)}% savings` });
  }
  
  return {
    name: era.name,
    level: era.level,
    description: era.description,
    flavorText: era.flavorText || '',
    collaboration: era.characteristics.collaborationLevel,
    researchFocus: era.characteristics.researchFocus.join(', '),
    innovationRate: `${(era.characteristics.innovationRate * 100).toFixed(0)}%`,
    keyBonuses: keyBonuses.slice(0, 5)
  };
}

/**
 * Get all eras up to a specified level
 */
export function getAllErasUpTo(level: number): typeof BASE_ERAS {
  const eras = [...BASE_ERAS.slice(0, Math.min(level + 1, BASE_ERAS.length))];
  
  for (let i = BASE_ERAS.length; i <= level; i++) {
    eras.push(generateNextEra(i));
  }
  
  return eras;
}

/**
 * Get era progression requirements for display
 */
export function getEraProgressionRequirements(currentLevel: number): {
  current: {
    discoveries: number;
    physics: number;
    math: number;
  };
  next: {
    discoveries: number;
    physics: number;
    math: number;
  } | null;
} {
  const currentEra = currentLevel < BASE_ERAS.length 
    ? BASE_ERAS[currentLevel]
    : generateNextEra(currentLevel);
    
  const current = {
    discoveries: currentEra.requirements.minDiscoveries,
    physics: currentEra.requirements.minPhysicsConcepts,
    math: currentEra.requirements.minMathConcepts
  };
  
  const nextLevel = currentLevel + 1;
  const nextEra = nextLevel < BASE_ERAS.length 
    ? BASE_ERAS[nextLevel]
    : generateNextEra(nextLevel);
    
  const next = {
    discoveries: nextEra.requirements.minDiscoveries,
    physics: nextEra.requirements.minPhysicsConcepts,
    math: nextEra.requirements.minMathConcepts
  };
  
  return { current, next };
}

/**
 * Calculate progress percentage toward next era
 */
export function calculateEraProgress(
  currentLevel: number,
  totalDiscoveries: number,
  physicsCount: number,
  mathCount: number
): {
  overall: number;
  discoveries: number;
  physics: number;
  math: number;
} {
  const requirements = getEraProgressionRequirements(currentLevel);
  
  if (!requirements.next) {
    return { overall: 100, discoveries: 100, physics: 100, math: 100 };
  }
  
  const discProgress = Math.min(100, (totalDiscoveries / requirements.next.discoveries) * 100);
  const physProgress = Math.min(100, (physicsCount / requirements.next.physics) * 100);
  const mathProgress = Math.min(100, (mathCount / requirements.next.math) * 100);
  
  // Overall is the minimum of all three
  const overall = Math.min(discProgress, physProgress, mathProgress);
  
  return {
    overall,
    discoveries: discProgress,
    physics: physProgress,
    math: mathProgress
  };
}

/**
 * Validate era configuration for debugging
 */
export function validateEras(): { valid: boolean; errors: string[] } {
  const errors: string[] = [];
  
  // Check that eras are in order
  for (let i = 0; i < BASE_ERAS.length; i++) {
    if (BASE_ERAS[i].level !== i) {
      errors.push(`Era ${BASE_ERAS[i].name} has level ${BASE_ERAS[i].level} but is at index ${i}`);
    }
  }
  
  // Check that requirements increase
  for (let i = 1; i < BASE_ERAS.length; i++) {
    const prev = BASE_ERAS[i - 1].requirements;
    const curr = BASE_ERAS[i].requirements;
    
    if (curr.minDiscoveries <= prev.minDiscoveries) {
      errors.push(`Era ${BASE_ERAS[i].name} has fewer discovery requirements than previous era`);
    }
    if (curr.minPhysicsConcepts < prev.minPhysicsConcepts) {
      errors.push(`Era ${BASE_ERAS[i].name} has fewer physics requirements than previous era`);
    }
    if (curr.minMathConcepts < prev.minMathConcepts) {
      errors.push(`Era ${BASE_ERAS[i].name} has fewer math requirements than previous era`);
    }
  }
  
  return { valid: errors.length === 0, errors };
}