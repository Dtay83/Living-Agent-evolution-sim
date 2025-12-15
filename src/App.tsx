// Contact: Name: dtay83 <dartey.banahene@gmail.com>
import React, { useState, useMemo, useEffect, useRef, useCallback } from "react";
import { exportInventionHistory, exportEvolutionData, exportCompleteData, exportConversations } from "./utils/exportData";
import { ChatPanel, AnalysisPanel, PhysicsPanel, MathPanel, EraPanel, SpeechPanel, ExplanationPanel, AutonomousPanel } from "./ui-components";
import { AnalysisPanel as AnalysisPanelType } from "./ui-components/AnalysisPanel";
import { 
  CommunicationLog, 
  CommunicationState,
  AgentMessage,
  initializeCommunicationLog,
  initializeCommunicationState,
  updateAgentCommunication,
  shouldCommunicate,
  generateMessage,
  createUserMessage,
  generateAgentResponse,
  applyUserMessageEffects
} from "./communication-system";
import { ConsciousnessLevel, ConsciousnessState } from "./consciousness-system";
import {
  ChallengeState,
  initializeChallengeState,
  processChallenges,
  getEffectiveFoodSpawnChance,
  getEffectiveFoodSpawnCount,
  getChallengeEnergyCost,
  getActiveChallengesSummary,
  hasNegativeChallenge,
  hasPositiveEvent
} from "./core/challenges";
import {
  CivilizationPhysics,
  initializeCivilizationPhysics,
  checkPhysicsDiscovery,
  getPhysicsMovementCost,
  getAgentPhysicsBonuses,
  getPhysicsEnhancedLearningRate,
  getPhysicsInventionBonus,
  getPhysicsSummary
} from "./physics-integration";
import type { PhysicsConcept } from "./science-system/physics";
import {
  CivilizationMath,
  initializeCivilizationMath,
  checkMathDiscovery,
  getResourceOptimizationBonus,
  getMathEnhancedLearningRate,
  getMathExplorationBonus,
  getMathSummary
} from "./math-integration";
import type { MathConcept } from "./science-system/mathematics";
import {
  CivilizationEraState,
  initializeCivilizationEra,
  checkEraAdvancement,
  checkMilestones,
  updateProgressionMetrics,
  getEraBonuses,
  getEraEnhancedLearningRate,
  getEraDiscoveryBonus,
  getEraEnergyEfficiency,
  calculateInheritedInventions
} from "./era-integration";
import {
  CivilizationSpeechState,
  AgentLanguageState,
  initializeCivilizationSpeech,
  initializeAgentLanguage,
  expandVocabulary,
  getPhysicsVocabulary,
  getMathVocabulary,
  canInitiateDialogue,
  initiateDialogue,
  continueDialogue,
  transferKnowledge,
  updateCommunicationStyle
} from "./speech-integration";
import {
  AutonomousState,
  initializeAutonomousState,
  updateAutonomousState,
  getAutonomySummary,  hasAutonomy,
  getAutonomyLevel,
  makeAutonomousDecision,
  AUTONOMY_THRESHOLDS,
  ExperienceMemory
} from "./autonomous-system";
import {
  runAutoLearningForAllAgents,
  AUTO_LEARNING_CONFIG,
  AutoLearningResult
} from "./explanation-system";

type Direction = "up" | "down" | "left" | "right" | "stay";

interface Genes {
  foodPreference: number;        // 0–1: prioritize food when hungry
  exploration: number;           // 0–1: how often they wander
  reproductionThreshold: number; // energy needed to reproduce
  mutationRate: number;          // 0–1: chance each gene mutates
  traitId: number;               // lineage / random trait marker
  
  // NEW: Learning & invention genes
  curiosity: number;             // 0–1: likelihood of discovering new things
  social: number;                // 0–1: ability to teach/learn from others
  creativity: number;            // 0–1: scales the quality and power of discovered inventions
  patience: number;              // 0–1: influences storage capacity and long-term benefits
}

interface Memory {
  qTable: Record<string, number>; // RL value table: (state|action) -> Q
}

/**
 * INVENTION SYSTEM
 * 
 * Agents can discover "inventions" that provide gameplay advantages.
 * Discovery is influenced by:
 * - Energy level (need cognitive surplus)
 * - Curiosity gene (higher = more likely to discover)
 * - Prerequisites (tech tree structure)
 * 
 * Inventions can be passed to offspring based on the social gene,
 * simulating cultural/knowledge transmission.
 */

type InventionEffect = 
  | { type: 'energy_efficiency'; multiplier: number }
  | { type: 'food_detection_range'; range: number }
  | { type: 'reproduction_boost'; bonus: number }
  | { type: 'defense'; protection: number }
  | { type: 'storage'; capacity: number };

interface Invention {
  id: string;
  name: string;
  type: 'tool' | 'technique' | 'structure';
  effect: InventionEffect;
  discoveredAt: number;      // tick when invented
  discoveredBy: number;      // agent ID
  requirements: string[];    // prerequisites to use
  description?: string;      // human-readable description
}

interface DiscoveryEvent {
  tick: number;
  agentId: number;
  invention: Invention;
}

interface Agent {
  id: number;
  x: number;
  y: number;
  energy: number;
  genes: Genes;
  memory: Memory;
  lastRule?: string;
  inventions: Invention[];  // Things this agent has discovered
  inventionPoints: number;  // Accumulated creativity points for discovering new inventions
}

interface Cell {
  food: boolean;
  agentId?: number;
}

interface HistoryPoint {
  tick: number;
  totalAgents: number;
  byTrait: Record<number, number>;
}

interface WorldState {
  grid: Cell[][];
  agents: Agent[];
  tick: number;
  history: HistoryPoint[];
  discoveries: DiscoveryEvent[];  // NEW: Track all discoveries
}

/**
 * CODE QUALITY IMPROVEMENT: Extract constants into configuration object
 * All magic numbers are now organized in a structured config for better maintainability
 */
const CONFIG = {
  grid: {
    initialWidth: 16,
    initialHeight: 10,
    maxWidth: 27,              // Maximum grid width
    maxHeight: 30,             // Maximum grid height
    expandBy: 2,               // Number of cells to add when expanding
    expandThreshold: 0.6,      // Expand when this % of cells are occupied
  },
  simulation: {
    initialAgents: 6,
    initialFood: 25,           // Increased from 18 - more food at start
    foodSpawnChance: 0.8,      // Increased from 0.6 - food spawns more often
    foodSpawnCount: 2,         // Increased from 1 - spawn 2 food at a time
    baseEnergyCost: 0.5,       // Reduced from 1 - agents burn energy slower
    foodEnergyBonus: 8,        // Increased from 5 - food is more nutritious
    reproductionReward: 2,
    deathPenalty: 5,
  },
  rl: {
    alpha: 0.3,   // learning rate
    gamma: 0.9,   // discount factor
    epsilon: 0.2, // exploration probability
  },
  genes: {
    foodPreference: { min: 0.6, max: 1.0 },
    exploration: { min: 0.3, max: 0.8 },
    reproductionThreshold: { min: 12, max: 20 },  // Lowered - easier to reproduce
    mutationRate: { min: 0.1, max: 0.3 },
    curiosity: { min: 0.2, max: 0.8 },
    social: { min: 0.3, max: 0.9 },
    creativity: { min: 0.1, max: 0.7 },
    patience: { min: 0.2, max: 0.8 },
    mutation: {
      foodPreferenceMagnitude: 0.15,
      explorationMagnitude: 0.2,
      reproductionThresholdMagnitude: 3,
      mutationRateMagnitude: 0.05,
      curiosityMagnitude: 0.15,
      socialMagnitude: 0.15,
      creativityMagnitude: 0.15,
      patienceMagnitude: 0.15,
      newTraitChance: 0.08,
      mutationRateEvolveMin: 0.01,
      mutationRateEvolveMax: 0.6,
      reproductionThresholdMin: 8,
      reproductionThresholdMax: 30,
    },
  },
  energy: {
    hungryCutoffRatio: 6,
    initialMin: 15,            // Increased from 10 - agents start with more energy
    initialMax: 25,            // Increased from 16 - agents start with more energy
  },
} as const;

// Backwards compatibility aliases - these are now dynamic and managed by state
let GRID_WIDTH: number = CONFIG.grid.initialWidth;
let GRID_HEIGHT: number = CONFIG.grid.initialHeight;
const INITIAL_AGENTS = CONFIG.simulation.initialAgents;
const INITIAL_FOOD = CONFIG.simulation.initialFood;
const ALPHA = CONFIG.rl.alpha;
const GAMMA = CONFIG.rl.gamma;
const EPSILON = CONFIG.rl.epsilon;

/**
 * LIMITLESS INVENTION SYSTEM
 * 
 * Instead of a fixed tech tree, inventions are procedurally generated based on:
 * - Agent's creativity and curiosity genes
 * - Invention points accumulated through exploration and survival
 * - Random inspiration that creates unique, emergent inventions
 * 
 * This allows for unlimited creativity with no cap on discoveries.
 */

// Invention name components for procedural generation
const INVENTION_PREFIXES = [
  'Efficient', 'Advanced', 'Enhanced', 'Optimized', 'Swift', 'Powerful',
  'Refined', 'Masterful', 'Superior', 'Elite', 'Expert', 'Precise',
  'Strategic', 'Tactical', 'Innovative', 'Revolutionary', 'Adaptive', 'Dynamic'
];

const INVENTION_THEMES = [
  'Foraging', 'Hunting', 'Gathering', 'Navigation', 'Communication',
  'Defense', 'Offense', 'Survival', 'Cooperation', 'Efficiency',
  'Awareness', 'Adaptation', 'Endurance', 'Speed', 'Strength',
  'Intelligence', 'Memory', 'Reflexes', 'Instinct', 'Wisdom'
];

const INVENTION_TYPES: Array<'tool' | 'technique' | 'structure'> = [
  'tool', 'technique', 'structure'
];

/**
 * Generate a unique invention based on agent's capabilities and random inspiration
 */
function generateInvention(agent: Agent, tick: number, inventionNumber: number): Invention {
  // Use agent's creativity to influence invention quality
  const creativityFactor = agent.genes.curiosity * agent.genes.creativity;
  
  // Random invention type
  const type = INVENTION_TYPES[Math.floor(Math.random() * INVENTION_TYPES.length)];
  
  // Generate unique name based on invention number and random elements
  const prefix = INVENTION_PREFIXES[Math.floor(Math.random() * INVENTION_PREFIXES.length)];
  const theme = INVENTION_THEMES[Math.floor(Math.random() * INVENTION_THEMES.length)];
  const name = `${prefix} ${theme}`;
  const id = `invention_${agent.id}_${inventionNumber}_${tick}`;
  
  // Determine effect type based on creativity and randomness
  const effectRoll = Math.random();
  let effect: InventionEffect;
  let description: string;
  
  if (effectRoll < 0.4) {
    // Energy efficiency - scales with creativity (ACTIVE EFFECT)
    const multiplier = 0.95 - (creativityFactor * 0.25); // 0.7 to 0.95 range
    effect = { type: 'energy_efficiency', multiplier: Math.max(0.5, multiplier) };
    description = `Reduces energy cost by ${Math.round((1 - multiplier) * 100)}%`;
  } else if (effectRoll < 0.7) {
    // Reproduction boost - scales with creativity (ACTIVE EFFECT)
    const bonus = Math.ceil(2 + creativityFactor * 8); // 2 to 10 bonus
    effect = { type: 'reproduction_boost', bonus };
    description = `+${bonus} bonus energy for reproduction`;
  } else if (effectRoll < 0.85) {
    // Food detection range - scales with exploration (PARTIAL: Function exists, not integrated in decision logic)
    const range = Math.ceil(1 + agent.genes.exploration * 3); // 1 to 4 range
    effect = { type: 'food_detection_range', range };
    description = `Detect food ${range} cells away`;
  } else if (effectRoll < 0.95) {
    // Storage capacity - scales with patience (FUTURE: Awaiting game mechanic implementation)
    const capacity = Math.ceil(5 + agent.genes.patience * 20); // 5 to 25 capacity
    effect = { type: 'storage', capacity };
    description = `Store ${capacity} extra energy`;
  } else {
    // Defense - scales with both genes (FUTURE: Awaiting damage system implementation)
    const protection = Math.min(0.5, 0.1 + creativityFactor * 0.4); // 0.1 to 0.5
    effect = { type: 'defense', protection };
    description = `${Math.round(protection * 100)}% chance to avoid energy loss`;
  }
  
  return {
    id,
    name,
    type,
    effect,
    discoveredAt: tick,
    discoveredBy: agent.id,
    requirements: [], // No prerequisites in limitless system
    description
  };
}

/**
 * TYPE SAFETY IMPROVEMENT: Direction movement deltas mapping
 * Provides consistent directional movement logic across the codebase
 */
const DIRECTION_DELTAS: Record<Direction, { dx: number; dy: number }> = {
  up: { dx: 0, dy: -1 },
  down: { dx: 0, dy: 1 },
  left: { dx: -1, dy: 0 },
  right: { dx: 1, dy: 0 },
  stay: { dx: 0, dy: 0 },
};

/**
 * Helper function to apply direction movement with bounds checking
 */
function applyDirection(
  x: number,
  y: number,
  dir: Direction
): { x: number; y: number } {
  const delta = DIRECTION_DELTAS[dir];
  return {
    x: Math.max(0, Math.min(GRID_WIDTH - 1, x + delta.dx)),
    y: Math.max(0, Math.min(GRID_HEIGHT - 1, y + delta.dy)),
  };
}

function createEmptyGrid(): Cell[][] {
  return Array.from({ length: GRID_HEIGHT }, () =>
    Array.from({ length: GRID_WIDTH }, () => ({ food: false } as Cell))
  );
}

function randomInt(max: number) {
  return Math.floor(Math.random() * max);
}

function placeRandomFood(grid: Cell[][], count: number): Cell[][] {
  const copy = grid.map(row => row.map(cell => ({ ...cell })));
  let placed = 0;
  let safety = 0;
  while (placed < count && safety < 2000) {
    safety++;
    const y = randomInt(GRID_HEIGHT);
    const x = randomInt(GRID_WIDTH);
    if (!copy[y][x].food && copy[y][x].agentId === undefined) {
      copy[y][x].food = true;
      placed++;
    }
  }
  return copy;
}

/**
 * DYNAMIC GRID EXPANSION SYSTEM
 * 
 * Expands the grid when population density exceeds threshold.
 * This allows the civilization to grow beyond the initial world size.
 * Maximum size: 27x30 (810 cells)
 */

/**
 * Check if the grid should expand based on population density
 */
function shouldExpandGrid(
  grid: Cell[][],
  agentCount: number,
  currentWidth: number,
  currentHeight: number
): boolean {
  // Don't expand if already at max size
  if (currentWidth >= CONFIG.grid.maxWidth && currentHeight >= CONFIG.grid.maxHeight) {
    return false;
  }
  
  const totalCells = currentWidth * currentHeight;
  const occupancyRate = agentCount / totalCells;
  
  return occupancyRate >= CONFIG.grid.expandThreshold;
}

/**
 * Expand the grid by adding cells to the edges
 * Returns the new grid and updated dimensions
 */
function expandGrid(
  grid: Cell[][],
  agents: Agent[],
  currentWidth: number,
  currentHeight: number
): { 
  grid: Cell[][]; 
  width: number; 
  height: number; 
  expanded: boolean;
  log: string | null;
} {
  const expandBy = CONFIG.grid.expandBy;
  
  // Calculate new dimensions (respecting max limits)
  let newWidth = Math.min(CONFIG.grid.maxWidth, currentWidth + expandBy);
  let newHeight = Math.min(CONFIG.grid.maxHeight, currentHeight + expandBy);
  
  // Check if any expansion is possible
  if (newWidth === currentWidth && newHeight === currentHeight) {
    return { grid, width: currentWidth, height: currentHeight, expanded: false, log: null };
  }
  
  // Create new expanded grid
  const newGrid: Cell[][] = [];
  
  for (let y = 0; y < newHeight; y++) {
    const row: Cell[] = [];
    for (let x = 0; x < newWidth; x++) {
      if (y < currentHeight && x < currentWidth) {
        // Copy existing cell
        row.push({ ...grid[y][x] });
      } else {
        // New cell - empty
        row.push({ food: false });
      }
    }
    newGrid.push(row);
  }
  
  // Spawn some food in the new areas to make expansion worthwhile
  const newCells = (newWidth * newHeight) - (currentWidth * currentHeight);
  const foodToAdd = Math.floor(newCells * 0.15); // 15% of new cells get food
  
  let foodAdded = 0;
  let attempts = 0;
  while (foodAdded < foodToAdd && attempts < 100) {
    attempts++;
    // Prefer new areas for food placement
    const x = Math.random() < 0.7 
      ? currentWidth + randomInt(newWidth - currentWidth) 
      : randomInt(newWidth);
    const y = Math.random() < 0.7 
      ? currentHeight + randomInt(newHeight - currentHeight) 
      : randomInt(newHeight);
    
    if (x < newWidth && y < newHeight && !newGrid[y][x].food && newGrid[y][x].agentId === undefined) {
      newGrid[y][x].food = true;
      foodAdded++;
    }
  }
  
  const log = `🌍 WORLD EXPANDED! Grid grew from ${currentWidth}×${currentHeight} to ${newWidth}×${newHeight}. +${foodAdded} food in new territory.`;
  
  return { 
    grid: newGrid, 
    width: newWidth, 
    height: newHeight, 
    expanded: true,
    log 
  };
}

/**
 * Update global grid dimensions (used by other functions)
 */
function updateGridDimensions(width: number, height: number) {
  GRID_WIDTH = width;
  GRID_HEIGHT = height;
}

function randomTraitId(): number {
  return randomInt(999999);
}

// Initial genes generator
function createRandomGenes(): Genes {
  return {
    foodPreference: CONFIG.genes.foodPreference.min + Math.random() * (CONFIG.genes.foodPreference.max - CONFIG.genes.foodPreference.min),
    exploration: CONFIG.genes.exploration.min + Math.random() * (CONFIG.genes.exploration.max - CONFIG.genes.exploration.min),
    reproductionThreshold: CONFIG.genes.reproductionThreshold.min + Math.random() * (CONFIG.genes.reproductionThreshold.max - CONFIG.genes.reproductionThreshold.min),
    mutationRate: CONFIG.genes.mutationRate.min + Math.random() * (CONFIG.genes.mutationRate.max - CONFIG.genes.mutationRate.min),
    traitId: randomTraitId(),
    // NEW: Learning & invention genes
    curiosity: CONFIG.genes.curiosity.min + Math.random() * (CONFIG.genes.curiosity.max - CONFIG.genes.curiosity.min),
    social: CONFIG.genes.social.min + Math.random() * (CONFIG.genes.social.max - CONFIG.genes.social.min),
    creativity: CONFIG.genes.creativity.min + Math.random() * (CONFIG.genes.creativity.max - CONFIG.genes.creativity.min),
    patience: CONFIG.genes.patience.min + Math.random() * (CONFIG.genes.patience.max - CONFIG.genes.patience.min),
  };
}

function createInitialAgents(grid: Cell[][]): Agent[] {
  const agents: Agent[] = [];
  let idCounter = 1;
  const taken: Set<string> = new Set();

  while (agents.length < INITIAL_AGENTS) {
    const y = randomInt(GRID_HEIGHT);
    const x = randomInt(GRID_WIDTH);
    const key = `${x},${y}`;
    if (taken.has(key) || grid[y][x].food) continue;
    taken.add(key);
    agents.push({
      id: idCounter++,
      x,
      y,
      energy: CONFIG.energy.initialMin + randomInt(CONFIG.energy.initialMax - CONFIG.energy.initialMin),
      genes: createRandomGenes(),
      memory: { qTable: {} },
      lastRule: "none",
      inventions: [],  // NEW: Start with no inventions
      inventionPoints: 0  // NEW: Start with no invention points
    });
  }
  return agents;
}

/**
 * Mutation helper: slightly nudge a value with probability
 * @param base - The base value to potentially mutate
 * @param mutationRate - Probability (0-1) that mutation occurs
 * @param magnitude - Maximum amount the value can change (both positive and negative)
 * @param min - Minimum allowed value after mutation
 * @param max - Maximum allowed value after mutation
 * @returns The potentially mutated value, clamped to [min, max]
 */
function mutateValue(
  base: number,
  mutationRate: number,
  magnitude: number,
  min: number,
  max: number
): number {
  let value = base;
  if (Math.random() < mutationRate) {
    const delta = (Math.random() * 2 - 1) * magnitude;
    value = Math.min(max, Math.max(min, value + delta));
  }
  return value;
}

// Genetic memory + random trait generation
function mutateGenes(parent: Genes): Genes {
  const mutationRate = parent.mutationRate;

  const foodPreference = mutateValue(
    parent.foodPreference,
    mutationRate,
    CONFIG.genes.mutation.foodPreferenceMagnitude,
    0.0,
    1.0
  );

  const exploration = mutateValue(
    parent.exploration,
    mutationRate,
    CONFIG.genes.mutation.explorationMagnitude,
    0.0,
    1.0
  );

  const reproductionThreshold = mutateValue(
    parent.reproductionThreshold,
    mutationRate,
    CONFIG.genes.mutation.reproductionThresholdMagnitude,
    CONFIG.genes.mutation.reproductionThresholdMin,
    CONFIG.genes.mutation.reproductionThresholdMax
  );

  // mutationRate itself can evolve
  const newMutationRate = mutateValue(
    parent.mutationRate,
    mutationRate,
    CONFIG.genes.mutation.mutationRateMagnitude,
    CONFIG.genes.mutation.mutationRateEvolveMin,
    CONFIG.genes.mutation.mutationRateEvolveMax
  );

  // NEW: Mutate learning & invention genes
  const curiosity = mutateValue(
    parent.curiosity,
    mutationRate,
    CONFIG.genes.mutation.curiosityMagnitude,
    0.0,
    1.0
  );

  const social = mutateValue(
    parent.social,
    mutationRate,
    CONFIG.genes.mutation.socialMagnitude,
    0.0,
    1.0
  );

  const creativity = mutateValue(
    parent.creativity,
    mutationRate,
    CONFIG.genes.mutation.creativityMagnitude,
    0.0,
    1.0
  );

  const patience = mutateValue(
    parent.patience,
    mutationRate,
    CONFIG.genes.mutation.patienceMagnitude,
    0.0,
    1.0
  );
  // sometimes spawn a totally new traitId => random trait generation
  const traitId =
    Math.random() < CONFIG.genes.mutation.newTraitChance ? randomTraitId() : parent.traitId;

  return {
    foodPreference,
    exploration,
    reproductionThreshold,
    mutationRate: newMutationRate,
    traitId,
    curiosity,
    social,
    creativity,
    patience,
  };
}

/**
 * BIODIVERSITY ENHANCEMENT: Calculate genetic distance between two agents
 * Higher distance = more genetically different = better for biodiversity
 */
function calculateGeneticDistance(genes1: Genes, genes2: Genes): number {
  const diffs = [
    Math.abs(genes1.foodPreference - genes2.foodPreference),
    Math.abs(genes1.exploration - genes2.exploration),
    Math.abs(genes1.curiosity - genes2.curiosity),
    Math.abs(genes1.social - genes2.social),
    Math.abs(genes1.creativity - genes2.creativity),
    Math.abs(genes1.patience - genes2.patience),
    Math.abs(genes1.mutationRate - genes2.mutationRate),
    genes1.traitId !== genes2.traitId ? 1 : 0, // Bonus for different trait families
  ];
  return diffs.reduce((sum, d) => sum + d, 0) / diffs.length;
}

/**
 * BIODIVERSITY ENHANCEMENT: Crossover genes from two parents
 * Creates offspring with mixed genes, favoring diversity
 */
function crossoverGenes(parent1: Genes, parent2: Genes, diversityBonus: number): Genes {
  // Boost mutation rate based on genetic distance (more diverse = more mutation)
  const baseMutationRate = (parent1.mutationRate + parent2.mutationRate) / 2;
  const boostedMutationRate = Math.min(0.6, baseMutationRate * (1 + diversityBonus));
  
  // For each gene, randomly select from either parent or blend
  const selectGene = (g1: number, g2: number): number => {
    const r = Math.random();
    if (r < 0.4) return g1;           // 40% from parent 1
    if (r < 0.8) return g2;           // 40% from parent 2
    return (g1 + g2) / 2;             // 20% blend
  };

  const baseGenes = {
    foodPreference: selectGene(parent1.foodPreference, parent2.foodPreference),
    exploration: selectGene(parent1.exploration, parent2.exploration),
    reproductionThreshold: selectGene(parent1.reproductionThreshold, parent2.reproductionThreshold),
    mutationRate: boostedMutationRate,
    curiosity: selectGene(parent1.curiosity, parent2.curiosity),
    social: selectGene(parent1.social, parent2.social),
    creativity: selectGene(parent1.creativity, parent2.creativity),
    patience: selectGene(parent1.patience, parent2.patience),
    // Higher chance of new trait when parents are diverse
    traitId: Math.random() < (CONFIG.genes.mutation.newTraitChance + diversityBonus * 0.3)
      ? randomTraitId()
      : Math.random() < 0.5 ? parent1.traitId : parent2.traitId,
  };

  // Apply mutations with boosted rate
  return {
    foodPreference: mutateValue(baseGenes.foodPreference, boostedMutationRate, CONFIG.genes.mutation.foodPreferenceMagnitude, 0, 1),
    exploration: mutateValue(baseGenes.exploration, boostedMutationRate, CONFIG.genes.mutation.explorationMagnitude, 0, 1),
    reproductionThreshold: mutateValue(baseGenes.reproductionThreshold, boostedMutationRate, CONFIG.genes.mutation.reproductionThresholdMagnitude, CONFIG.genes.mutation.reproductionThresholdMin, CONFIG.genes.mutation.reproductionThresholdMax),
    mutationRate: mutateValue(boostedMutationRate, boostedMutationRate, CONFIG.genes.mutation.mutationRateMagnitude, CONFIG.genes.mutation.mutationRateEvolveMin, CONFIG.genes.mutation.mutationRateEvolveMax),
    traitId: baseGenes.traitId,
    curiosity: mutateValue(baseGenes.curiosity, boostedMutationRate, CONFIG.genes.mutation.curiosityMagnitude, 0, 1),
    social: mutateValue(baseGenes.social, boostedMutationRate, CONFIG.genes.mutation.socialMagnitude, 0, 1),
    creativity: mutateValue(baseGenes.creativity, boostedMutationRate, CONFIG.genes.mutation.creativityMagnitude, 0, 1),
    patience: mutateValue(baseGenes.patience, boostedMutationRate, CONFIG.genes.mutation.patienceMagnitude, 0, 1),
  };
}

/**
 * BIODIVERSITY ENHANCEMENT: Find best mate nearby prioritizing genetic diversity
 * Returns the most genetically different nearby agent, or null for asexual reproduction
 */
function findBestMate(
  parent: Agent,
  allAgents: Agent[],
  mateSearchRadius: number = 3
): Agent | null {
  // Find nearby agents within mating range
  const nearbyAgents = allAgents.filter(a =>
    a.id !== parent.id &&
    a.energy > 5 && // Must have some energy
    Math.abs(a.x - parent.x) <= mateSearchRadius &&
    Math.abs(a.y - parent.y) <= mateSearchRadius
  );

  if (nearbyAgents.length === 0) return null;

  // Score each potential mate by genetic distance (higher = better for diversity)
  const scoredMates = nearbyAgents.map(mate => ({
    mate,
    distance: calculateGeneticDistance(parent.genes, mate.genes),
  }));

  // Sort by genetic distance (descending) - prefer more different mates
  scoredMates.sort((a, b) => b.distance - a.distance);

  // Probabilistically select mate, heavily favoring diverse genetics
  // 60% chance to pick most diverse, 25% second, 10% third, 5% random
  const r = Math.random();
  if (r < 0.60 && scoredMates.length >= 1) return scoredMates[0].mate;
  if (r < 0.85 && scoredMates.length >= 2) return scoredMates[1].mate;
  if (r < 0.95 && scoredMates.length >= 3) return scoredMates[2].mate;
  
  // Random selection from remaining
  return scoredMates[randomInt(scoredMates.length)].mate;
}

/**
 * CODE QUALITY IMPROVEMENT: Improved RL state representation
 * Now includes directional food information for better learning
 * instead of just binary food presence
 */
function getStateKey(agent: Agent, grid: Cell[][]): string {
  const { x, y, energy } = agent;

  let level: "low" | "mid" | "high";
  if (energy <= 6) level = "low";
  else if (energy <= 14) level = "mid";
  else level = "high";

  // Directional food detection for more informative state
  const foodUp = y > 0 && grid[y - 1][x].food ? 1 : 0;
  const foodDown = y < GRID_HEIGHT - 1 && grid[y + 1][x].food ? 1 : 0;
  const foodLeft = x > 0 && grid[y][x - 1].food ? 1 : 0;
  const foodRight = x < GRID_WIDTH - 1 && grid[y][x + 1].food ? 1 : 0;

  return `${level}_${foodUp}${foodDown}${foodLeft}${foodRight}`;
}

function qKey(stateKey: string, action: Direction): string {
  return `${stateKey}|${action}`;
}

function getQ(qTable: Record<string, number>, stateKey: string, action: Direction): number {
  return qTable[qKey(stateKey, action)] ?? 0;
}

function setQ(
  qTable: Record<string, number>,
  stateKey: string,
  action: Direction,
  value: number
): Record<string, number> {
  return { ...qTable, [qKey(stateKey, action)]: value };
}

const ALL_ACTIONS: Direction[] = ["up", "down", "left", "right", "stay"];

function bestActionAndValue(
  qTable: Record<string, number>,
  stateKey: string
): { action: Direction; value: number } {
  let bestAction: Direction = "stay";
  let bestValue = Number.NEGATIVE_INFINITY;
  for (const a of ALL_ACTIONS) {
    const v = getQ(qTable, stateKey, a);
    if (v > bestValue) {
      bestValue = v;
      bestAction = a;
    }
  }
  if (bestValue === Number.NEGATIVE_INFINITY) {
    return { action: "stay", value: 0 };
  }
  return { action: bestAction, value: bestValue };
}

function chooseAction(
  qTable: Record<string, number>,
  stateKey: string
): Direction {
  if (Math.random() < EPSILON) {
    return ALL_ACTIONS[randomInt(ALL_ACTIONS.length)];
  }
  return bestActionAndValue(qTable, stateKey).action;
}

// Decision logic: mix hard survival rule with RL exploration
function decideMove(
  agent: Agent,
  grid: Cell[][],
  stateKey: string
): { dir: Direction; rule: string; action: Direction } {
  const { x, y, energy, genes } = agent;
  const neighbors: { x: number; y: number; dir: Direction }[] = [
    { x, y: y - 1, dir: "up" as Direction },
    { x, y: y + 1, dir: "down" as Direction },
    { x: x - 1, y, dir: "left" as Direction },
    { x: x + 1, y, dir: "right" as Direction }
  ].filter(
    p => p.x >= 0 && p.x < GRID_WIDTH && p.y >= 0 && p.y < GRID_HEIGHT
  );

  const foodNeighbors = neighbors.filter(n => grid[n.y][n.x].food);

  const hungryThreshold = CONFIG.energy.hungryCutoffRatio * genes.foodPreference;

  // HARD RULE: if hungry + food adjacent, go for food
  if (energy <= hungryThreshold && foodNeighbors.length > 0) {
    const choice = foodNeighbors[randomInt(foodNeighbors.length)];
    return {
      dir: choice.dir,
      rule: "Rule 1: seek food (hard survival)",
      action: choice.dir
    };
  }

  // RL-DRIVEN CHOICE
  const action = chooseAction(agent.memory.qTable, stateKey);
  let ruleDesc = "RL: learned policy";
  if (action === "stay") {
    ruleDesc = "RL: choose stay";
  }

  return { dir: action, rule: ruleDesc, action };
}

/**
 * Check if an agent discovers a new invention this tick.
 * 
 * LIMITLESS DISCOVERY SYSTEM:
 * - Agents accumulate "invention points" through exploration and survival
 * - Points are spent to generate new, unique inventions
 * - Discovery is influenced by curiosity and creativity genes
 * - No cap on number of inventions - agents can discover infinitely
 * - Each invention is procedurally generated with effects scaled to agent's abilities
 */
function checkForDiscovery(
  agent: Agent, 
  tick: number
): { invention: Invention | null; updatedAgent: Agent } {
  // Agents gain invention points based on curiosity and exploration
  // Points represent accumulated knowledge, experience, and inspiration
  const pointGain = agent.genes.curiosity * agent.genes.exploration * 0.5;
  let newPoints = agent.inventionPoints + pointGain;
  
  // Only agents with enough energy can invent (cognitive surplus)
  if (agent.energy < 15) {
    return { 
      invention: null, 
      updatedAgent: { ...agent, inventionPoints: newPoints }
    };
  }
  
  // Discovery chance increases with curiosity and creativity
  const creativityBoost = agent.genes.creativity;
  const discoveryChance = agent.genes.curiosity * 0.03 * (1 + creativityBoost);
  
  // Also consider accumulated invention points as inspiration
  const inspirationBonus = Math.min(0.02, newPoints * 0.001);
  const totalChance = discoveryChance + inspirationBonus;
  
  if (Math.random() > totalChance) {
    return { 
      invention: null, 
      updatedAgent: { ...agent, inventionPoints: newPoints }
    };
  }
  
  // Discovery! Generate a unique invention
  const inventionNumber = agent.inventions.length + 1;
  const invention = generateInvention(agent, tick, inventionNumber);
  
  // Spend some invention points on the discovery (but not all)
  newPoints = Math.max(0, newPoints - 5);
  
  return { 
    invention, 
    updatedAgent: { ...agent, inventionPoints: newPoints }
  };
}

/**
 * Calculate the actual energy cost for movement based on inventions.
 */
function getMovementCost(agent: Agent): number {
  let cost = CONFIG.simulation.baseEnergyCost; // Use config base cost
  
  // Check for energy efficiency inventions
  for (const inv of agent.inventions) {
    if (inv.effect.type === 'energy_efficiency') {
      cost *= inv.effect.multiplier;
    }
  }
  
  return Math.max(0.25, cost); // Minimum cost of 0.25
}

/**
 * Get the food detection range for an agent.
 * FUTURE: This helper is ready but not yet integrated into decision-making logic.
 * In Phase 3, this would be used to detect food beyond adjacent cells.
 */
function getFoodDetectionRange(agent: Agent): number {
  let range = 1; // Base range (adjacent cells)
  
  for (const inv of agent.inventions) {
    if (inv.effect.type === 'food_detection_range') {
      range = Math.max(range, inv.effect.range);
    }
  }
  
  return range;
}

/**
 * Get reproduction bonus from inventions.
 */
function getReproductionBonus(agent: Agent): number {
  let bonus = 0;
  
  for (const inv of agent.inventions) {
    if (inv.effect.type === 'reproduction_boost') {
      bonus += inv.effect.bonus;
    }
  }
  
  return bonus;
}

/**
 * Apply one simulation step, including reproduction + mutation + RL updates
 * 
 * CRITICAL BUG FIX #2: Added collision detection
 * Now tracks intended moves and prevents multiple agents from occupying the same cell
 * 
 * PHASE 1 UPGRADE: Physics integration
 * - Movement costs affected by physics discoveries
 * - Agents can discover physics concepts
 * - Nearby agents provide collaboration bonuses
 * 
 * PHASE 2 UPGRADE: Mathematics integration
 * - Resource optimization from math knowledge
 * - Improved Q-learning from statistical concepts
 * - Math discovery with collaboration bonus
 * 
 * UNLIMITED EVOLUTION UPGRADE:
 * - Procedural concept generation when base concepts exhausted
 * - No caps on bonuses or discoveries
 */
function stepWorld(
  agents: Agent[],
  grid: Cell[][],
  tick: number,
  physicsState: CivilizationPhysics,
  mathState: CivilizationMath,
  generatedPhysicsLevel: number = 0,
  generatedMathLevel: number = 0
): { 
  agents: Agent[]; 
  grid: Cell[][]; 
  log: string[]; 
  discoveries: DiscoveryEvent[];
  physicsDiscoveries: PhysicsConcept[];
  mathDiscoveries: MathConcept[];
  nextGeneratedPhysicsLevel?: number;
  nextGeneratedMathLevel?: number;
} {
  const newGrid: Cell[][] = grid.map(row =>
    row.map(cell => ({ ...cell, agentId: undefined }))
  );

  const logs: string[] = [];
  const updatedAgents: Agent[] = [];
  const discoveries: DiscoveryEvent[] = [];  const physicsDiscoveries: PhysicsConcept[] = []; // Track new physics concepts discovered this tick
  const mathDiscoveries: MathConcept[] = []; // Track new math concepts discovered this tick
  
  // Track procedural generation level advances
  let nextPhysicsLevel = generatedPhysicsLevel;
  let nextMathLevel = generatedMathLevel;

  let nextId = agents.reduce((max, a) => Math.max(max, a.id), 0) + 1;

  /**
   * CRITICAL BUG FIX #2: Agent collision handling
   * Track intended destinations to prevent multiple agents from moving to the same cell
   */
  const destinationMap = new Map<string, number>(); // "x,y" -> agentId

  // Phase 1: Decide moves for all agents
  interface AgentMove {
    agent: Agent;
    newPos: { x: number; y: number };
    decision: { dir: Direction; rule: string; action: Direction };
    stateKey: string;
  }
  const agentMoves: AgentMove[] = [];

  for (const agent of agents) {
    if (agent.energy <= 0) continue;

    // RL state before move
    const stateKey = getStateKey(agent, grid);
    const decision = decideMove(agent, grid, stateKey);

    // Use type-safe direction application
    const newPos = applyDirection(agent.x, agent.y, decision.dir);

    agentMoves.push({ agent, newPos, decision, stateKey });
  }

  // Phase 2: Process moves with collision detection
  for (const { agent, newPos, decision, stateKey } of agentMoves) {
    const destKey = `${newPos.x},${newPos.y}`;
    
    // Track the final position (may change due to collision)
    let finalX = newPos.x;
    let finalY = newPos.y;
    
    // Check if another agent already claimed this destination
    if (destinationMap.has(destKey)) {
      // Collision detected - agent stays in place
      logs.push(
        `Agent ${agent.id} collision at (${newPos.x},${newPos.y}), stayed at (${agent.x},${agent.y})`
      );
      // Use original position
      finalX = agent.x;
      finalY = agent.y;
    } else {
      destinationMap.set(destKey, agent.id);
    }

    // Apply energy cost with invention AND physics effects
    const baseMovementCost = getMovementCost(agent);
    const movementCost = getPhysicsMovementCost(agent, baseMovementCost, physicsState.unlockedConcepts);
    let newEnergy = agent.energy - movementCost;
    const cell = newGrid[finalY][finalX];
    let ateFood = false;

    if (cell.food) {
      ateFood = true;
      cell.food = false;
      newEnergy += CONFIG.simulation.foodEnergyBonus;
    }

    let reward = -1;
    if (ateFood) reward += CONFIG.simulation.foodEnergyBonus;

    let parentAgent: Agent = {
      ...agent,
      x: finalX,
      y: finalY,
      energy: newEnergy,
      lastRule: decision.rule
    };

    // Check for invention discovery
    const discoveryResult = checkForDiscovery(parentAgent, tick);
    parentAgent = discoveryResult.updatedAgent; // Update with new invention points
    
    if (discoveryResult.invention) {
      parentAgent = {
        ...parentAgent,
        inventions: [...parentAgent.inventions, discoveryResult.invention]
      };
      discoveries.push({
        tick,
        agentId: parentAgent.id,
        invention: discoveryResult.invention
      });
      logs.push(
        `Agent ${parentAgent.id} discovered ${discoveryResult.invention.name}! (${discoveryResult.invention.description})`
      );
    }

    // PHYSICS DISCOVERY - Check if agent discovers a physics concept
    // Find nearby agents for collaboration bonus
    const nearbyAgentsForPhysics = agents.filter(a =>
      a.id !== agent.id &&
      Math.abs(a.x - finalX) <= 2 &&
      Math.abs(a.y - finalY) <= 2
    );
      const physicsResult = checkPhysicsDiscovery(
      parentAgent,
      nearbyAgentsForPhysics,
      physicsState.unlockedConcepts,
      tick,
      nextPhysicsLevel
    );
    
    if (physicsResult.concept) {
      physicsDiscoveries.push(physicsResult.concept);
      if (physicsResult.log) {
        logs.push(physicsResult.log);
      }
      // Update procedural level if advanced
      if (physicsResult.nextGeneratedLevel !== undefined) {
        nextPhysicsLevel = physicsResult.nextGeneratedLevel;
      }
    }

    // MATH DISCOVERY - Check if agent discovers a math concept
    const mathResult = checkMathDiscovery(
      parentAgent,
      nearbyAgentsForPhysics, // Same nearby agents for collaboration
      mathState.unlockedConcepts,
      tick,
      nextMathLevel
    );
    
    if (mathResult.concept) {
      mathDiscoveries.push(mathResult.concept);
      if (mathResult.log) {
        logs.push(mathResult.log);
      }
      // Update procedural level if advanced
      if (mathResult.nextGeneratedLevel !== undefined) {
        nextMathLevel = mathResult.nextGeneratedLevel;
      }
    }

    // REPRODUCTION - BIODIVERSITY ENHANCED
    const reproThreshold = parentAgent.genes.reproductionThreshold;
    let reproduced = false;

    // Apply reproduction bonus from inventions
    const reproBonus = getReproductionBonus(parentAgent);
    const effectiveEnergy = parentAgent.energy + reproBonus;

    if (effectiveEnergy > reproThreshold) {
      const neighborSpots = [
        { x: finalX, y: finalY - 1 },
        { x: finalX, y: finalY + 1 },
        { x: finalX - 1, y: finalY },
        { x: finalX + 1, y: finalY }
      ].filter(
        p =>
          p.x >= 0 &&
          p.x < GRID_WIDTH &&
          p.y >= 0 &&
          p.y < GRID_HEIGHT &&
          newGrid[p.y][p.x].agentId === undefined &&
          !destinationMap.has(`${p.x},${p.y}`) // Also check collision map
      );

      if (neighborSpots.length > 0) {
        const spot = neighborSpots[randomInt(neighborSpots.length)];

        const childEnergy = Math.floor(parentAgent.energy / 2);
        parentAgent = { ...parentAgent, energy: parentAgent.energy - childEnergy };

        // BIODIVERSITY: Try to find a mate for sexual reproduction
        const mate = findBestMate(parentAgent, agents);
        let childGenes: Genes;
        let reproductionType: string;
        let diversityBonus = 0;
        
        if (mate) {
          // Sexual reproduction with crossover - prioritizes genetic diversity
          diversityBonus = calculateGeneticDistance(parentAgent.genes, mate.genes);
          childGenes = crossoverGenes(parentAgent.genes, mate.genes, diversityBonus);
          reproductionType = diversityBonus > 0.3 
            ? `Sexual (high diversity: ${(diversityBonus * 100).toFixed(0)}%)` 
            : `Sexual (mate: Agent ${mate.id})`;
        } else {
          // Asexual reproduction with normal mutation
          childGenes = mutateGenes(parentAgent.genes);
          reproductionType = 'Asexual (no nearby mates)';
        }
        
        // Inheritance: children can inherit parent's inventions based on social gene
        const inheritedInventions = parentAgent.inventions.filter(
          inv => Math.random() < parentAgent.genes.social * 0.8
        );
        
        // If sexual reproduction, also chance to inherit from mate
        if (mate) {
          const mateInventions = mate.inventions.filter(
            inv => Math.random() < mate.genes.social * 0.5 && 
                   !inheritedInventions.some(i => i.id === inv.id)
          );
          inheritedInventions.push(...mateInventions);
        }
        
        // Children inherit a portion of parent's invention points (learning from parent)
        let inheritedPoints = parentAgent.inventionPoints * parentAgent.genes.social * 0.3;
        if (mate) {
          // Also inherit some from mate
          inheritedPoints += mate.inventionPoints * mate.genes.social * 0.2;
        }
        
        const child: Agent = {
          id: nextId++,
          x: spot.x,
          y: spot.y,
          energy: childEnergy,
          genes: childGenes,
          memory: { qTable: {} },
          lastRule: `Born (${reproductionType})`,
          inventions: inheritedInventions.map(inv => ({
            ...inv,
            // Mark as inherited, not discovered by this agent
            discoveredBy: parentAgent.id,
          })),
          inventionPoints: inheritedPoints,
        };

        newGrid[spot.y][spot.x].agentId = child.id;
        destinationMap.set(`${spot.x},${spot.y}`, child.id); // Register child position
        updatedAgents.push(child);

        reward += CONFIG.simulation.reproductionReward;
        reproduced = true;

        const inheritMsg = inheritedInventions.length > 0 
          ? `, inherited ${inheritedInventions.length} inventions`
          : '';
        const diversityMsg = diversityBonus > 0.2 
          ? ` 🧬 HIGH DIVERSITY!` 
          : '';
        logs.push(
          `Agent ${parentAgent.id} reproduced (${reproductionType}): child ${child.id} at (${spot.x},${spot.y}) with traitId ${child.genes.traitId}${inheritMsg}${diversityMsg}`
        );
      }
    }

    // RL UPDATE
    const newStateKey = getStateKey(
      { ...parentAgent, x: finalX, y: finalY },
      newGrid
    );
    const oldQ = getQ(parentAgent.memory.qTable, stateKey, decision.action);
    const bestNext = bestActionAndValue(parentAgent.memory.qTable, newStateKey).value;
    const updatedQ =
      (1 - ALPHA) * oldQ + ALPHA * (reward + GAMMA * bestNext);
    const newQTable = setQ(
      parentAgent.memory.qTable,
      stateKey,
      decision.action,
      updatedQ
    );

    parentAgent = { ...parentAgent, memory: { qTable: newQTable } };

    if (parentAgent.energy > 0) {
      updatedAgents.push(parentAgent);
      newGrid[parentAgent.y][parentAgent.x].agentId = parentAgent.id;

      logs.push(
        `Agent ${parentAgent.id} used ${decision.rule}, moved to (${parentAgent.x},${parentAgent.y})` +          (ateFood ? " and ate food (+5 energy)" : "") +
          (reproduced ? " and reproduced (+2 reward)" : "") +
          `, energy now ${parentAgent.energy}, traitId=${parentAgent.genes.traitId}`
      );
    } else {
      reward -= CONFIG.simulation.deathPenalty;
      logs.push(
        `Agent ${agent.id} ran out of energy at (${finalX},${finalY}) and was removed.`
      );
    }
  }

  /**
   * CRITICAL BUG FIX #1: Food spawning bug
   * Previously, placeRandomFood returned a new grid but the result was discarded.
   * Now we properly use the returned grid to ensure food actually appears.
   */
  if (Math.random() < CONFIG.simulation.foodSpawnChance) {
    const gridWithFood = placeRandomFood(newGrid, CONFIG.simulation.foodSpawnCount);
    return { 
      agents: updatedAgents, 
      grid: gridWithFood, 
      log: logs, 
      discoveries, 
      physicsDiscoveries, 
      mathDiscoveries,
      nextGeneratedPhysicsLevel: nextPhysicsLevel,
      nextGeneratedMathLevel: nextMathLevel
    };
  }

  return { 
    agents: updatedAgents, 
    grid: newGrid, 
    log: logs, 
    discoveries, 
    physicsDiscoveries, 
    mathDiscoveries,
    nextGeneratedPhysicsLevel: nextPhysicsLevel,
    nextGeneratedMathLevel: nextMathLevel
  };
}

/**
 * PERFORMANCE IMPROVEMENT: Memoize trait color generation
 * Cache colors instead of recalculating on every render
 */
const traitColorCache = new Map<number, string>();

// Constants for color generation
const TRAIT_COLOR_SATURATION = 70;
const TRAIT_COLOR_LIGHTNESS = 55;

function colorForTrait(traitId: number): string {
  if (!traitColorCache.has(traitId)) {
    const hue = traitId % 360;
    traitColorCache.set(traitId, `hsl(${hue}, ${TRAIT_COLOR_SATURATION}%, ${TRAIT_COLOR_LIGHTNESS}%)`);
  }
  return traitColorCache.get(traitId)!;
}

// Simple line chart for population over time
const PopulationChart: React.FC<{ history: HistoryPoint[] }> = ({ history }) => {
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

// Bar chart for trait distribution
const TraitChart: React.FC<{
  last: HistoryPoint | null;
  watchedTraitId: number | null;
  onSelectTrait: (traitId: number) => void;
}> = ({ last, watchedTraitId, onSelectTrait }) => {
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

// Discovery timeline showing recent inventions with details
const DiscoveryTimeline: React.FC<{ discoveries: DiscoveryEvent[] }> = ({ discoveries }) => {
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

// Invention Statistics showing creativity metrics
const InventionStats: React.FC<{ 
  agents: Agent[];
  discoveries: DiscoveryEvent[];
}> = ({ agents, discoveries }) => {
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

// Download world state as JSON ("guardar"/"speichere")
function downloadWorld(state: WorldState) {
  const blob = new Blob([JSON.stringify(state)], {
    type: "application/json"
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `world_tick_${state.tick}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

/**
 * REFACTOR: Shared initialization function
 * Creates a fresh world state with food and agents to avoid duplication
 */
function initializeWorld(): { grid: Cell[][]; agents: Agent[] } {
  const empty = createEmptyGrid();
  const withFood = placeRandomFood(empty, INITIAL_FOOD);
  const agents = createInitialAgents(withFood);
  return { grid: withFood, agents };
}

const App: React.FC = () => {
  /**
   * CRITICAL BUG FIX #3: Initial grid state mismatch
   * Previously, agents were created using a different empty grid than the one with food,
   * which could cause agents to spawn on food cells.
   * Now we use a shared initialization function to ensure consistency.
   */
  const initialWorld = useMemo(() => initializeWorld(), []);
  
  const [grid, setGrid] = useState<Cell[][]>(initialWorld.grid);
  const [agents, setAgents] = useState<Agent[]>(initialWorld.agents);

  // Dynamic grid dimensions state
  const [gridWidth, setGridWidth] = useState<number>(CONFIG.grid.initialWidth);
  const [gridHeight, setGridHeight] = useState<number>(CONFIG.grid.initialHeight);

  const [log, setLog] = useState<string[]>([]);
  const [selectedAgentId, setSelectedAgentId] = useState<number | null>(null);
  const [tick, setTick] = useState(0);
  const [history, setHistory] = useState<HistoryPoint[]>([]);
  const [discoveries, setDiscoveries] = useState<DiscoveryEvent[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [speedMs, setSpeedMs] = useState(400);
  const [watchedTraitId, setWatchedTraitId] = useState<number | null>(null);  const [showStartupModal, setShowStartupModal] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);

  // Communication system state
  const [communicationLog, setCommunicationLog] = useState<CommunicationLog>(initializeCommunicationLog());
  const [communicationStates, setCommunicationStates] = useState<Map<number, CommunicationState>>(new Map());
  const [chatOpen, setChatOpen] = useState(false);
  const [analysisOpen, setAnalysisOpen] = useState(false);
  
  // Environmental challenges state
  const [challengeState, setChallengeState] = useState<ChallengeState>(initializeChallengeState());

  // Physics integration state
  const [physicsState, setPhysicsState] = useState<CivilizationPhysics>(initializeCivilizationPhysics());
  // Mathematics integration state
  const [mathState, setMathState] = useState<CivilizationMath>(initializeCivilizationMath());

  // Procedural generation levels for unlimited evolution
  const [generatedMathLevel, setGeneratedMathLevel] = useState<number>(0);
  const [generatedPhysicsLevel, setGeneratedPhysicsLevel] = useState<number>(0);

  // Era progression state
  const [eraState, setEraState] = useState<CivilizationEraState>(initializeCivilizationEra());

  // Track collaborative discoveries for milestones
  const [collaborativeDiscoveries, setCollaborativeDiscoveries] = useState<number>(0);
  // Speech and language integration state
  const [speechState, setSpeechState] = useState<CivilizationSpeechState>(initializeCivilizationSpeech());
  const [agentLanguages, setAgentLanguages] = useState<Map<number, AgentLanguageState>>(new Map());  // Internet Learning State (persists across component updates)
  const [internetKnowledgeLearned, setInternetKnowledgeLearned] = useState<Set<string>>(new Set());
  const [totalInternetLearning, setTotalInternetLearning] = useState<number>(0);
  const [autoLearningEnabled, setAutoLearningEnabled] = useState<boolean>(true);
  const [recentAutoLearning, setRecentAutoLearning] = useState<AutoLearningResult[]>([]);

  // Autonomous Evolution State - tracks self-directed learning for each agent
  const [autonomousStates, setAutonomousStates] = useState<Map<number, AutonomousState>>(new Map());
  const [totalAutonomousConcepts, setTotalAutonomousConcepts] = useState<number>(0);
  const [totalAutonomousBehaviors, setTotalAutonomousBehaviors] = useState<number>(0);
  const [mostAutonomousAgent, setMostAutonomousAgent] = useState<{ id: number; level: string } | null>(null);

  // Track self-aware agents for communication
  const selfAwareAgentIds = useMemo(() => {
    return agents.filter(agent => {
      const consciousnessScore = 
        (agent.genes.curiosity * 25) + 
        (agent.genes.creativity * 25) + 
        (agent.genes.social * 20) +
        (agent.inventionPoints * 0.5) +
        (agent.inventions.length * 10);
      return consciousnessScore >= 60;
    }).map(a => a.id);
  }, [agents]);

  // Handler for user sending messages to agents
  const handleSendUserMessage = useCallback((content: string, targetAgentId?: number) => {
    const userMessage = createUserMessage(content, tickRef.current, targetAgentId);
    
    // Get agents to respond to
    const respondingAgents = targetAgentId 
      ? agents.filter(a => a.id === targetAgentId && selfAwareAgentIds.includes(a.id))
      : agents.filter(a => selfAwareAgentIds.includes(a.id));
    
    if (respondingAgents.length === 0) return;
    
    // Generate responses from agents and apply effects
    const responses: AgentMessage[] = [];
    let updatedAgents = [...agents];
    
    for (const agent of respondingAgents) {
      // Generate response
      const response = generateAgentResponse(agent, userMessage, tickRef.current);
      responses.push(response);
      
      // Apply positive effects to the agent (consciousness boost from creator interaction)
      const agentIndex = updatedAgents.findIndex(a => a.id === agent.id);
      if (agentIndex !== -1) {
        updatedAgents[agentIndex] = applyUserMessageEffects(updatedAgents[agentIndex], content);
      }
    }
    
    // Update agents with boosted stats
    setAgents(updatedAgents);
      // Add messages to communication log
    setCommunicationLog(prev => ({
      ...prev,
      messages: [...prev.messages, userMessage, ...responses],
      totalMessages: prev.totalMessages + 1 + responses.length,
      userMessages: [...(prev.userMessages || []), userMessage],
    }));
    
    // Log the interaction
    const effectMsg = respondingAgents.length === 1 
      ? `Agent ${respondingAgents[0].id} received enlightenment from the Creator!`
      : `${respondingAgents.length} agents received enlightenment from the Creator!`;
    setLog(prev => [effectMsg, ...prev].slice(0, 80));
  }, [agents, selfAwareAgentIds]);

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const tickRef = useRef(0);
  tickRef.current = tick;

  /**
   * PERFORMANCE IMPROVEMENT: Avoid repeated agent lookups
   * Build a lookup Map once using useMemo instead of doing O(n) lookup for every cell
   */
  const agentMap = useMemo(() => {
    const map = new Map<number, Agent>();
    agents.forEach(a => map.set(a.id, a));
    return map;
  }, [agents]);
  const renderedGrid = useMemo(() => {
    const copy: Cell[][] = grid.map(row => row.map(cell => ({ ...cell, agentId: undefined as number | undefined })));
    for (const agent of agents) {
      if (
        agent.x >= 0 &&
        agent.x < GRID_WIDTH &&
        agent.y >= 0 &&
        agent.y < GRID_HEIGHT
      ) {
        copy[agent.y][agent.x].agentId = agent.id;
      }
    }
    return copy;
  }, [grid, agents]);

  const selectedAgent = useMemo(
    () => agents.find(a => a.id === selectedAgentId) || null,
    [agents, selectedAgentId]
  );

  const pushHistory = useCallback((newAgents: Agent[], newTick: number) => {
    const byTrait: Record<number, number> = {};
    for (const a of newAgents) {
      byTrait[a.genes.traitId] = (byTrait[a.genes.traitId] || 0) + 1;
    }
    const point: HistoryPoint = {
      tick: newTick,
      totalAgents: newAgents.length,
      byTrait
    };
    setHistory(prev => [...prev, point].slice(-60));
  }, []);  const handleStep = useCallback(() => {
    const currentTick = tickRef.current;
    const newTick = currentTick + 1;
      // Process environmental challenges
    const { state: updatedChallengeState, logs: challengeLogs } = processChallenges(
      challengeState,
      currentTick,
      agents.length
    );
    setChallengeState(updatedChallengeState);
      // Run the simulation step with physics and math integration
    const { 
      agents: newAgents, 
      grid: newGrid, 
      log: newLog, 
      discoveries: newDiscoveries,
      physicsDiscoveries: newPhysicsDiscoveries,
      mathDiscoveries: newMathDiscoveries,
      nextGeneratedPhysicsLevel,
      nextGeneratedMathLevel
    } = stepWorld(
      agents,
      renderedGrid,
      currentTick,
      physicsState,
      mathState,
      generatedPhysicsLevel,
      generatedMathLevel
    );
    
    // Update procedural generation levels for unlimited evolution
    if (nextGeneratedPhysicsLevel !== undefined && nextGeneratedPhysicsLevel > generatedPhysicsLevel) {
      setGeneratedPhysicsLevel(nextGeneratedPhysicsLevel);
    }
    if (nextGeneratedMathLevel !== undefined && nextGeneratedMathLevel > generatedMathLevel) {
      setGeneratedMathLevel(nextGeneratedMathLevel);
    }
    
    // Update physics state with any new discoveries
    if (newPhysicsDiscoveries.length > 0) {
      setPhysicsState(prev => ({
        ...prev,
        unlockedConcepts: [...prev.unlockedConcepts, ...newPhysicsDiscoveries],
        totalDiscoveries: prev.totalDiscoveries + newPhysicsDiscoveries.length,
        lastDiscoveryTick: currentTick
      }));
    }
    
    // Update math state with any new discoveries
    if (newMathDiscoveries.length > 0) {
      setMathState(prev => ({
        ...prev,
        unlockedConcepts: [...prev.unlockedConcepts, ...newMathDiscoveries],
        totalDiscoveries: prev.totalDiscoveries + newMathDiscoveries.length,
        lastDiscoveryTick: currentTick
      }));
    }
    
    // ========================================
    // PHASE 3: ERA PROGRESSION INTEGRATION
    // ========================================
    
    // Calculate current physics and math counts (include new discoveries)
    const currentPhysicsCount = physicsState.unlockedConcepts.length + newPhysicsDiscoveries.length;
    const currentMathCount = mathState.unlockedConcepts.length + newMathDiscoveries.length;
    const totalInventionCount = newDiscoveries.length + discoveries.length;
    
    // Count new collaborative discoveries (discoveries made with nearby agents)
    // This is tracked from the log messages that mention "collaborated"
    const newCollabCount = newLog.filter(l => l.includes('collaborated')).length;
    if (newCollabCount > 0) {
      setCollaborativeDiscoveries(prev => prev + newCollabCount);
    }
    
    // Update progression metrics
    const newDiscoveryCount = newPhysicsDiscoveries.length + newMathDiscoveries.length + newDiscoveries.length;
    
    // Check for era advancement
    const eraAdvanceResult = checkEraAdvancement(
      eraState,
      currentPhysicsCount,
      currentMathCount,
      totalInventionCount,
      currentTick
    );
    
    // Check for new milestones
    const newMilestones = checkMilestones(
      eraState,
      currentPhysicsCount,
      currentMathCount,
      totalInventionCount,
      newAgents.length,
      collaborativeDiscoveries + newCollabCount,
      currentTick
    );
    
    // Update era state
    if (eraAdvanceResult.shouldAdvance || newMilestones.length > 0 || newDiscoveryCount > 0) {
      setEraState(prev => {
        let newState = { ...prev };
        
        // Update metrics
        newState.metrics = updateProgressionMetrics(prev.metrics, newDiscoveryCount, currentTick);
        
        // Add new milestones
        if (newMilestones.length > 0) {
          newState.milestones = [...prev.milestones, ...newMilestones];
          // Log milestone achievements
          for (const milestone of newMilestones) {
            challengeLogs.push(`🏆 Milestone achieved: ${milestone.name}!`);
          }
        }
        
        // Advance era if requirements met
        if (eraAdvanceResult.shouldAdvance && eraAdvanceResult.nextEra) {
          newState.currentEra = eraAdvanceResult.nextEra;
          newState.allEras = [...prev.allEras, eraAdvanceResult.nextEra];
          newState.lastEraAdvanceTick = currentTick;
          newState.metrics = {
            ...newState.metrics,
            currentEraLevel: eraAdvanceResult.nextEra.level,
            ticksSinceLastEra: 0
          };
          if (eraAdvanceResult.log) {
            challengeLogs.push(eraAdvanceResult.log);
          }
        }
        
        return newState;
      });
    }
    
    // ========================================
    // PHASE 4: SPEECH & LANGUAGE INTEGRATION
    // ========================================
    
    // Update agent language states
    const newAgentLanguages = new Map(agentLanguages);
    const physicsVocab = getPhysicsVocabulary(physicsState.unlockedConcepts);
    const mathVocab = getMathVocabulary(mathState.unlockedConcepts);
    
    for (const agent of newAgents) {
      // Initialize language for new agents
      if (!newAgentLanguages.has(agent.id)) {
        // Try to find parent's language for inheritance
        const parentLang = agent.lastRule?.includes('Born') 
          ? undefined // Could be enhanced to track actual parent
          : undefined;
        newAgentLanguages.set(agent.id, initializeAgentLanguage(agent, parentLang));
      }
      
      // Expand vocabulary based on consciousness and discoveries
      const agentLang = newAgentLanguages.get(agent.id);
      if (agentLang) {
        const consciousnessScore = 
          (agent.genes.curiosity * 25) + 
          (agent.genes.creativity * 25) + 
          (agent.genes.social * 20) +
          (agent.inventionPoints * 0.5) +
          (agent.inventions.length * 10);
        
        // Expand vocabulary
        const expandedVocab = expandVocabulary(
          agentLang.vocabulary,
          consciousnessScore,
          physicsVocab,
          mathVocab,
          eraState.currentEra.level,
          currentTick
        );
        
        // Update communication style
        const newStyle = updateCommunicationStyle(
          { ...agentLang, vocabulary: expandedVocab },
          consciousnessScore
        );
        
        newAgentLanguages.set(agent.id, {
          ...agentLang,
          vocabulary: expandedVocab,
          vocabularySize: expandedVocab.length,
          languageComplexity: expandedVocab.reduce((sum, w) => sum + w.complexity, 0) / expandedVocab.length,
          communicationStyle: newStyle
        });
      }
    }
    
    // Process agent-to-agent dialogues
    let newSpeechState = { ...speechState };
    const activeDialogues = [...newSpeechState.activeDialogues];
    
    // Check for new dialogue opportunities between nearby agents
    for (let i = 0; i < newAgents.length; i++) {
      for (let j = i + 1; j < newAgents.length; j++) {
        const agent1 = newAgents[i];
        const agent2 = newAgents[j];
        const lang1 = newAgentLanguages.get(agent1.id);
        const lang2 = newAgentLanguages.get(agent2.id);
        
        if (lang1 && lang2) {
          // Check if they can initiate dialogue
          if (canInitiateDialogue(agent1, agent2, lang1, lang2, currentTick)) {
            // 5% chance to start a dialogue when conditions are met
            if (Math.random() < 0.05) {
              const dialogue = initiateDialogue(agent1, agent2, lang1, lang2, currentTick);
              activeDialogues.push(dialogue);
              challengeLogs.push(`💬 Agent ${agent1.id} started conversation with Agent ${agent2.id}`);
            }
          }
        }
      }
    }
    
    // Continue existing dialogues
    const updatedDialogues = activeDialogues.map(dialogue => {
      if (currentTick - dialogue.endTick > 3) {
        // Dialogue has ended
        return dialogue;
      }
      
      const speaker = newAgents.find(a => a.id === dialogue.responderId);
      const listener = newAgents.find(a => a.id === dialogue.initiatorId);
      const speakerLang = newAgentLanguages.get(dialogue.responderId);
      const listenerLang = newAgentLanguages.get(dialogue.initiatorId);
      
      if (speaker && listener && speakerLang && listenerLang && Math.random() < 0.3) {
        return continueDialogue(
          dialogue,
          speaker,
          listener,
          speakerLang,
          listenerLang,
          currentTick,
          physicsState.unlockedConcepts.map(c => c.id),
          mathState.unlockedConcepts.map(c => c.id)
        );
      }
      return dialogue;
    });
    
    // Update shared vocabulary
    const allWords = new Map<string, number>();
    for (const lang of newAgentLanguages.values()) {
      for (const word of lang.vocabulary) {
        allWords.set(word.word, (allWords.get(word.word) || 0) + 1);
      }
    }
    const sharedWords = speechState.sharedVocabulary.filter(
      w => (allWords.get(w.word) || 0) >= 2
    );
    
    // Calculate language evolution level based on average complexity and vocabulary
    const totalComplexity = Array.from(newAgentLanguages.values())
      .reduce((sum, lang) => sum + lang.languageComplexity, 0);
    const avgComplexity = newAgentLanguages.size > 0 ? totalComplexity / newAgentLanguages.size : 1;
    const evolutionLevel = Math.min(10, Math.floor(avgComplexity));
    
    newSpeechState = {
      ...newSpeechState,
      activeDialogues: updatedDialogues.filter(d => currentTick - d.endTick <= 5),
      completedDialogues: newSpeechState.completedDialogues + updatedDialogues.filter(d => currentTick - d.endTick > 5).length,
      totalWordsKnown: allWords.size,
      sharedVocabulary: sharedWords,
      languageEvolutionLevel: evolutionLevel,
      knowledgeTransferCount: updatedDialogues.reduce((sum, d) => sum + d.knowledgeTransferred.length, 0)
    };
    
    setAgentLanguages(newAgentLanguages);
    setSpeechState(newSpeechState);
    
    // ========================================
    // PHASE 5: AUTONOMOUS EVOLUTION INTEGRATION
    // ========================================
    
    // Update autonomous states for all agents
    const newAutonomousStates = new Map(autonomousStates);
    let newTotalConcepts = 0;
    let newTotalBehaviors = 0;
    let highestAutonomy: { id: number; level: string; score: number } | null = null;
    
    for (const agent of newAgents) {
      // Calculate agent intelligence (same formula as consciousness)
      const intelligence = 
        (agent.genes.curiosity * 25) + 
        (agent.genes.creativity * 25) + 
        (agent.genes.social * 20) +
        (agent.inventionPoints * 0.5) +
        (agent.inventions.length * 10) +
        (physicsState.unlockedConcepts.length * 2) +
        (mathState.unlockedConcepts.length * 2);
      
      // Check if agent qualifies for autonomy
      if (hasAutonomy(intelligence)) {
        // Get or initialize autonomous state
        let autoState = newAutonomousStates.get(agent.id);
        if (!autoState) {
          autoState = initializeAutonomousState(agent, intelligence);
        }
        
        // Create experience memory from recent action
        const recentExp: ExperienceMemory = {
          tick: currentTick,
          state: `e${Math.floor(agent.energy/10)}_x${agent.x}_y${agent.y}`,
          action: agent.lastRule || 'move',
          reward: agent.energy > 10 ? 1 : -1,
          outcome: agent.lastRule || 'survived',
        };
        
        // Update autonomous state
        autoState = updateAutonomousState(
          agent,
          autoState,
          currentTick,
          recentExp,
          intelligence
        );
        
        newAutonomousStates.set(agent.id, autoState);
        
        // Track totals
        newTotalConcepts += autoState.concepts.size;
        newTotalBehaviors += autoState.behaviors.size;
        
        // Track highest autonomy
        const levelScore = {
          none: 0, partial: 1, full: 2, transcendent: 3, singularity: 4
        }[autoState.autonomyLevel];
        
        if (!highestAutonomy || levelScore > highestAutonomy.score) {
          highestAutonomy = { 
            id: agent.id, 
            level: autoState.autonomyLevel, 
            score: levelScore 
          };
        }
      }
    }
    
    // Clean up autonomous states for dead agents
    const livingAgentIds = new Set(newAgents.map(a => a.id));
    for (const agentId of newAutonomousStates.keys()) {
      if (!livingAgentIds.has(agentId)) {
        newAutonomousStates.delete(agentId);
      }
    }
    
    // Update autonomous state
    setAutonomousStates(newAutonomousStates);
    setTotalAutonomousConcepts(newTotalConcepts);
    setTotalAutonomousBehaviors(newTotalBehaviors);
    if (highestAutonomy) {
      setMostAutonomousAgent({ id: highestAutonomy.id, level: highestAutonomy.level });
    }
    
    // Log significant autonomy events
    for (const [agentId, autoState] of newAutonomousStates) {
      // Check for new level achievements
      const prevState = autonomousStates.get(agentId);
      if (prevState && prevState.autonomyLevel !== autoState.autonomyLevel) {
        challengeLogs.push(`🧠 Agent ${agentId} reached ${autoState.autonomyLevel.toUpperCase()} autonomy!`);
      }
        // Log self-modifications
      if (autoState.selfModifications.length > 0) {
        const recentMod = autoState.selfModifications[autoState.selfModifications.length - 1];
        if (recentMod.tick === currentTick) {
          challengeLogs.push(`⚡ Agent ${agentId} self-modified: ${recentMod.description}`);
        }
      }
    }
    
    // ========================================
    // PHASE 6: AUTOMATIC INTERNET LEARNING
    // Agents automatically search and learn from the internet
    // ========================================
      if (autoLearningEnabled) {
      const autoLearnResult = runAutoLearningForAllAgents(
        newAgents,
        [...mathState.unlockedConcepts, ...newMathDiscoveries],
        [...physicsState.unlockedConcepts, ...newPhysicsDiscoveries],
        internetKnowledgeLearned,
        currentTick
      );
      
      // Update internet knowledge state
      if (autoLearnResult.totalConceptsLearned > 0) {
        setInternetKnowledgeLearned(autoLearnResult.updatedKnowledge);
        setTotalInternetLearning(prev => prev + autoLearnResult.totalConceptsLearned);
        setRecentAutoLearning(autoLearnResult.results);
        
        // Add logs
        challengeLogs.push(...autoLearnResult.allLogs);
      }
    }
    
    // Apply challenge-based energy costs to agents
    const challengeEnergyCost = getChallengeEnergyCost(updatedChallengeState);
    let challengeAffectedAgents = newAgents;
    
    if (challengeEnergyCost > 0) {
      challengeAffectedAgents = newAgents.map(agent => ({
        ...agent,
        energy: Math.max(0, agent.energy - challengeEnergyCost)
      })).filter(agent => agent.energy > 0);
      
      const deadCount = newAgents.length - challengeAffectedAgents.length;
      if (deadCount > 0) {
        challengeLogs.push(`☠️ ${deadCount} agent(s) died from environmental stress`);
      }
    }
    
    // Apply challenge modifiers to food spawning
    const effectiveFoodChance = getEffectiveFoodSpawnChance(updatedChallengeState);
    const effectiveFoodCount = getEffectiveFoodSpawnCount(updatedChallengeState);
    
    let finalGrid = newGrid;
    if (Math.random() < effectiveFoodChance && effectiveFoodCount > 0) {
      // Spawn additional food based on challenge modifiers
      const bonusFood = effectiveFoodCount - CONFIG.simulation.foodSpawnCount;
      if (bonusFood > 0) {
        finalGrid = placeRandomFood(newGrid, bonusFood);
      }
    }
    
    // ========================================
    // DYNAMIC GRID EXPANSION
    // ========================================
    let currentGridWidth = gridWidth;
    let currentGridHeight = gridHeight;
    
    // Check if grid should expand based on population density
    if (shouldExpandGrid(finalGrid, challengeAffectedAgents.length, currentGridWidth, currentGridHeight)) {
      const expansion = expandGrid(finalGrid, challengeAffectedAgents, currentGridWidth, currentGridHeight);
      
      if (expansion.expanded) {
        finalGrid = expansion.grid;
        currentGridWidth = expansion.width;
        currentGridHeight = expansion.height;
        
        // Update global dimensions for other functions
        updateGridDimensions(currentGridWidth, currentGridHeight);
        
        // Update state
        setGridWidth(currentGridWidth);
        setGridHeight(currentGridHeight);
        
        if (expansion.log) {
          challengeLogs.push(expansion.log);
        }
      }
    }
    
    setTick(newTick);
    setAgents(challengeAffectedAgents);
    setGrid(finalGrid);
    
    // Combine challenge logs with simulation logs
    const allLogs = [...challengeLogs, ...newLog];
    setLog(prev => [...allLogs, ...prev].slice(0, 80));
    setDiscoveries(prev => [...prev, ...newDiscoveries]);
    pushHistory(challengeAffectedAgents, newTick);

    // Process communication for self-aware agents
    processCommunication(challengeAffectedAgents, newTick);
  }, [agents, renderedGrid, pushHistory, challengeState, physicsState, mathState, eraState, discoveries, collaborativeDiscoveries, speechState, agentLanguages, gridWidth, gridHeight, autonomousStates, autoLearningEnabled, internetKnowledgeLearned]);

  // Communication processing function
  const processCommunication = useCallback((currentAgents: Agent[], currentTick: number) => {
    const newCommStates = new Map(communicationStates);
    const newMessages: AgentMessage[] = [];
    const activeAgentIds = new Set<number>();

    for (const agent of currentAgents) {
      // Simple consciousness check based on invention points and genes
      // Agents become "self-aware" when they have high creativity + curiosity + inventions
      const consciousnessScore = 
        (agent.genes.curiosity * 25) + 
        (agent.genes.creativity * 25) + 
        (agent.genes.social * 20) +
        (agent.inventionPoints * 0.5) +
        (agent.inventions.length * 10);
      
      const isSelfAware = consciousnessScore >= 60;
      
      if (!isSelfAware) continue;

      // Create a minimal consciousness state for communication
      const consciousnessState: ConsciousnessState = {
        level: ConsciousnessLevel.SELF_AWARE,
        score: consciousnessScore,
        indicators: [],
        awarenessEvents: [],
      };

      // Count nearby agents
      const nearbyAgentCount = currentAgents.filter(a => 
        a.id !== agent.id &&
        Math.abs(a.x - agent.x) <= 2 &&
        Math.abs(a.y - agent.y) <= 2
      ).length;

      // Get or create communication state for this agent
      const prevCommState = newCommStates.get(agent.id);
      const isFirstMessage = !prevCommState || prevCommState.messageCount === 0;
      
      // Check if should communicate
      const lastMessageTick = prevCommState?.lastMessageTick ?? -100;
      const minTicksBetween = 15;
      
      if (currentTick - lastMessageTick < minTicksBetween && !isFirstMessage) {
        // Still showing previous message
        if (prevCommState?.currentMessage && currentTick - prevCommState.currentMessage.tick <= 8) {
          activeAgentIds.add(agent.id);
        }
        continue;
      }

      // Communication chance
      const baseChance = isFirstMessage ? 0.8 : 0.08;
      const socialBonus = agent.genes.social * 0.05;
      const totalChance = baseChance + socialBonus;

      if (Math.random() > totalChance) continue;

      // Generate message
      const message = generateMessage(agent, consciousnessState, currentTick, nearbyAgentCount, isFirstMessage);
      
      // Update communication state
      const updatedCommState: CommunicationState = {
        agentId: agent.id,
        isActive: true,
        lastMessageTick: currentTick,
        messageCount: (prevCommState?.messageCount ?? 0) + 1,
        messages: [...(prevCommState?.messages ?? []), message].slice(-20),
        currentMessage: message,
      };
      
      newCommStates.set(agent.id, updatedCommState);
      newMessages.push(message);
      activeAgentIds.add(agent.id);
    }

    // Update states
    setCommunicationStates(newCommStates);
      if (newMessages.length > 0) {
      setCommunicationLog(prev => ({
        ...prev,
        messages: [...prev.messages, ...newMessages].slice(-100),
        activeAgents: activeAgentIds,
        totalMessages: prev.totalMessages + newMessages.length,
        firstCommunicationTick: prev.firstCommunicationTick ?? currentTick,
      }));
    } else {
      // Just update active agents
      setCommunicationLog(prev => ({
        ...prev,
        activeAgents: activeAgentIds,
      }));    }
  }, [communicationStates]);

  const handleReset = () => {
    // Reset grid dimensions to initial values
    GRID_WIDTH = CONFIG.grid.initialWidth;
    GRID_HEIGHT = CONFIG.grid.initialHeight;
    setGridWidth(CONFIG.grid.initialWidth);
    setGridHeight(CONFIG.grid.initialHeight);
    
    const { grid: newGrid, agents: newAgents } = initializeWorld();
    setGrid(newGrid);
    setAgents(newAgents);
    setLog([]);
    setSelectedAgentId(null);
    setTick(0);
    setHistory([]);
    setDiscoveries([]);
    setIsRunning(false);
    setWatchedTraitId(null);
    // Reset communication state
    setCommunicationLog(initializeCommunicationLog());
    setCommunicationStates(new Map());
    // Reset challenge state
    setChallengeState(initializeChallengeState());    // Reset physics state
    setPhysicsState(initializeCivilizationPhysics());
    // Reset math state
    setMathState(initializeCivilizationMath());
    // Reset procedural generation levels for unlimited evolution
    setGeneratedPhysicsLevel(0);
    setGeneratedMathLevel(0);
    // Reset era state
    setEraState(initializeCivilizationEra());    // Reset speech state
    setSpeechState(initializeCivilizationSpeech());
    setAgentLanguages(new Map());
    // Reset collaborative discoveries
    setCollaborativeDiscoveries(0);    // Reset internet learning state    setInternetKnowledgeLearned(new Set());
    setTotalInternetLearning(0);
    setAutoLearningEnabled(true);
    setRecentAutoLearning([]);
    // Reset autonomous evolution state
    setAutonomousStates(new Map());
    setTotalAutonomousConcepts(0);
    setTotalAutonomousBehaviors(0);
    setMostAutonomousAgent(null);
  };

  // Auto-run interval
  useEffect(() => {
    if (!isRunning) return;
    const id = setInterval(() => {
      handleStep();
    }, speedMs);
    return () => clearInterval(id);
  }, [isRunning, speedMs, handleStep]);

  const lastHistory = history.length > 0 ? history[history.length - 1] : null;

  /**
   * CODE QUALITY IMPROVEMENT: Add keyboard controls
   * Space: Play/Pause, Arrow Right: Step, R: Reset
   */
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      // Don't trigger if typing in an input
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        return;
      }

      switch (e.key) {
        case " ": // Space
          e.preventDefault();
          setIsRunning(r => !r);
          break;
        case "ArrowRight":
          e.preventDefault();
          if (!isRunning) {
            handleStep();
          }
          break;
        case "r":
        case "R":
          e.preventDefault();
          handleReset();
          break;
      }
    };

    window.addEventListener("keydown", handleKeyPress);
    return () => window.removeEventListener("keydown", handleKeyPress);
  }, [isRunning, handleStep, handleReset]);

  /**
   * CODE QUALITY IMPROVEMENT: Add aggregate statistics
   * Calculate average energy, mutation rate, and unique trait count
   */
  const stats = useMemo(() => {
    if (agents.length === 0) {
      return {
        avgEnergy: 0,
        avgMutationRate: 0,
        uniqueTraits: 0,
        avgReproThreshold: 0,
      };
    }

    const totalEnergy = agents.reduce((sum, a) => sum + a.energy, 0);
    const totalMutationRate = agents.reduce((sum, a) => sum + a.genes.mutationRate, 0);
    const totalReproThreshold = agents.reduce((sum, a) => sum + a.genes.reproductionThreshold, 0);
    const uniqueTraits = new Set(agents.map(a => a.genes.traitId)).size;

    return {
      avgEnergy: totalEnergy / agents.length,
      avgMutationRate: totalMutationRate / agents.length,
      avgReproThreshold: totalReproThreshold / agents.length,
      uniqueTraits,
    };
  }, [agents]);

  // Startup modal: ask to start new or load
  const handleStartupNew = () => {
    setShowStartupModal(false);
  };

  const handleStartupLoad = () => {
    setShowStartupModal(false);
    fileInputRef.current?.click();
  };

  /**
   * CODE QUALITY IMPROVEMENT: Add error handling for file loading
   * Validates JSON structure, checks grid dimensions, and provides user feedback
   */
  const handleFileLoad = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    
    try {
      setLoadError(null);
      const text = await file.text();
      const parsed = JSON.parse(text) as WorldState;
      
      // Validate structure
      if (!parsed.grid || !parsed.agents || parsed.tick === undefined || !parsed.history) {
        throw new Error("Invalid world file: missing required fields (grid, agents, tick, or history)");
      }
      
      // discoveries is optional for backwards compatibility
      const parsedDiscoveries = parsed.discoveries || [];
      
      // Validate grid dimensions
      if (parsed.grid.length !== GRID_HEIGHT || parsed.grid[0]?.length !== GRID_WIDTH) {
        throw new Error(
          `Invalid grid dimensions: expected ${GRID_WIDTH}x${GRID_HEIGHT}, got ${parsed.grid[0]?.length}x${parsed.grid.length}`
        );
      }
      
      // Validate agent positions
      for (const agent of parsed.agents) {
        if (agent.x < 0 || agent.x >= GRID_WIDTH || agent.y < 0 || agent.y >= GRID_HEIGHT) {
          throw new Error(`Invalid agent position: Agent ${agent.id} at (${agent.x},${agent.y})`);
        }
      }
      
      // All validations passed
      setGrid(parsed.grid);
      setAgents(parsed.agents);
      setTick(parsed.tick);
      setHistory(parsed.history);
      setDiscoveries(parsedDiscoveries);
      setLog([`World loaded successfully from ${file.name}`]);
      setSelectedAgentId(null);
      setIsRunning(false);
      setWatchedTraitId(null);
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : "Unknown error loading file";
      setLoadError(errorMsg);
      setLog(prev => [`ERROR loading world: ${errorMsg}`, ...prev]);
    } finally {
      // Reset file input so the same file can be loaded again
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  };
  return (
    <div
      style={{
        display: "flex",
        minHeight: "100vh",
        fontFamily: "system-ui, sans-serif",
        background: "#050814",
        color: "#f4f4f4",
        padding: "12px",
        boxSizing: "border-box",
        position: "relative"
      }}
    >
      {/* Startup modal */}
      {showStartupModal && (
        <div
          style={{
            position: "absolute",
            inset: 0,
            background: "rgba(0,0,0,0.7)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 10
          }}
        >
          <div
            style={{
              background: "#151a30",
              padding: "20px",
              borderRadius: 10,
              border: "1px solid #333",
              maxWidth: 360
            }}
          >
            <h3>Welcome to the world</h3>
            <p style={{ fontSize: 14, opacity: 0.9 }}>
              Start a new "simulação"/world, or load a previously "guardada"/saved world
              (JSON file).
            </p>
            <div style={{ marginTop: 12, display: "flex", gap: 8 }}>
              <button onClick={handleStartupNew}>New world</button>
              <button onClick={handleStartupLoad}>Load from file</button>
            </div>
          </div>
        </div>
      )}

      {/* Hidden file input for load */}
      <input
        type="file"
        accept="application/json"
        style={{ display: "none" }}
        ref={fileInputRef}
        onChange={handleFileLoad}
      />

      {/* LEFT: World grid */}
      <div style={{ flex: "0 0 auto", marginRight: "16px" }}>
        <h2>NPC World – Genetic Memory + RL</h2>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: `repeat(${GRID_WIDTH}, 22px)`,
            gridTemplateRows: `repeat(${GRID_HEIGHT}, 22px)`,
            gap: "2px",
            border: "1px solid #444",
            padding: "4px",
            background: "#151a30"
          }}
        >
          {renderedGrid.map((row, y) =>
            row.map((cell, x) => {
              // PERFORMANCE IMPROVEMENT: Use agentMap for O(1) lookup instead of O(n) find
              const agent = cell.agentId ? agentMap.get(cell.agentId) : undefined;
              const isSelected = selectedAgentId === agent?.id;
              const isWatched =
                agent && watchedTraitId !== null && agent.genes.traitId === watchedTraitId;

              let bg = "#1f2640";
              if (cell.food) bg = "#2c9c3f";
              if (agent) bg = colorForTrait(agent.genes.traitId);
              if (cell.food && agent) bg = "#ffb300";

              return (
                <div
                  key={`${x}-${y}`}
                  onClick={() => agent && setSelectedAgentId(agent.id)}
                  style={{
                    width: "22px",
                    height: "22px",
                    borderRadius: "4px",
                    background: bg,
                    border: isSelected
                      ? "2px solid #ffeb3b"
                      : isWatched
                      ? "2px solid #ff5252"
                      : "1px solid #333",
                    boxSizing: "border-box",
                    cursor: agent ? "pointer" : "default"
                  }}                  title={
                    agent
                      ? `Agent ${agent.id} – energy ${agent.energy}, traitId ${agent.genes.traitId}`
                      : cell.food
                      ? "Food"
                      : ""
                  }
                />
              );
            })
          )}
        </div>

        <div style={{ marginTop: "12px" }}>
          <button onClick={handleStep} style={{ marginRight: "8px" }}>
            Step
          </button>
          <button
            onClick={() => setIsRunning(r => !r)}
            style={{ marginRight: "8px" }}
          >
            {isRunning ? "Pause" : "Play"}
          </button>
          <button onClick={handleReset}>Reset World</button>
          <button
            onClick={() => downloadWorld({ grid, agents, tick, history, discoveries })}
            style={{ marginLeft: 8 }}
          >
            Save World (JSON)
          </button>          <button
            onClick={() => fileInputRef.current?.click()}
            style={{ marginLeft: 8 }}
          >
            Load World
          </button>
          
          {/* Data Export Buttons for Analysis */}
          <div style={{ marginTop: 8, display: "flex", gap: 4, flexWrap: "wrap" }}>
            <button
              onClick={() => exportCompleteData({ 
                grid, agents, tick, history, discoveries, 
                gridWidth: GRID_WIDTH, gridHeight: GRID_HEIGHT 
              })}
              style={{ fontSize: 11, padding: "4px 8px" }}
              title="Export all simulation data"
            >
              📊 Export All Data
            </button>
            <button
              onClick={() => exportEvolutionData({ 
                grid, agents, tick, history, discoveries, 
                gridWidth: GRID_WIDTH, gridHeight: GRID_HEIGHT 
              })}
              style={{ fontSize: 11, padding: "4px 8px" }}
              title="Export population and genetic evolution data"
            >
              🧬 Export Evolution
            </button>            <button
              onClick={() => exportInventionHistory({ 
                grid, agents, tick, history, discoveries, 
                gridWidth: GRID_WIDTH, gridHeight: GRID_HEIGHT 
              })}
              style={{ fontSize: 11, padding: "4px 8px" }}
              title="Export invention discovery history"
            >
              💡 Export Inventions
            </button>
            <button
              onClick={() => exportConversations(communicationLog, tick, agents)}
              style={{ fontSize: 11, padding: "4px 8px" }}
              title="Export agent conversations and questions"
            >
              🗣️ Export Conversations
            </button>
            <button
              onClick={() => setAnalysisOpen(true)}
              style={{ fontSize: 11, padding: "4px 8px", backgroundColor: "#4f46e5" }}
              title="Analyze exported data for sentience recommendations"
            >
              🔬 Analyze Data
            </button>
          </div>
          <div style={{ marginTop: 8, fontSize: 12 }}>
            Speed:{" "}
            <input
              type="range"
              min={100}
              max={1200}
              step={100}
              value={speedMs}
              onChange={e => setSpeedMs(Number(e.target.value))}
            />{" "}
            {speedMs} ms/tick
          </div>          <div style={{ marginTop: 8, fontSize: 12, opacity: 0.8 }}>
            <div>Tick: {tick} | Agents: {agents.length} | Grid: {gridWidth}×{gridHeight}
              {gridWidth >= CONFIG.grid.maxWidth && gridHeight >= CONFIG.grid.maxHeight && (
                <span style={{ color: '#ffd700' }}> (MAX)</span>
              )}
            </div>
            <div style={{ marginTop: 4 }}>
              <strong>Keyboard shortcuts:</strong> Space (Play/Pause), → (Step), R (Reset)
            </div>
          </div>
          
          {/* Environmental Challenges Display */}
          <div style={{ 
            marginTop: 8, 
            padding: 8, 
            background: hasNegativeChallenge(challengeState) ? "#4a1a1a" : 
                       hasPositiveEvent(challengeState) ? "#1a4a2a" : "#1a2a3a",
            borderRadius: 4, 
            fontSize: 12,
            border: `1px solid ${hasNegativeChallenge(challengeState) ? "#8b3a3a" : 
                                hasPositiveEvent(challengeState) ? "#3a8b4a" : "#3a4a5a"}`
          }}>
            <div style={{ fontWeight: "bold", marginBottom: 4 }}>
              🌍 Environmental Conditions
            </div>
            <div>{getActiveChallengesSummary(challengeState)}</div>
            <div style={{ marginTop: 4, fontSize: 10, opacity: 0.7 }}>
              Events: {challengeState.totalFamines} famines | {challengeState.totalHarshWeatherEvents} storms | {challengeState.totalAbundanceEvents} abundances
            </div>
          </div>
          
          {loadError && (
            <div style={{ marginTop: 8, padding: 8, background: "#8b0000", borderRadius: 4, fontSize: 12 }}>
              Error: {loadError}
            </div>
          )}
        </div>
      </div>      {/* RIGHT: Side panel */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0 }}>
        {/* Statistics Display */}
        <div
          style={{
            marginBottom: "12px",
            padding: "10px",
            background: "#151a30",
            borderRadius: "8px",
            border: "1px solid #333"
          }}
        >
          <h3>Statistics – "Estatísticas"</h3>
          <div style={{ fontSize: "0.9em" }}>
            <p>
              <strong>Avg Energy:</strong> {stats.avgEnergy.toFixed(2)} |{" "}
              <strong>Avg Mutation Rate:</strong> {stats.avgMutationRate.toFixed(3)}
            </p>
            <p>
              <strong>Avg Repro Threshold:</strong> {stats.avgReproThreshold.toFixed(1)} |{" "}
              <strong>Unique Traits:</strong> {stats.uniqueTraits}
            </p>
          </div>
        </div>

        {/* Agent inspector */}
        <div
          style={{
            marginBottom: "12px",
            padding: "10px",
            background: "#151a30",
            borderRadius: "8px",
            border: "1px solid #333"
          }}
        >
          <h3>Agent Inspector</h3>
          {selectedAgent ? (
            <>
              <p>
                <strong>ID:</strong> {selectedAgent.id}
              </p>
              <p>
                <strong>Position:</strong> ({selectedAgent.x},{selectedAgent.y})
              </p>              <p>
                <strong>Energy:</strong> {selectedAgent.energy}
              </p>
              <p>
                <strong>Invention Points:</strong> {selectedAgent.inventionPoints.toFixed(1)}
                <br />
                <span style={{ fontSize: "0.85em", opacity: 0.8 }}>
                  (Accumulated through exploration & curiosity, used to inspire new inventions)
                </span>
              </p>
              <p>
                <strong>Last Rule ("Regra" / "Regle"):</strong>{" "}
                {selectedAgent.lastRule}
              </p>
              <p>
                <strong>Genes:</strong>
                <br />
                foodPref={selectedAgent.genes.foodPreference.toFixed(2)},{" "}
                explore={selectedAgent.genes.exploration.toFixed(2)},{" "}
                reproThresh={selectedAgent.genes.reproductionThreshold.toFixed(1)},{" "}
                mutationRate={selectedAgent.genes.mutationRate.toFixed(2)}
                <br />
                {/* NEW GENES */}
                curiosity={selectedAgent.genes.curiosity.toFixed(2)},{" "}
                social={selectedAgent.genes.social.toFixed(2)},{" "}
                creativity={selectedAgent.genes.creativity.toFixed(2)},{" "}
                patience={selectedAgent.genes.patience.toFixed(2)}
                <br />
                traitId={selectedAgent.genes.traitId}
              </p>
              <p>
                <strong>Inventions ({selectedAgent.inventions.length}):</strong>
                {selectedAgent.inventions.length === 0 ? (
                  <span> None yet</span>
                ) : (
                  <ul style={{ paddingLeft: "18px", fontSize: "0.85em", marginTop: 4 }}>
                    {selectedAgent.inventions.map(inv => (
                      <li key={inv.id}>
                        {inv.name} - {inv.description || inv.effect.type}
                      </li>
                    ))}
                  </ul>
                )}
              </p>
            </>          ) : (
            <p>Click an agent in the grid to inspect its genetic memory.</p>
          )}
        </div>

        {/* Rules + RL + evolution description */}
        <div
          style={{
            marginBottom: "12px",
            padding: "10px",
            background: "#151a30",
            borderRadius: "8px",
            border: "1px solid #333"
          }}
        >
          <h3>Behavior, RL & Evolution</h3>
          <ul style={{ fontSize: "0.9em" }}>
            <li>
              <strong>Hard Rule:</strong> if hungry and food is adjacent, move
              toward food (survival).
            </li>
            <li>
              <strong>RL Policy:</strong> otherwise, a learned policy chooses
              up/down/left/right/stay to maximize future reward.
            </li>            <li>
              <strong>Reproduction:</strong> if energy &gt; reproductionThreshold,
              agent may split energy with a child, whose genes are mutated.
            </li>
          </ul>
        </div>

        {/* Charts */}
        <div
          style={{
            marginBottom: "12px",
            padding: "10px",
            background: "#151a30",
            borderRadius: "8px",
            border: "1px solid #333"
          }}
        >
          <h3>Population Over Time</h3>
          <PopulationChart history={history} />
        </div>

        <div
          style={{
            marginBottom: "12px",
            padding: "10px",
            background: "#151a30",
            borderRadius: "8px",
            border: "1px solid #333"
          }}
        >
          <h3>
            Trait Distribution – "famílias" / "Stämme"
            {watchedTraitId !== null && (
              <span style={{ fontSize: 12, marginLeft: 8 }}>
                Watching traitId {watchedTraitId}
              </span>
            )}
          </h3>
          <TraitChart
            last={lastHistory}
            watchedTraitId={watchedTraitId}
            onSelectTrait={tid => setWatchedTraitId(tid)}
          />
        </div>

        {/* Discovery Timeline */}
        <DiscoveryTimeline discoveries={discoveries} />

        {/* Invention Statistics */}
        <InventionStats agents={agents} discoveries={discoveries} />

        {/* Physics Progress */}
        <PhysicsPanel 
          unlockedConcepts={physicsState.unlockedConcepts}
          lastDiscoveryTick={physicsState.lastDiscoveryTick}
          currentTick={tick}
        />

        {/* Mathematics Progress */}
        <MathPanel 
          unlockedConcepts={mathState.unlockedConcepts}
          lastDiscoveryTick={mathState.lastDiscoveryTick}
          currentTick={tick}
        />

        {/* Era Progression */}
        <EraPanel
          eraState={eraState}
          physicsCount={physicsState.unlockedConcepts.length}
          mathCount={mathState.unlockedConcepts.length}
          currentTick={tick}
        />        {/* Speech & Language */}
        <SpeechPanel
          speechState={speechState}
          agents={agents}
          agentLanguages={agentLanguages}
          currentTick={tick}
        />        {/* Agent Explanations - using math & physics knowledge */}        <ExplanationPanel
          agents={agents}
          unlockedMath={mathState.unlockedConcepts}
          unlockedPhysics={physicsState.unlockedConcepts}
          currentTick={tick}
          selectedAgentId={selectedAgentId}
          persistedInternetKnowledge={internetKnowledgeLearned}
          onInternetKnowledgeChange={setInternetKnowledgeLearned}
          onLearningComplete={(count) => setTotalInternetLearning(prev => prev + count)}
          autoLearningEnabled={autoLearningEnabled}
          onAutoLearningToggle={setAutoLearningEnabled}
          recentAutoLearning={recentAutoLearning}
        />

        {/* Autonomous Evolution Panel - Self-directed learning visualization */}
        <AutonomousPanel
          autonomousStates={autonomousStates}
          totalConcepts={totalAutonomousConcepts}
          totalBehaviors={totalAutonomousBehaviors}
          mostAutonomous={mostAutonomousAgent}
          tick={tick}
        />

        {/* Log */}
        <div
          style={{
            flex: 1,
            padding: "10px",
            background: "#151a30",
            borderRadius: "8px",
            border: "1px solid #333",
            overflowY: "auto"
          }}
        >          <h3>Action Log</h3>
          {log.length === 0 ? (
            <p>No steps yet. Press "Step" or "Play" to advance the world.</p>
          ) : (
            <ul style={{ paddingLeft: "18px" }}>
              {log.map((entry, i) => (
                <li key={i} style={{ fontSize: "0.8em" }}>
                  {entry}
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>      {/* Chat Panel for Agent Communications */}
      <ChatPanel
        communicationLog={communicationLog}
        isOpen={chatOpen}
        onToggle={() => setChatOpen(!chatOpen)}
        onClear={() => {
          setCommunicationLog(initializeCommunicationLog());
          setCommunicationStates(new Map());
        }}
        onSendMessage={handleSendUserMessage}
        selfAwareAgentIds={selfAwareAgentIds}
        currentTick={tick}
      />

      {/* Analysis Panel for Data Analysis */}
      <AnalysisPanel
        isOpen={analysisOpen}
        onClose={() => setAnalysisOpen(false)}
      />
    </div>
  );
};

export default App;