// Contact: Name: dtay83 <dartey.banahene@gmail.com>
import React, { useState, useMemo, useEffect, useRef, useCallback } from "react";

type Direction = "up" | "down" | "left" | "right" | "stay";

interface Genes {
  foodPreference: number;        // 0–1: prioritize food when hungry
  exploration: number;           // 0–1: how often they wander
  reproductionThreshold: number; // energia needed to reproduce
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
  energy: number; // "energia"
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
    width: 16,
    height: 10,
  },
  simulation: {
    initialAgents: 6,
    initialFood: 18,
    foodSpawnChance: 0.6,
    foodSpawnCount: 1,
    baseEnergyCost: 1,
    foodEnergyBonus: 5,
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
    reproductionThreshold: { min: 15, max: 23 },
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
    initialMin: 10,
    initialMax: 16,
  },
} as const;

// Backwards compatibility aliases
const GRID_WIDTH = CONFIG.grid.width;
const GRID_HEIGHT = CONFIG.grid.height;
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
    // Energy efficiency - scales with creativity
    const multiplier = 0.95 - (creativityFactor * 0.25); // 0.7 to 0.95
    effect = { type: 'energy_efficiency', multiplier: Math.max(0.5, multiplier) };
    description = `Reduces energy cost by ${Math.round((1 - multiplier) * 100)}%`;
  } else if (effectRoll < 0.7) {
    // Reproduction boost - scales with creativity
    const bonus = Math.ceil(2 + creativityFactor * 8); // 2 to 10 bonus
    effect = { type: 'reproduction_boost', bonus };
    description = `+${bonus} bonus energy for reproduction`;
  } else if (effectRoll < 0.85) {
    // Food detection range - scales with exploration
    const range = Math.ceil(1 + agent.genes.exploration * 3); // 1 to 4 range
    effect = { type: 'food_detection_range', range };
    description = `Detect food ${range} cells away`;
  } else if (effectRoll < 0.95) {
    // Storage capacity - scales with patience
    const capacity = Math.ceil(5 + agent.genes.patience * 20); // 5 to 25 capacity
    effect = { type: 'storage', capacity };
    description = `Store ${capacity} extra energy`;
  } else {
    // Defense - scales with both genes
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

// Genetic memória + random trait generation
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
    { x, y: y - 1, dir: "up" },
    { x, y: y + 1, dir: "down" },
    { x: x - 1, y, dir: "left" },
    { x: x + 1, y, dir: "right" }
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
  let cost = 1; // Base cost
  
  // Check for energy efficiency inventions
  for (const inv of agent.inventions) {
    if (inv.effect.type === 'energy_efficiency') {
      cost *= inv.effect.multiplier;
    }
  }
  
  return Math.max(0.5, cost); // Minimum cost of 0.5
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
 */
function stepWorld(
  agents: Agent[],
  grid: Cell[][],
  tick: number
): { agents: Agent[]; grid: Cell[][]; log: string[]; discoveries: DiscoveryEvent[] } {
  const newGrid: Cell[][] = grid.map(row =>
    row.map(cell => ({ ...cell, agentId: undefined }))
  );

  const logs: string[] = [];
  const updatedAgents: Agent[] = [];
  const discoveries: DiscoveryEvent[] = [];

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

    // Apply energy cost with invention effects
    const movementCost = getMovementCost(agent);
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

    // REPRODUCTION
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

        const childGenes = mutateGenes(parentAgent.genes);
        
        // Inheritance: children can inherit parent's inventions based on social gene
        const inheritedInventions = parentAgent.inventions.filter(
          inv => Math.random() < parentAgent.genes.social * 0.8
        );
        
        // Children inherit a portion of parent's invention points (learning from parent)
        const inheritedPoints = parentAgent.inventionPoints * parentAgent.genes.social * 0.3;
        
        const child: Agent = {
          id: nextId++,
          x: spot.x,
          y: spot.y,
          energy: childEnergy,
          genes: childGenes,
          memory: { qTable: {} },
          lastRule: 'Born (memória genética + "Mutation")',
          inventions: inheritedInventions.map(inv => ({
            ...inv,
            // Mark as inherited, not discovered by this agent
            discoveredBy: parentAgent.id,
          })),
          inventionPoints: inheritedPoints, // NEW: Inherit some creativity points
        };

        newGrid[spot.y][spot.x].agentId = child.id;
        destinationMap.set(`${spot.x},${spot.y}`, child.id); // Register child position
        updatedAgents.push(child);

        reward += CONFIG.simulation.reproductionReward;
        reproduced = true;

        const inheritMsg = inheritedInventions.length > 0 
          ? ` inherited ${inheritedInventions.length} inventions`
          : '';
        logs.push(
          `Agent ${parentAgent.id} reproduced: child ${child.id} at (${spot.x},${spot.y}) with traitId ${child.genes.traitId}, energia ${childEnergy}${inheritMsg}`
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
        `Agent ${parentAgent.id} used ${decision.rule}, moved to (${parentAgent.x},${parentAgent.y})` +
          (ateFood ? " and ate food (+5 energia)" : "") +
          (reproduced ? " and reproduced (+2 reward)" : "") +
          `, energia now ${parentAgent.energy}, traitId=${parentAgent.genes.traitId}`
      );
    } else {
      reward -= CONFIG.simulation.deathPenalty;
      logs.push(
        `Agent ${agent.id} ran out of energia at (${finalX},${finalY}) and was removed.`
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
    return { agents: updatedAgents, grid: gridWithFood, log: logs, discoveries };
  }

  return { agents: updatedAgents, grid: newGrid, log: logs, discoveries };
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
  if (entries.length === 0) return <p>No agentes alive.</p>;

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

  const [log, setLog] = useState<string[]>([]);
  const [selectedAgentId, setSelectedAgentId] = useState<number | null>(null);
  const [tick, setTick] = useState(0);
  const [history, setHistory] = useState<HistoryPoint[]>([]);
  const [discoveries, setDiscoveries] = useState<DiscoveryEvent[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [speedMs, setSpeedMs] = useState(400);
  const [watchedTraitId, setWatchedTraitId] = useState<number | null>(null);
  const [showStartupModal, setShowStartupModal] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);

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
    const copy = grid.map(row => row.map(cell => ({ ...cell, agentId: undefined })));
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
  }, []);

  const handleStep = useCallback(() => {
    const { agents: newAgents, grid: newGrid, log: newLog, discoveries: newDiscoveries } = stepWorld(
      agents,
      renderedGrid,
      tickRef.current
    );
    const newTick = tickRef.current + 1;
    setTick(newTick);
    setAgents(newAgents);
    setGrid(newGrid);
    setLog(prev => [...newLog, ...prev].slice(0, 80));
    setDiscoveries(prev => [...prev, ...newDiscoveries]);
    pushHistory(newAgents, newTick);
  }, [agents, renderedGrid, pushHistory]);

  const handleReset = () => {
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
        height: "100vh",
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
        <h2>NPC World – "memória genética" + RL</h2>
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
                  }}
                  title={
                    agent
                      ? `Agent ${agent.id} – energia ${agent.energy}, traitId ${agent.genes.traitId}`
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
          </button>
          <button
            onClick={() => fileInputRef.current?.click()}
            style={{ marginLeft: 8 }}
          >
            Load World
          </button>
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
          </div>
          <div style={{ marginTop: 8, fontSize: 12, opacity: 0.8 }}>
            <div>Tick: {tick} | Agentes: {agents.length}</div>
            <div style={{ marginTop: 4 }}>
              <strong>Keyboard shortcuts:</strong> Space (Play/Pause), → (Step), R (Reset)
            </div>
          </div>
          {loadError && (
            <div style={{ marginTop: 8, padding: 8, background: "#8b0000", borderRadius: 4, fontSize: 12 }}>
              Error: {loadError}
            </div>
          )}
        </div>
      </div>

      {/* RIGHT: Side panel */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column" }}>
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
          <h3>Agent Inspector – "agente" / "Figur"</h3>
          {selectedAgent ? (
            <>
              <p>
                <strong>ID:</strong> {selectedAgent.id}
              </p>
              <p>
                <strong>Position:</strong> ({selectedAgent.x},{selectedAgent.y})
              </p>
              <p>
                <strong>Energia:</strong> {selectedAgent.energy}
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
            </>
          ) : (
            <p>Click an agente in the grid to inspect its "memória genética".</p>
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
            </li>
            <li>
              <strong>Reproduction:</strong> if energia &gt; reproductionThreshold,
              agente may split energia with a child, whose genes are mutated.
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
          <h3>Population Over Time – "população"</h3>
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
        >
          <h3>Action Log – "histórico" / "Protokoll"</h3>
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
      </div>
    </div>
  );
};

export default App;