// Contact: Name: dtay83 <dartey.banahene@gmail.com>
import React, { useState, useMemo, useEffect, useRef, useCallback } from "react";

type Direction = "up" | "down" | "left" | "right" | "stay";

// Tools and inventions that agents can discover/create
type ToolType = "none" | "stick" | "stone_tool" | "spear" | "basket" | "fire";

interface Tool {
  type: ToolType;
  durability: number; // 0-100, decreases with use
}

// Knowledge represents discoveries that can be passed down
interface Knowledge {
  toolmaking: number;      // 0-1: ability to craft tools
  firemaking: number;      // 0-1: ability to create/use fire
  foodStorage: number;     // 0-1: ability to store food efficiently
  shelterBuilding: number; // 0-1: ability to build shelters
  hunting: number;         // 0-1: hunting efficiency
}

interface Genes {
  foodPreference: number;        // 0–1: prioritize food when hungry
  exploration: number;           // 0–1: how often they wander
  reproductionThreshold: number; // energia needed to reproduce
  mutationRate: number;          // 0–1: chance each gene mutates
  traitId: number;               // lineage / random trait marker
  riskTolerance: number;         // 0–1: willingness to act with lower energy
  socialDrive: number;           // 0–1: preference for being near other agents
  intelligence: number;          // 0–1: ability to learn/discover new things
  creativity: number;            // 0–1: chance to invent new tools/techniques
}

interface Memory {
  qTable: Record<string, number>; // RL value table: (state|action) -> Q
  knowledge: Knowledge;           // learned skills and discoveries
}

interface Agent {
  id: number;
  x: number;
  y: number;
  energy: number;
  sex: "M" | "F";
  ageTicks: number;
  genes: Genes;
  memory: Memory;
  lastRule?: string;
  // Pregnancy/reproduction state
  pregnantWith?: {
    mateId: number;
    gestationTicks: number;
    childGenes: Genes;
  };
  reproductionCooldown?: number; // ticks until can reproduce again
  // Visual state
  lastDirection?: Direction;
  // Tools and inventory
  tool?: Tool;
  storedFood?: number; // food stored (if has basket)
}

interface Cell {
  food: boolean;
  agentId?: number;
  terrain: "plain" | "water" | "rich" | "hazard";
  // Structures
  shelter?: {
    builderId: number;
    durability: number;
  };
  fire?: {
    ticksRemaining: number;
  };
  // Resources for crafting
  hasSticks?: boolean;
  hasStones?: boolean;
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
}

interface SimulationSettings {
  initialAgents: number;
  initialFood: number;
  foodSpawnChance: number;
  foodSpawnCount: number;
  foodRecycleEnabled: boolean;
  maxFoodPerAgent: number; // max food cells allowed per living agent
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
    initialAgents: 8,
    initialFood: 20,
    foodSpawnChance: 0.85,    // increased from 0.75 - more food spawns
    foodSpawnCount: 3,         // increased from 2 - more food per spawn
    baseEnergyCost: 0.5,       // reduced from 1 - agents burn energy slower
    foodEnergyBonus: 8,        // increased from 5 - eating is more rewarding
    reproductionReward: 8,     // increased from 2 - BIG reward for passing genes
    matingReward: 3,           // new: reward for successful mating
    birthReward: 5,            // new: reward for giving birth
    deathPenalty: 2,           // reduced from 5 - death is less punishing to learning
  },
  rl: {
    alpha: 0.3,   // learning rate
    gamma: 0.95,  // increased from 0.9 - value future rewards more (genetic legacy)
    epsilon: 0.15, // reduced from 0.2 - less random exploration, more learned behavior
  },
  genes: {
    foodPreference: { min: 0.6, max: 1.0 },
    exploration: { min: 0.3, max: 0.8 },
    reproductionThreshold: { min: 10, max: 18 }, // lowered to make reproduction easier
    mutationRate: { min: 0.12, max: 0.35 },      // slightly higher mutation rates
    riskTolerance: { min: 0.2, max: 0.8 },
    socialDrive: { min: 0.3, max: 0.9 },         // higher social drive to encourage proximity/mating
    mutation: {
      foodPreferenceMagnitude: 0.15,
      explorationMagnitude: 0.2,
      reproductionThresholdMagnitude: 2,         // smaller mutations to threshold
      mutationRateMagnitude: 0.06,               // mutation rate can evolve more
      riskToleranceMagnitude: 0.15,
      socialDriveMagnitude: 0.15,
      intelligenceMagnitude: 0.12,
      creativityMagnitude: 0.12,
      newTraitChance: 0.10,                      // increased from 0.08 - more genetic diversity
      mutationRateEvolveMin: 0.05,               // minimum mutation rate higher
      mutationRateEvolveMax: 0.5,
      reproductionThresholdMin: 6,               // can evolve very low threshold
      reproductionThresholdMax: 25,
    },
    intelligence: { min: 0.1, max: 0.5 },        // starting intelligence range
    creativity: { min: 0.1, max: 0.4 },          // starting creativity range
  },energy: {
    hungryCutoffRatio: 8,      // increased from 6 - seek food earlier
    initialMin: 15,            // increased from 10 - start with more energy
    initialMax: 25,            // increased from 16 - start with more energy
  },
  time: {
    yearsPerTick: 0.5,       // conversion from ticks → years
    minReproAgeYears: 18,    // reduced from 20 - can reproduce earlier
    maxReproAgeYears: 140,   // increased from 120 - longer fertile period
    maxAgeYears: 250         // increased from 200 - longer lifespan
  },
  reproduction: {
    gestationTicks: 3,       // reduced from 4 - faster pregnancy
    cooldownTicks: 4,        // reduced from 6 - can reproduce more often
    energyCostRatio: 0.25,   // reduced from 0.33 - cheaper to reproduce
    matingEnergyCost: 1,     // reduced mating cost
  },
  terrain: {
    hazardDamage: 0.5,       // reduced from 1 - hazards less deadly
    richFoodBonus: 3,        // increased from 2 - rich terrain more valuable
    waterSpawnChance: 0.04,  // reduced from 0.05 - less water obstacles
    richSpawnChance: 0.12,   // increased from 0.10 - more rich terrain
    hazardSpawnChance: 0.02, // reduced from 0.03 - fewer hazards
    resourceSpawnChance: 0.08, // chance for sticks/stones to spawn
  },
  invention: {
    discoveryChance: 0.02,   // base chance per tick to discover something new
    toolDurability: 100,     // starting durability for tools
    shelterProtection: 0.5,  // damage reduction from shelter
    fireWarmth: 2,           // energy saved per tick near fire
    fireDuration: 20,        // ticks a fire lasts
    basketCapacity: 5,       // max food a basket can hold
    inventionReward: 5,      // RL reward for inventing something
    knowledgeTransferRate: 0.7, // how much knowledge passes to children
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
    Array.from({ length: GRID_WIDTH }, () => ({ food: false, terrain: "plain" as const }))
  );
}

function randomInt(max: number) {
  return Math.floor(Math.random() * max);
}

/**
 * Add random terrain to the grid
 * Includes water, rich soil, and hazard terrain
 */
function addRandomTerrain(grid: Cell[][]): Cell[][] {
  const copy = grid.map(row => row.map(cell => ({ ...cell })));
  for (let y = 0; y < GRID_HEIGHT; y++) {
    for (let x = 0; x < GRID_WIDTH; x++) {
      const roll = Math.random();
      if (roll < CONFIG.terrain.waterSpawnChance) {
        copy[y][x].terrain = "water";
      } else if (roll < CONFIG.terrain.waterSpawnChance + CONFIG.terrain.richSpawnChance) {
        copy[y][x].terrain = "rich";
      } else if (roll < CONFIG.terrain.waterSpawnChance + CONFIG.terrain.richSpawnChance + CONFIG.terrain.hazardSpawnChance) {
        copy[y][x].terrain = "hazard";
      }
    }
  }
  return copy;
}

function placeRandomFood(grid: Cell[][], count: number): Cell[][] {
  const copy = grid.map(row => row.map(cell => ({ ...cell })));
  let placed = 0;
  let safety = 0;
  while (placed < count && safety < 2000) {
    safety++;
    const y = randomInt(GRID_HEIGHT);
    const x = randomInt(GRID_WIDTH);
    // Don't place food on water or occupied cells
    if (!copy[y][x].food && copy[y][x].agentId === undefined && copy[y][x].terrain !== "water") {
      copy[y][x].food = true;
      placed++;
    }
  }
  return copy;
}

/**
 * Spawn sticks and stones on the grid for crafting
 */
function spawnResources(grid: Cell[][]): Cell[][] {
  const copy = grid.map(row => row.map(cell => ({ ...cell })));
  for (let y = 0; y < GRID_HEIGHT; y++) {
    for (let x = 0; x < GRID_WIDTH; x++) {
      if (copy[y][x].terrain === "water") continue;
      
      // Chance to spawn sticks (more common)
      if (!copy[y][x].hasSticks && Math.random() < CONFIG.terrain.resourceSpawnChance) {
        copy[y][x].hasSticks = true;
      }
      // Chance to spawn stones (less common, prefer plain/hazard terrain)
      if (!copy[y][x].hasStones && Math.random() < CONFIG.terrain.resourceSpawnChance * 0.6) {
        if (copy[y][x].terrain === "plain" || copy[y][x].terrain === "hazard") {
          copy[y][x].hasStones = true;
        }
      }
    }
  }
  return copy;
}

/**
 * Food recycling: remove excess food when there are too few agents
 * This prevents the board from getting overcrowded with food when agents die off
 * @param grid - Current grid state
 * @param agentCount - Number of living agents
 * @param maxFoodPerAgent - Maximum food cells allowed per agent
 * @returns Object with updated grid and number of food cells removed
 */
function recycleFoodIfNeeded(
  grid: Cell[][],
  agentCount: number,
  maxFoodPerAgent: number
): { grid: Cell[][]; removedCount: number } {
  // Count current food
  let foodCount = 0;
  const foodPositions: { x: number; y: number }[] = [];
  
  for (let y = 0; y < GRID_HEIGHT; y++) {
    for (let x = 0; x < GRID_WIDTH; x++) {
      if (grid[y][x].food) {
        foodCount++;
        foodPositions.push({ x, y });
      }
    }
  }
  
  // Calculate max allowed food (minimum of 5 to keep some food even with 0-1 agents)
  const maxAllowedFood = Math.max(5, agentCount * maxFoodPerAgent);
  
  // If under limit, no recycling needed
  if (foodCount <= maxAllowedFood) {
    return { grid, removedCount: 0 };
  }
  
  // Need to remove excess food
  const toRemove = foodCount - maxAllowedFood;
  const copy = grid.map(row => row.map(cell => ({ ...cell })));
  
  // Shuffle food positions and remove from random locations
  const shuffled = foodPositions.sort(() => Math.random() - 0.5);
  let removed = 0;
  
  for (const pos of shuffled) {
    if (removed >= toRemove) break;
    copy[pos.y][pos.x].food = false;
    removed++;
  }
  
  return { grid: copy, removedCount: removed };
}

function randomTraitId(): number {
  return randomInt(999999);
}

// Create initial empty knowledge
function createInitialKnowledge(): Knowledge {
  return {
    toolmaking: 0,
    firemaking: 0,
    foodStorage: 0,
    shelterBuilding: 0,
    hunting: 0,
  };
}

// Create initial memory with empty Q-table and knowledge
function createInitialMemory(): Memory {
  return {
    qTable: {},
    knowledge: createInitialKnowledge(),
  };
}

// Initial genes generator
function createRandomGenes(): Genes {
  return {
    foodPreference: CONFIG.genes.foodPreference.min + Math.random() * (CONFIG.genes.foodPreference.max - CONFIG.genes.foodPreference.min),
    exploration: CONFIG.genes.exploration.min + Math.random() * (CONFIG.genes.exploration.max - CONFIG.genes.exploration.min),
    reproductionThreshold: CONFIG.genes.reproductionThreshold.min + Math.random() * (CONFIG.genes.reproductionThreshold.max - CONFIG.genes.reproductionThreshold.min),
    mutationRate: CONFIG.genes.mutationRate.min + Math.random() * (CONFIG.genes.mutationRate.max - CONFIG.genes.mutationRate.min),
    traitId: randomTraitId(),
    riskTolerance: CONFIG.genes.riskTolerance.min + Math.random() * (CONFIG.genes.riskTolerance.max - CONFIG.genes.riskTolerance.min),
    socialDrive: CONFIG.genes.socialDrive.min + Math.random() * (CONFIG.genes.socialDrive.max - CONFIG.genes.socialDrive.min),
    intelligence: CONFIG.genes.intelligence.min + Math.random() * (CONFIG.genes.intelligence.max - CONFIG.genes.intelligence.min),
    creativity: CONFIG.genes.creativity.min + Math.random() * (CONFIG.genes.creativity.max - CONFIG.genes.creativity.min),
  };
}

function createInitialAgents(grid: Cell[][], count: number = INITIAL_AGENTS): Agent[] {
  const agents: Agent[] = [];
  let idCounter = 1;
  const taken: Set<string> = new Set();

  while (agents.length < count) {
    const y = randomInt(GRID_HEIGHT);
    const x = randomInt(GRID_WIDTH);
    const key = `${x},${y}`;
    // Don't spawn on water, food, or occupied cells
    if (taken.has(key) || grid[y][x].food || grid[y][x].terrain === "water") continue;
    taken.add(key);
    agents.push({
      id: idCounter++,
      x,
      y,
      energy: CONFIG.energy.initialMin + randomInt(CONFIG.energy.initialMax - CONFIG.energy.initialMin),
      sex: Math.random() < 0.5 ? "M" : "F",
      ageTicks: 0,
      genes: createRandomGenes(),
      memory: createInitialMemory(),
      lastRule: "none"
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

  // sometimes spawn a totally new traitId => random trait generation
  const traitId =
    Math.random() < CONFIG.genes.mutation.newTraitChance ? randomTraitId() : parent.traitId;

  // Mutate personality traits
  const riskTolerance = mutateValue(
    parent.riskTolerance,
    mutationRate,
    CONFIG.genes.mutation.riskToleranceMagnitude,
    0.0,
    1.0
  );

  const socialDrive = mutateValue(
    parent.socialDrive,
    mutationRate,
    CONFIG.genes.mutation.socialDriveMagnitude,
    0.0,
    1.0
  );

  const intelligence = mutateValue(
    parent.intelligence,
    mutationRate,
    CONFIG.genes.mutation.intelligenceMagnitude,
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

  return {
    foodPreference,
    exploration,
    reproductionThreshold,
    mutationRate: newMutationRate,
    traitId,
    riskTolerance,
    socialDrive,
    intelligence,
    creativity,
  };
}

/**
 * Combine genes from two parents for sexual reproduction.
 * Averages numeric gene fields, handles traitId inheritance,
 * then applies mutation.
 */
function combineGenes(parent1: Genes, parent2: Genes): Genes {
  const avgMutationRate = (parent1.mutationRate + parent2.mutationRate) / 2;
  
  // For traitId: if same, keep it; if different, generate new
  const combinedTraitId = parent1.traitId === parent2.traitId 
    ? parent1.traitId 
    : randomTraitId();
  
  const combinedGenes: Genes = {
    foodPreference: (parent1.foodPreference + parent2.foodPreference) / 2,
    exploration: (parent1.exploration + parent2.exploration) / 2,
    reproductionThreshold: (parent1.reproductionThreshold + parent2.reproductionThreshold) / 2,
    mutationRate: avgMutationRate,
    traitId: combinedTraitId,
    riskTolerance: (parent1.riskTolerance + parent2.riskTolerance) / 2,
    socialDrive: (parent1.socialDrive + parent2.socialDrive) / 2,
    intelligence: (parent1.intelligence + parent2.intelligence) / 2,
    creativity: (parent1.creativity + parent2.creativity) / 2,
  };
  
  // Apply mutation to the combined genes
  return mutateGenes(combinedGenes);
}

/**
 * INVENTION SYSTEM
 * Agents can discover tools and techniques based on intelligence + creativity + experience
 */

// Check if agent discovers something new this tick
function tryDiscovery(agent: Agent, cell: Cell): { discovery: string | null; knowledge: Knowledge } {
  const { intelligence, creativity } = agent.genes;
  const knowledge = { ...agent.memory.knowledge };
  
  // Base discovery chance modified by intelligence and creativity
  const discoveryChance = CONFIG.invention.discoveryChance * (1 + intelligence + creativity);
  
  if (Math.random() > discoveryChance) {
    return { discovery: null, knowledge };
  }
  
  // What can they discover? Depends on current knowledge and available resources
  const discoveries: { name: string; field: keyof Knowledge; requires: () => boolean; boost: number }[] = [
    { 
      name: "basic toolmaking", 
      field: "toolmaking", 
      requires: () => (cell.hasSticks === true || cell.hasStones === true) && knowledge.toolmaking < 0.3,
      boost: 0.15 + creativity * 0.1
    },
    { 
      name: "advanced toolmaking", 
      field: "toolmaking", 
      requires: () => knowledge.toolmaking >= 0.3 && knowledge.toolmaking < 0.7,
      boost: 0.1 + intelligence * 0.1
    },
    { 
      name: "fire starting", 
      field: "firemaking", 
      requires: () => cell.hasSticks === true && knowledge.toolmaking >= 0.2 && knowledge.firemaking < 0.5,
      boost: 0.2 + creativity * 0.15
    },
    { 
      name: "food preservation", 
      field: "foodStorage", 
      requires: () => knowledge.toolmaking >= 0.3 && knowledge.foodStorage < 0.5,
      boost: 0.15 + intelligence * 0.1
    },
    { 
      name: "shelter construction", 
      field: "shelterBuilding", 
      requires: () => knowledge.toolmaking >= 0.4 && knowledge.shelterBuilding < 0.6,
      boost: 0.15 + creativity * 0.1
    },
    { 
      name: "hunting techniques", 
      field: "hunting", 
      requires: () => knowledge.toolmaking >= 0.2 && knowledge.hunting < 0.6,
      boost: 0.1 + intelligence * 0.1
    },
  ];
  
  // Filter to valid discoveries and pick one randomly
  const valid = discoveries.filter(d => d.requires());
  if (valid.length === 0) {
    return { discovery: null, knowledge };
  }
  
  const chosen = valid[Math.floor(Math.random() * valid.length)];
  knowledge[chosen.field] = Math.min(1.0, knowledge[chosen.field] + chosen.boost);
  
  return { discovery: chosen.name, knowledge };
}

// Craft a tool if agent has knowledge and resources
function tryCraftTool(agent: Agent, cell: Cell): Tool | null {
  const knowledge = agent.memory.knowledge;
  
  // Already has a tool? Can't carry another
  if (agent.tool && agent.tool.type !== "none") {
    return null;
  }
  
  // Need resources and knowledge
  if (!cell.hasSticks && !cell.hasStones) {
    return null;
  }
  
  // What can they craft?
  if (cell.hasStones && knowledge.toolmaking >= 0.3 && Math.random() < knowledge.toolmaking) {
    return { type: "stone_tool", durability: CONFIG.invention.toolDurability };
  }
  
  if (cell.hasSticks && knowledge.toolmaking >= 0.5 && knowledge.hunting >= 0.3 && Math.random() < knowledge.toolmaking) {
    return { type: "spear", durability: CONFIG.invention.toolDurability };
  }
  
  if (cell.hasSticks && knowledge.foodStorage >= 0.4 && Math.random() < knowledge.foodStorage) {
    return { type: "basket", durability: CONFIG.invention.toolDurability };
  }
  
  if (cell.hasSticks && knowledge.toolmaking >= 0.1) {
    return { type: "stick", durability: CONFIG.invention.toolDurability };
  }
  
  return null;
}

// Apply tool benefits during the simulation step
function applyToolBenefits(agent: Agent, cell: Cell, ateFood: boolean): { energyBonus: number; toolUsed: boolean } {
  if (!agent.tool || agent.tool.type === "none") {
    return { energyBonus: 0, toolUsed: false };
  }
  
  let bonus = 0;
  let used = false;
  
  switch (agent.tool.type) {
    case "spear":
      // Spear gives hunting bonus (chance for extra food)
      if (Math.random() < agent.memory.knowledge.hunting * 0.3) {
        bonus += 3; // Caught extra food through hunting
        used = true;
      }
      break;
    case "stone_tool":
      // Stone tool makes food processing more efficient
      if (ateFood) {
        bonus += 2;
        used = true;
      }
      break;
    case "basket":
      // Basket allows storing food (handled separately)
      break;
    case "stick":
      // Basic stick helps with foraging
      if (ateFood) {
        bonus += 1;
        used = true;
      }
      break;
  }
  
  return { energyBonus: bonus, toolUsed: used };
}

/**
 * Fertility factor based on age.
 * Returns 0-1 representing reproductive probability.
 * - 0 at/below minReproAgeYears
 * - Ramps up to 1 at prime age (~28 years)
 * - Ramps down to 0 at maxReproAgeYears
 */
function fertilityFactor(ageYears: number): number {
  const minAge = CONFIG.time.minReproAgeYears;
  const maxAge = CONFIG.time.maxReproAgeYears;
  const primeAge = 28; // Peak fertility age

  if (ageYears <= minAge || ageYears >= maxAge) {
    return 0;
  }

  if (ageYears <= primeAge) {
    // Ramp up from minAge to primeAge
    return (ageYears - minAge) / (primeAge - minAge);
  } else {
    // Ramp down from primeAge to maxAge
    return 1 - (ageYears - primeAge) / (maxAge - primeAge);
  }
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
  stateKey: string,
  allAgents: Agent[]
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

  // SOCIAL DRIVE BIAS: if not starving and socialDrive is significant, bias movement
  if (energy > hungryThreshold && Math.abs(genes.socialDrive - 0.5) > 0.1) {
    // Count nearby agents for each neighbor cell (2-cell radius)
    const neighborScores = neighbors.map(n => {
      let nearbyAgentCount = 0;
      for (const other of allAgents) {
        if (other.id === agent.id) continue;
        const dist = Math.abs(other.x - n.x) + Math.abs(other.y - n.y);
        if (dist <= 2) nearbyAgentCount++;
      }
      return { ...n, score: nearbyAgentCount };
    });

    // High socialDrive prefers cells near agents; low socialDrive avoids them
    const sortedByScore = [...neighborScores].sort((a, b) => {
      if (genes.socialDrive > 0.5) {
        return b.score - a.score; // Prefer cells with more nearby agents
      } else {
        return a.score - b.score; // Prefer cells with fewer nearby agents
      }
    });

    // 30% chance to follow social preference instead of RL
    if (Math.random() < 0.3 && sortedByScore.length > 0) {
      const choice = sortedByScore[0];
      return {
        dir: choice.dir,
        rule: genes.socialDrive > 0.5 ? "Social: seeking others" : "Social: avoiding others",
        action: choice.dir
      };
    }
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
 * Apply one simulation step, including reproduction + mutation + RL updates
 * 
 * CRITICAL BUG FIX #2: Added collision detection
 * Now tracks intended moves and prevents multiple agents from occupying the same cell
 */
function stepWorld(
  agents: Agent[],
  grid: Cell[][],
  foodRecycleSettings?: { enabled: boolean; maxFoodPerAgent: number }
): { agents: Agent[]; grid: Cell[][]; log: string[] } {
  const newGrid: Cell[][] = grid.map(row =>
    row.map(cell => ({ ...cell, agentId: undefined }))
  );

  const logs: string[] = [];
  const updatedAgents: Agent[] = [];

  let nextId = agents.reduce((max, a) => Math.max(max, a.id), 0) + 1;

  /**
   * CRITICAL BUG FIX #2: Agent collision handling
   * Track intended destinations to prevent multiple agents from moving to the same cell
   */
  const destinationMap = new Map<string, number>(); // "x,y" -> agentId
  
  // Track agents that have already reproduced this tick (each agent can only reproduce once)
  const usedForRepro = new Set<number>();

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
    const decision = decideMove(agent, grid, stateKey, agents);

    // Use type-safe direction application
    const newPos = applyDirection(agent.x, agent.y, decision.dir);

    agentMoves.push({ agent, newPos, decision, stateKey });
  }

  // Phase 2: Process moves with collision detection and terrain checks
  for (const { agent, newPos, decision, stateKey } of agentMoves) {
    const destKey = `${newPos.x},${newPos.y}`;
    
    // Track the final position (may change due to collision or terrain)
    let finalX = newPos.x;
    let finalY = newPos.y;
    
    // Check if target cell is water - prevent movement
    if (grid[newPos.y][newPos.x].terrain === "water") {
      logs.push(
        `Agent ${agent.id} blocked by water at (${newPos.x},${newPos.y}), stayed at (${agent.x},${agent.y})`
      );
      finalX = agent.x;
      finalY = agent.y;
    }
    // Check if another agent already claimed this destination
    else if (destinationMap.has(destKey)) {
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

    let newEnergy = agent.energy - CONFIG.simulation.baseEnergyCost;
    const cell = newGrid[finalY][finalX];
    let ateFood = false;
    let richBonus = 0;

    // Hazard terrain causes extra energy drain
    if (cell.terrain === "hazard") {
      newEnergy -= CONFIG.terrain.hazardDamage;
      logs.push(`Agent ${agent.id} takes ${CONFIG.terrain.hazardDamage} hazard damage at (${finalX},${finalY})`);
    }

    if (cell.food) {
      ateFood = true;
      cell.food = false;
      newEnergy += CONFIG.simulation.foodEnergyBonus;
      // Rich terrain gives extra energy when eating food
      if (cell.terrain === "rich") {
        richBonus = CONFIG.terrain.richFoodBonus;
        newEnergy += richBonus;
      }
    }    let reward = -0.5; // small negative for existing (encourages action)
    if (ateFood) reward += CONFIG.simulation.foodEnergyBonus + richBonus; // eating is good

    // === INVENTION SYSTEM INTEGRATION ===
    
    // Fire provides warmth (energy savings) if agent is on a cell with fire
    if (cell.fire && cell.fire.ticksRemaining > 0) {
      newEnergy += CONFIG.invention.fireWarmth;
      cell.fire.ticksRemaining--;
      if (cell.fire.ticksRemaining <= 0) {
        cell.fire = undefined;
        logs.push(`🔥 Fire at (${finalX},${finalY}) burned out`);
      }
    }
    
    // Shelter provides hazard protection
    if (cell.shelter && cell.terrain === "hazard") {
      // Refund some of the hazard damage
      newEnergy += CONFIG.terrain.hazardDamage * CONFIG.invention.shelterProtection;
    }
    
    // Try to discover new knowledge
    const discoveryResult = tryDiscovery(agent, cell);
    let agentKnowledge = discoveryResult.knowledge;
    if (discoveryResult.discovery) {
      logs.push(`💡 Agent ${agent.id} discovered: ${discoveryResult.discovery}!`);
      reward += CONFIG.invention.inventionReward;
    }
    
    // Try to craft a tool if on a resource cell
    let agentTool = agent.tool;
    if (cell.hasSticks || cell.hasStones) {
      const craftedTool = tryCraftTool({ ...agent, memory: { ...agent.memory, knowledge: agentKnowledge } }, cell);
      if (craftedTool) {
        agentTool = craftedTool;
        logs.push(`🔧 Agent ${agent.id} crafted: ${craftedTool.type}!`);
        reward += CONFIG.invention.inventionReward * 0.5;
        // Consume the resource
        if (craftedTool.type === "stone_tool" && cell.hasStones) {
          cell.hasStones = false;
        } else if (cell.hasSticks) {
          cell.hasSticks = false;
        }
      }
    }
    
    // Try to start a fire if agent has firemaking knowledge and sticks
    if (cell.hasSticks && !cell.fire && agentKnowledge.firemaking >= 0.3 && Math.random() < agentKnowledge.firemaking * 0.2) {
      cell.fire = { ticksRemaining: CONFIG.invention.fireDuration };
      cell.hasSticks = false;
      logs.push(`🔥 Agent ${agent.id} started a fire at (${finalX},${finalY})!`);
      reward += CONFIG.invention.inventionReward * 0.3;
    }
    
    // Try to build shelter if agent has knowledge and resources
    if (!cell.shelter && cell.hasSticks && cell.hasStones && agentKnowledge.shelterBuilding >= 0.4 && Math.random() < agentKnowledge.shelterBuilding * 0.15) {
      cell.shelter = { builderId: agent.id, durability: 100 };
      cell.hasSticks = false;
      cell.hasStones = false;
      logs.push(`🏠 Agent ${agent.id} built a shelter at (${finalX},${finalY})!`);
      reward += CONFIG.invention.inventionReward;
    }
    
    // Apply tool benefits
    const toolBenefits = applyToolBenefits({ ...agent, tool: agentTool }, cell, ateFood);
    newEnergy += toolBenefits.energyBonus;
    
    // Degrade tool if used
    if (toolBenefits.toolUsed && agentTool && agentTool.type !== "none") {
      agentTool = { ...agentTool, durability: agentTool.durability - 5 };
      if (agentTool.durability <= 0) {
        logs.push(`🔨 Agent ${agent.id}'s ${agentTool.type} broke!`);
        agentTool = undefined;
      }
    }
    
    // Basket allows storing food for later
    let agentStoredFood = agent.storedFood ?? 0;
    if (agentTool?.type === "basket" && ateFood && agentStoredFood < CONFIG.invention.basketCapacity) {
      // Store some of the food energy instead of consuming it all now
      const storeAmount = Math.min(2, CONFIG.invention.basketCapacity - agentStoredFood);
      agentStoredFood += storeAmount;
    }
    
    // Use stored food if hungry
    if (agentStoredFood > 0 && newEnergy < CONFIG.energy.hungryCutoffRatio) {
      newEnergy += 3; // eat from storage
      agentStoredFood--;
      logs.push(`🧺 Agent ${agent.id} ate from stored food`);
    }

    // === END INVENTION SYSTEM ===

    // Increment age
    const newAgeTicks = (agent.ageTicks ?? 0) + 1;
    const ageYears = newAgeTicks * CONFIG.time.yearsPerTick;

    // Decrement reproduction cooldown
    const newCooldown = agent.reproductionCooldown 
      ? Math.max(0, agent.reproductionCooldown - 1) 
      : 0;    let parentAgent: Agent = {
      ...agent,
      x: finalX,
      y: finalY,
      energy: newEnergy,
      lastRule: decision.rule,
      ageTicks: newAgeTicks,
      lastDirection: decision.dir,
      reproductionCooldown: newCooldown,
      pregnantWith: agent.pregnantWith, // carry forward pregnancy state
      tool: agentTool,
      storedFood: agentStoredFood,
      memory: { ...agent.memory, knowledge: agentKnowledge },
    };

    // PROCESS PREGNANCY - Check if gestation complete
    let gaveBirth = false;
    const currentPregnancy = parentAgent.pregnantWith;
    if (currentPregnancy) {
      const gestation = currentPregnancy.gestationTicks + 1;
      
      if (gestation >= CONFIG.reproduction.gestationTicks) {
        // Time to give birth! Find a spot for the child
        const adjacentPositions = [
          { x: finalX, y: finalY - 1 },
          { x: finalX, y: finalY + 1 },
          { x: finalX - 1, y: finalY },
          { x: finalX + 1, y: finalY }
        ].filter(
          p => p.x >= 0 && p.x < GRID_WIDTH && p.y >= 0 && p.y < GRID_HEIGHT
        );

        const birthSpots = adjacentPositions.filter(
          p =>
            newGrid[p.y][p.x].agentId === undefined &&
            newGrid[p.y][p.x].terrain !== "water" &&
            !destinationMap.has(`${p.x},${p.y}`)
        );

        if (birthSpots.length > 0) {
          const spot = birthSpots[randomInt(birthSpots.length)];
          
          // Child gets portion of mother's current energy (reduced cost to encourage reproduction)
          const childEnergy = Math.floor(parentAgent.energy * CONFIG.reproduction.energyCostRatio);
          parentAgent = { ...parentAgent, energy: parentAgent.energy - childEnergy };

          const child: Agent = {
            id: nextId++,
            x: spot.x,
            y: spot.y,
            energy: Math.max(childEnergy, 8), // minimum starting energy for babies
            sex: Math.random() < 0.5 ? "M" : "F",
            ageTicks: 0,
            genes: currentPregnancy.childGenes,
            // Inherit genetic memory: Q-table and partial knowledge from parent
            memory: { 
              qTable: { ...parentAgent.memory.qTable },
              knowledge: {
                toolmaking: parentAgent.memory.knowledge.toolmaking * CONFIG.invention.knowledgeTransferRate,
                firemaking: parentAgent.memory.knowledge.firemaking * CONFIG.invention.knowledgeTransferRate,
                foodStorage: parentAgent.memory.knowledge.foodStorage * CONFIG.invention.knowledgeTransferRate,
                shelterBuilding: parentAgent.memory.knowledge.shelterBuilding * CONFIG.invention.knowledgeTransferRate,
                hunting: parentAgent.memory.knowledge.hunting * CONFIG.invention.knowledgeTransferRate,
              },
            },
            lastRule: 'Born',
            reproductionCooldown: 0,
          };

          newGrid[spot.y][spot.x].agentId = child.id;
          destinationMap.set(`${spot.x},${spot.y}`, child.id);
          updatedAgents.push(child);

          logs.push(
            `🍼 Agent ${parentAgent.id}(${parentAgent.sex}) gave birth to ${child.id}(${child.sex}) at (${spot.x},${spot.y}), traitId ${child.genes.traitId}`
          );

          gaveBirth = true;
          // BIG reward for giving birth - this is the ultimate goal: genetic preservation!
          reward += CONFIG.simulation.reproductionReward + CONFIG.simulation.birthReward;
        } else {
          // No space - pregnancy continues (difficult birth)
          parentAgent = {
            ...parentAgent,
            pregnantWith: { ...currentPregnancy, gestationTicks: gestation },
          };
          logs.push(`Agent ${parentAgent.id} couldn't find space to give birth`);
        }

        // Clear pregnancy after birth
        if (gaveBirth) {
          parentAgent = { 
            ...parentAgent, 
            pregnantWith: undefined,
            reproductionCooldown: CONFIG.reproduction.cooldownTicks,
          };
        }
      } else {
        // Continue gestation
        parentAgent = {
          ...parentAgent,
          pregnantWith: { ...currentPregnancy, gestationTicks: gestation },
        };
      }
    }

    // REPRODUCTION (Sexual - requires opposite-sex mate) - MATING PHASE
    let reproduced = gaveBirth;

    // Age-based reproduction rules
    const ageYearsForRepro = parentAgent.ageTicks * CONFIG.time.yearsPerTick;
    const canReproduceByAge =
      ageYearsForRepro >= CONFIG.time.minReproAgeYears &&
      ageYearsForRepro <= CONFIG.time.maxReproAgeYears;

    // Risk tolerance affects effective reproduction threshold
    // Higher risk tolerance = lower effective threshold (willing to reproduce with less energy)
    const riskAdjustment = 1 - 0.3 * (parentAgent.genes.riskTolerance - 0.5);
    const effectiveReproThreshold = parentAgent.genes.reproductionThreshold * riskAdjustment;

    // Check if this agent can attempt mating:
    // - Not already used for repro this tick
    // - Within reproductive age
    // - Has enough energy
    // - Not already pregnant
    // - Not on cooldown
    const canAttemptMating = !usedForRepro.has(parentAgent.id) && 
      canReproduceByAge && 
      parentAgent.energy > effectiveReproThreshold &&
      !parentAgent.pregnantWith &&
      (parentAgent.reproductionCooldown ?? 0) === 0;

    if (canAttemptMating) {
      // Find adjacent agents of opposite sex with sufficient energy
      const adjacentPositions = [
        { x: finalX, y: finalY - 1 },
        { x: finalX, y: finalY + 1 },
        { x: finalX - 1, y: finalY },
        { x: finalX + 1, y: finalY }
      ].filter(
        p => p.x >= 0 && p.x < GRID_WIDTH && p.y >= 0 && p.y < GRID_HEIGHT
      );

      // Look for a mate - opposite sex agent with enough energy, not already used for repro
      let mate: Agent | undefined;
      for (const pos of adjacentPositions) {
        const mateId = newGrid[pos.y][pos.x].agentId;
        if (mateId !== undefined && !usedForRepro.has(mateId)) {
          // Check both updatedAgents (already processed) and original agents
          const potentialMate = 
            updatedAgents.find(a => a.id === mateId) || 
            agents.find(a => a.id === mateId);
          if (potentialMate) {
            const mateAgeYears = potentialMate.ageTicks * CONFIG.time.yearsPerTick;
            const mateCanReproByAge = mateAgeYears >= CONFIG.time.minReproAgeYears && 
              mateAgeYears <= CONFIG.time.maxReproAgeYears;
            const mateRiskAdjustment = 1 - 0.3 * (potentialMate.genes.riskTolerance - 0.5);
            const mateEffectiveThreshold = potentialMate.genes.reproductionThreshold * mateRiskAdjustment;
            
            // Mate must be opposite sex, fertile age, enough energy, not pregnant, not on cooldown
            if (
              potentialMate.sex !== parentAgent.sex &&
              mateCanReproByAge &&
              potentialMate.energy > mateEffectiveThreshold &&
              !potentialMate.pregnantWith &&
              (potentialMate.reproductionCooldown ?? 0) === 0
            ) {
              mate = potentialMate;
              break;
            }
          }
        }
      }

      if (mate) {
        // Calculate fertility factor for both parents
        const fertParent = fertilityFactor(ageYearsForRepro);
        const fertMate = fertilityFactor(mate.ageTicks * CONFIG.time.yearsPerTick);
        const combinedFert = Math.min(fertParent, fertMate);

        // Roll fertility check - reproduction more likely at prime ages
        if (Math.random() <= combinedFert) {
          // Mating successful! Determine which parent gets pregnant (female)
          const female = parentAgent.sex === "F" ? parentAgent : mate;
          const male = parentAgent.sex === "M" ? parentAgent : mate;
          
          // Combine genes from both parents for the future child
          const childGenes = combineGenes(female.genes, male.genes);
          
          // Both parents pay initial energy cost for mating (low cost to encourage mating)
          const matingEnergyCost = CONFIG.reproduction.matingEnergyCost;
          parentAgent = { ...parentAgent, energy: parentAgent.energy - matingEnergyCost };
          
          // Update mate's energy
          const mateInUpdated = updatedAgents.findIndex(a => a.id === mate!.id);
          if (mateInUpdated !== -1) {
            updatedAgents[mateInUpdated] = { 
              ...updatedAgents[mateInUpdated], 
              energy: updatedAgents[mateInUpdated].energy - matingEnergyCost 
            };
          }

          // Mark both parents as having mated this tick
          usedForRepro.add(parentAgent.id);
          usedForRepro.add(mate.id);

          // Set pregnancy on the female
          if (parentAgent.sex === "F") {
            parentAgent = {
              ...parentAgent,
              pregnantWith: {
                mateId: male.id,
                gestationTicks: 0,
                childGenes,
              },
            };
            logs.push(
              `💕 Agent ${parentAgent.id}(F) mated with ${male.id}(M) - now pregnant!`
            );
          } else {
            // Parent is male, update the female mate
            if (mateInUpdated !== -1) {
              updatedAgents[mateInUpdated] = {
                ...updatedAgents[mateInUpdated],
                pregnantWith: {
                  mateId: parentAgent.id,
                  gestationTicks: 0,
                  childGenes,
                },
              };
            }
            logs.push(
              `💕 Agent ${female.id}(F) mated with ${parentAgent.id}(M) - now pregnant!`
            );
          }

          // Male goes on cooldown
          if (parentAgent.sex === "M") {
            parentAgent = {
              ...parentAgent,
              reproductionCooldown: CONFIG.reproduction.cooldownTicks,
            };
          } else if (mateInUpdated !== -1) {
            updatedAgents[mateInUpdated] = {
              ...updatedAgents[mateInUpdated],
              reproductionCooldown: CONFIG.reproduction.cooldownTicks,
            };
          }

          reproduced = true;
          reward += CONFIG.simulation.matingReward; // Good reward for successful mating - passing genes is the goal!
        }
      }
    }

    // Determine if agent will die after this step (energy or old age)
    const isTooOld = ageYears > CONFIG.time.maxAgeYears;
    const isDeadAfterStep = parentAgent.energy <= 0 || isTooOld;
    const deathReason = isTooOld ? "old age" : "low energy";

    // Apply death penalty BEFORE Q-update so the agent learns from death
    // But keep it small - death is natural, the goal is reproduction not immortality
    if (isDeadAfterStep) {
      reward -= CONFIG.simulation.deathPenalty;
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

    parentAgent = { ...parentAgent, memory: { qTable: newQTable, knowledge: parentAgent.memory.knowledge } };

    if (!isDeadAfterStep) {
      updatedAgents.push(parentAgent);
      newGrid[parentAgent.y][parentAgent.x].agentId = parentAgent.id;

      logs.push(
        `Agent ${parentAgent.id} used ${decision.rule}, moved to (${parentAgent.x},${parentAgent.y})` +
          (ateFood ? ` and ate food (+${CONFIG.simulation.foodEnergyBonus}${richBonus > 0 ? `+${richBonus} rich` : ""} energy)` : "") +
          (reproduced ? " and reproduced (+2 reward)" : "") +
          `, energy now ${parentAgent.energy}, age ${ageYears.toFixed(1)}y, traitId=${parentAgent.genes.traitId}`
      );
    } else {
      logs.push(
        `Agent ${agent.id} died (${deathReason}) at (${finalX},${finalY}), age ${ageYears.toFixed(1)} years.`
      );
    }
  }

  let finalGrid = newGrid;

  /**
   * Food recycling: remove excess food when there are too few agents
   * This prevents the board from getting overcrowded with food
   */
  if (foodRecycleSettings?.enabled) {
    const recycleResult = recycleFoodIfNeeded(
      finalGrid, 
      updatedAgents.length, 
      foodRecycleSettings.maxFoodPerAgent
    );
    finalGrid = recycleResult.grid;
    if (recycleResult.removedCount > 0) {
      logs.push(`Food recycled: removed ${recycleResult.removedCount} excess food cells`);
    }
  }
  /**
   * CRITICAL BUG FIX #1: Food spawning bug
   * Previously, placeRandomFood returned a new grid but the result was discarded.
   * Now we properly use the returned grid to ensure food actually appears.
   */
  if (Math.random() < CONFIG.simulation.foodSpawnChance) {
    finalGrid = placeRandomFood(finalGrid, CONFIG.simulation.foodSpawnCount);
  }

  // Spawn resources (sticks and stones) periodically
  if (Math.random() < 0.3) { // 30% chance each tick to spawn some resources
    finalGrid = spawnResources(finalGrid);
  }

  return { agents: updatedAgents, grid: finalGrid, log: logs };
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

/**
 * AgentAvatar - SVG stick figure representing an agent
 * Visual indicators for sex, energy level, pregnancy, and direction
 */
const AgentAvatar: React.FC<{
  agent: Agent;
  size: number;
  isSelected: boolean;
  isWatched: boolean;
}> = ({ agent, size, isSelected, isWatched }) => {
  const traitColor = colorForTrait(agent.genes.traitId);
  const ageYears = agent.ageTicks * CONFIG.time.yearsPerTick;
  
  // Energy determines body opacity/glow
  const energyRatio = Math.min(agent.energy / 20, 1);
  const energyColor = energyRatio > 0.5 
    ? `rgba(100, 255, 100, ${0.3 + energyRatio * 0.4})` 
    : `rgba(255, 100, 100, ${0.3 + (1 - energyRatio) * 0.4})`;
  
  // Age affects posture (older = slightly bent)
  const ageFactor = Math.min(ageYears / CONFIG.time.maxAgeYears, 1);
  const headTilt = ageFactor * 2;
  
  // Sex determines body style
  const isFemale = agent.sex === "F";
  const isPregnant = agent.pregnantWith !== undefined;
  
  // Direction affects leg/arm pose
  const dir = agent.lastDirection || "stay";
  const isMoving = dir !== "stay";
  
  // Scale everything to fit in the cell
  const scale = size / 24;
  
  return (
    <svg 
      width={size} 
      height={size} 
      viewBox="0 0 24 24"
      style={{
        filter: isSelected 
          ? "drop-shadow(0 0 4px #ffeb3b)" 
          : isWatched 
          ? "drop-shadow(0 0 3px #ff5252)"
          : `drop-shadow(0 0 2px ${traitColor})`,
        transition: "all 0.15s ease"
      }}
    >
      {/* Energy glow background */}
      <circle cx="12" cy="12" r="10" fill={energyColor} opacity="0.3" />
      
      {/* Head */}
      <circle 
        cx={12 + headTilt} 
        cy="6" 
        r={isFemale ? 3.5 : 3} 
        fill={traitColor}
        stroke="#fff"
        strokeWidth="0.5"
      />
      
      {/* Eyes */}
      <circle cx={11 + headTilt} cy="5.5" r="0.6" fill="#222" />
      <circle cx={13 + headTilt} cy="5.5" r="0.6" fill="#222" />
      
      {/* Body */}
      <line 
        x1={12 + headTilt * 0.5} 
        y1="9" 
        x2={12 + headTilt} 
        y2={isFemale ? 14 : 15} 
        stroke={traitColor}
        strokeWidth={isFemale ? 2.5 : 2}
        strokeLinecap="round"
      />
      
      {/* Pregnancy indicator - belly bulge */}
      {isPregnant && (
        <ellipse 
          cx={12 + headTilt * 0.5 + 1} 
          cy="12" 
          rx="2.5" 
          ry="2" 
          fill={traitColor}
          stroke="#ffd700"
          strokeWidth="0.5"
        />
      )}
      
      {/* Female dress/skirt indicator */}
      {isFemale && (
        <polygon 
          points={`${10 + headTilt},14 ${14 + headTilt},14 ${15 + headTilt},18 ${9 + headTilt},18`}
          fill={traitColor}
          opacity="0.8"
        />
      )}
      
      {/* Arms - animated based on direction */}
      <line 
        x1={12 + headTilt * 0.5} 
        y1="10" 
        x2={isMoving && (dir === "left" || dir === "up") ? 7 : 8} 
        y2={isMoving ? 8 : 12} 
        stroke={traitColor}
        strokeWidth="1.5"
        strokeLinecap="round"
      />
      <line 
        x1={12 + headTilt * 0.5} 
        y1="10" 
        x2={isMoving && (dir === "right" || dir === "up") ? 17 : 16} 
        y2={isMoving ? 8 : 12} 
        stroke={traitColor}
        strokeWidth="1.5"
        strokeLinecap="round"
      />
      
      {/* Legs - animated based on direction */}
      {!isFemale && (
        <>
          <line 
            x1={12 + headTilt} 
            y1="15" 
            x2={isMoving && (dir === "up" || dir === "left") ? 9 : 10} 
            y2="21" 
            stroke={traitColor}
            strokeWidth="1.5"
            strokeLinecap="round"
          />
          <line 
            x1={12 + headTilt} 
            y1="15" 
            x2={isMoving && (dir === "up" || dir === "right") ? 15 : 14} 
            y2="21" 
            stroke={traitColor}
            strokeWidth="1.5"
            strokeLinecap="round"
          />
        </>
      )}
      {isFemale && (
        <>
          <line 
            x1={11 + headTilt} 
            y1="18" 
            x2={isMoving && dir === "left" ? 8 : 10} 
            y2="22" 
            stroke={traitColor}
            strokeWidth="1.5"
            strokeLinecap="round"
          />
          <line 
            x1={13 + headTilt} 
            y1="18" 
            x2={isMoving && dir === "right" ? 16 : 14} 
            y2="22" 
            stroke={traitColor}
            strokeWidth="1.5"
            strokeLinecap="round"
          />
        </>
      )}
      
      {/* Cooldown indicator (small timer icon) */}
      {agent.reproductionCooldown && agent.reproductionCooldown > 0 && (
        <circle 
          cx="19" 
          cy="5" 
          r="3" 
          fill="rgba(255,100,100,0.8)"
          stroke="#fff"
          strokeWidth="0.3"
        />
      )}
        {/* Young indicator (baby icon) */}
      {ageYears < CONFIG.time.minReproAgeYears && (
        <text x="2" y="8" fontSize="6" fill="#fff">👶</text>
      )}
      
      {/* Elder indicator */}
      {ageYears > CONFIG.time.maxReproAgeYears && (
        <text x="2" y="8" fontSize="5" fill="#fff">👴</text>
      )}
      
      {/* Tool indicator */}
      {agent.tool && agent.tool.type !== "none" && (
        <text x="17" y="20" fontSize="5" fill="#fff">
          {agent.tool.type === "spear" ? "🗡️" : 
           agent.tool.type === "stone_tool" ? "🪨" : 
           agent.tool.type === "basket" ? "🧺" : 
           agent.tool.type === "stick" ? "🪵" : 
           agent.tool.type === "fire" ? "🔥" : ""}
        </text>
      )}
      
      {/* Stored food indicator */}
      {agent.storedFood && agent.storedFood > 0 && (
        <text x="1" y="20" fontSize="5" fill="#fff">🍎{agent.storedFood}</text>
      )}
    </svg>
  );
};

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

// Download world state as JSON
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
 * Creates a fresh world state with terrain, food, resources, and agents to avoid duplication
 */
function initializeWorld(): { grid: Cell[][]; agents: Agent[] } {
  const empty = createEmptyGrid();
  const withTerrain = addRandomTerrain(empty);
  const withFood = placeRandomFood(withTerrain, INITIAL_FOOD);
  const withResources = spawnResources(withFood); // Add initial resources
  const agents = createInitialAgents(withResources);
  return { grid: withResources, agents };
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
  const [isRunning, setIsRunning] = useState(false);
  const [speedMs, setSpeedMs] = useState(400);
  const [watchedTraitId, setWatchedTraitId] = useState<number | null>(null);
  const [showStartupModal, setShowStartupModal] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  // Simulation settings state (user-adjustable)
  const [settings, setSettings] = useState<SimulationSettings>({
    initialAgents: CONFIG.simulation.initialAgents,
    initialFood: CONFIG.simulation.initialFood,
    foodSpawnChance: CONFIG.simulation.foodSpawnChance,
    foodSpawnCount: CONFIG.simulation.foodSpawnCount,
    foodRecycleEnabled: true,
    maxFoodPerAgent: 4, // allow up to 4 food cells per living agent
  });

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
    const copy: Cell[][] = grid.map(row =>
      row.map(cell => ({ ...cell, agentId: undefined } as Cell))
    );
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
    const { agents: newAgents, grid: newGrid, log: newLog } = stepWorld(
      agents,
      renderedGrid,
      { enabled: settings.foodRecycleEnabled, maxFoodPerAgent: settings.maxFoodPerAgent }
    );
    const newTick = tickRef.current + 1;
    setTick(newTick);
    setAgents(newAgents);
    setGrid(newGrid);
    setLog(prev => [...newLog, ...prev].slice(0, 80));
    pushHistory(newAgents, newTick);
  }, [agents, renderedGrid, pushHistory, settings.foodRecycleEnabled, settings.maxFoodPerAgent]);

  const handleReset = () => {
    const { grid: newGrid, agents: newAgents } = initializeWorld();
    setGrid(newGrid);
    setAgents(newAgents);
    setLog([]);
    setSelectedAgentId(null);
    setTick(0);
    setHistory([]);
    setIsRunning(false);
    setWatchedTraitId(null);
  };

  // Create a new world from current settings
  const createWorldFromSettings = useCallback(() => {
    const empty = createEmptyGrid();
    const withTerrain = addRandomTerrain(empty);
    const withFood = placeRandomFood(withTerrain, settings.initialFood);
    const withResources = spawnResources(withFood); // Add initial resources
    const newAgents = createInitialAgents(withResources, settings.initialAgents);
    setGrid(withResources);
    setAgents(newAgents);
    setLog([`New world created with ${settings.initialAgents} agents, ${settings.initialFood} food, and resources`]);
    setSelectedAgentId(null);
    setTick(0);
    setHistory([]);
    setIsRunning(false);
    setWatchedTraitId(null);
  }, [settings]);

  // God's Finger: Spawn replacement agents when population collapses
  const godsFinger = useCallback(() => {
    setAgents(prevAgents => {
      const agents = [...prevAgents];
      const gridCopy = grid.map(row => row.map(cell => ({ ...cell })));

      // If zero agents: spawn 1 male + 1 female anywhere empty
      if (agents.length === 0) {
        const empties: { x: number; y: number }[] = [];
        for (let y = 0; y < GRID_HEIGHT; y++)
          for (let x = 0; x < GRID_WIDTH; x++)
            if (!gridCopy[y][x].agentId && gridCopy[y][x].terrain !== "water") 
              empties.push({ x, y });

        if (empties.length < 2) return agents;

        const pos1 = empties[Math.floor(Math.random() * empties.length)];
        let pos2 = empties[Math.floor(Math.random() * empties.length)];
        // Ensure different positions
        while (pos2.x === pos1.x && pos2.y === pos1.y && empties.length > 1) {
          pos2 = empties[Math.floor(Math.random() * empties.length)];
        }

        const nextId = 1;

        const male: Agent = {
          id: nextId,
          x: pos1.x,
          y: pos1.y,
          energy: 20,
          sex: "M",
          ageTicks: 0,
          genes: createRandomGenes(),
          memory: createInitialMemory(),
          lastRule: "Spawned by God's Finger",
        };

        const female: Agent = {
          id: nextId + 1,
          x: pos2.x,
          y: pos2.y,
          energy: 20,
          sex: "F",
          ageTicks: 0,
          genes: createRandomGenes(),
          memory: createInitialMemory(),
          lastRule: "Spawned by God's Finger",
        };

        gridCopy[pos1.y][pos1.x].agentId = male.id;
        gridCopy[pos2.y][pos2.x].agentId = female.id;

        setGrid(gridCopy);
        setLog(prev => ["God's Finger: Spawned M + F agents.", ...prev]);
        return [male, female];
      }

      // If one agent: spawn an opposite-sex partner
      if (agents.length === 1) {
        const survivor = agents[0];
        const targetSex: "M" | "F" = survivor.sex === "M" ? "F" : "M";

        const empties: { x: number; y: number }[] = [];
        for (let y = 0; y < GRID_HEIGHT; y++)
          for (let x = 0; x < GRID_WIDTH; x++)
            if (!gridCopy[y][x].agentId && gridCopy[y][x].terrain !== "water") 
              empties.push({ x, y });

        if (empties.length === 0) return agents;

        const spot = empties[Math.floor(Math.random() * empties.length)];
        const nextId = survivor.id + 1;

        const partner: Agent = {
          id: nextId,
          x: spot.x,
          y: spot.y,
          energy: 20,
          sex: targetSex,
          ageTicks: 0,
          genes: createRandomGenes(),
          memory: createInitialMemory(),
          lastRule: "Spawned by God's Finger",
        };

        gridCopy[spot.y][spot.x].agentId = partner.id;

        setGrid(gridCopy);
        setLog(prev => [
          `God's Finger: Spawned ${targetSex} partner (ID ${partner.id}) for survivor ${survivor.id}.`,
          ...prev
        ]);

        return [...agents, partner];
      }

      // If 2+ agents: no intervention needed
      setLog(prev => ["God's Finger: No action (population > 1).", ...prev]);
      return agents;
    });
  }, [grid]);

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
        pregnantCount: 0,
        maleCount: 0,
        femaleCount: 0,
        avgAge: 0,
      };
    }

    const totalEnergy = agents.reduce((sum, a) => sum + a.energy, 0);
    const totalMutationRate = agents.reduce((sum, a) => sum + a.genes.mutationRate, 0);
    const totalReproThreshold = agents.reduce((sum, a) => sum + a.genes.reproductionThreshold, 0);
    const totalAge = agents.reduce((sum, a) => sum + a.ageTicks * CONFIG.time.yearsPerTick, 0);
    const uniqueTraits = new Set(agents.map(a => a.genes.traitId)).size;
    const pregnantCount = agents.filter(a => a.pregnantWith).length;
    const maleCount = agents.filter(a => a.sex === "M").length;
    const femaleCount = agents.filter(a => a.sex === "F").length;

    return {
      avgEnergy: totalEnergy / agents.length,
      avgMutationRate: totalMutationRate / agents.length,
      uniqueTraits,
      avgReproThreshold: totalReproThreshold / agents.length,
      pregnantCount,
      maleCount,
      femaleCount,
      avgAge: totalAge / agents.length,
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
  };  return (
    <div
      style={{
        display: "flex",
        minHeight: "100vh",
        fontFamily: "'Segoe UI', system-ui, -apple-system, sans-serif",
        background: "linear-gradient(135deg, #050814 0%, #0a1628 100%)",
        color: "#f4f4f4",
        padding: "20px",
        boxSizing: "border-box",
        position: "relative",
        justifyContent: "center",
        alignItems: "flex-start"
      }}
    >
      {/* Startup modal */}
      {showStartupModal && (
        <div
          style={{
            position: "fixed",
            inset: 0,
            background: "rgba(0,0,0,0.8)",
            backdropFilter: "blur(4px)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 100
          }}
        >
          <div
            style={{
              background: "linear-gradient(180deg, #1a2240 0%, #151a30 100%)",
              padding: "28px",
              borderRadius: 16,
              border: "1px solid #334",
              maxWidth: 380,
              boxShadow: "0 20px 60px rgba(0,0,0,0.5)"
            }}
          >
            <h3 style={{ margin: "0 0 12px 0", fontSize: 22 }}>Welcome to the World</h3>
            <p style={{ fontSize: 14, opacity: 0.85, lineHeight: 1.5, margin: "0 0 20px 0" }}>
              Start a new world, or load a previously saved world (JSON file).
            </p>
            <div style={{ display: "flex", gap: 12 }}>
              <button 
                onClick={handleStartupNew}
                style={{
                  flex: 1,
                  padding: "10px 16px",
                  fontSize: 14,
                  fontWeight: 600,
                  background: "linear-gradient(180deg, #4a6cf7 0%, #3b5ce4 100%)",
                  border: "none",
                  borderRadius: 8,
                  color: "#fff",
                  cursor: "pointer"
                }}
              >
                New World
              </button>
              <button 
                onClick={handleStartupLoad}
                style={{
                  flex: 1,
                  padding: "10px 16px",
                  fontSize: 14,
                  fontWeight: 600,
                  background: "linear-gradient(180deg, #2a3a5a 0%, #1e2a45 100%)",
                  border: "1px solid #445",
                  borderRadius: 8,
                  color: "#fff",
                  cursor: "pointer"
                }}              >
                Load from File
              </button>
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
      />      {/* Centered inner container */}
      <div
        style={{
          display: "flex",
          maxWidth: 1200,
          width: "100%",
          gap: "24px"
        }}
      >
        {/* LEFT: World grid */}
        <div style={{ flex: "0 0 auto" }}>
          <h2 style={{ margin: "0 0 16px 0", fontSize: 20, fontWeight: 600 }}>
            NPC World – Genetic Memory + RL
          </h2>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: `repeat(${GRID_WIDTH}, 32px)`,
              gridTemplateRows: `repeat(${GRID_HEIGHT}, 32px)`,
              gap: "2px",
              border: "2px solid #2a3a5a",
              padding: "10px",
              background: "linear-gradient(180deg, #0d1220 0%, #080c18 100%)",
              borderRadius: 16,
              boxShadow: "0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.05)"
            }}
          >
          {renderedGrid.map((row, y) =>
            row.map((cell, x) => {
              // PERFORMANCE IMPROVEMENT: Use agentMap for O(1) lookup instead of O(n) find
              const agent = cell.agentId ? agentMap.get(cell.agentId) : undefined;
              const isSelected = selectedAgentId === agent?.id;
              const isWatched =
                agent && watchedTraitId !== null && agent.genes.traitId === watchedTraitId;

              // Determine background color based on terrain
              let bg = "#1a2035"; // plain terrain default
              let terrainIcon = "";
              if (cell.terrain === "water") {
                bg = "linear-gradient(135deg, #1a4a7a 0%, #0d2840 100%)";
                terrainIcon = "🌊";
              } else if (cell.terrain === "rich") {
                bg = "linear-gradient(135deg, #3d4a2a 0%, #2a3520 100%)";
              } else if (cell.terrain === "hazard") {
                bg = "linear-gradient(135deg, #5a2a2a 0%, #3a1a1a 100%)";
                terrainIcon = "⚠️";
              }
              
              // Food styling
              if (cell.food && !agent) {
                bg = cell.terrain === "rich" 
                  ? "linear-gradient(135deg, #4caf50 0%, #2e7d32 100%)" 
                  : "linear-gradient(135deg, #66bb6a 0%, #388e3c 100%)";
              }

              return (
                <div
                  key={`${x}-${y}`}
                  onClick={() => agent && setSelectedAgentId(agent.id)}
                  style={{
                    width: "32px",
                    height: "32px",
                    borderRadius: "6px",
                    background: agent ? "transparent" : bg,
                    border: isSelected
                      ? "2px solid #ffeb3b"
                      : isWatched
                      ? "2px solid #ff5252"
                      : "1px solid rgba(255,255,255,0.08)",
                    boxSizing: "border-box",
                    cursor: agent ? "pointer" : "default",
                    transition: "all 0.15s ease",
                    boxShadow: agent 
                      ? "inset 0 0 8px rgba(0,0,0,0.3)" 
                      : cell.food 
                      ? "0 2px 8px rgba(76,175,80,0.3)"
                      : "none",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    position: "relative",
                    overflow: "hidden",
                  }}
                  title={
                    agent
                      ? `Agent ${agent.id} (${agent.sex}) – energy ${agent.energy}, age ${(agent.ageTicks * CONFIG.time.yearsPerTick).toFixed(0)}y${agent.pregnantWith ? " 🤰" : ""}`
                      : cell.food
                      ? `Food${cell.terrain === "rich" ? " (rich soil +bonus)" : ""}`
                      : cell.terrain !== "plain"
                      ? `Terrain: ${cell.terrain}`
                      : ""
                  }
                >                  {/* Terrain indicator (background) */}
                  {!agent && !cell.food && terrainIcon && (
                    <span style={{ fontSize: 12, opacity: 0.5 }}>{terrainIcon}</span>
                  )}
                  
                  {/* Resource indicators (sticks/stones) */}
                  {!agent && !cell.food && (cell.hasSticks || cell.hasStones) && (
                    <div style={{ 
                      position: "absolute", 
                      bottom: 1, 
                      left: 1, 
                      fontSize: 8,
                      display: "flex",
                      gap: 1
                    }}>
                      {cell.hasSticks && <span title="Sticks">🪵</span>}
                      {cell.hasStones && <span title="Stones">🪨</span>}
                    </div>
                  )}
                  
                  {/* Fire indicator */}
                  {cell.fire && cell.fire.ticksRemaining > 0 && (
                    <div style={{ 
                      position: "absolute", 
                      top: 1, 
                      right: 1, 
                      fontSize: 10,
                      animation: "pulse 0.5s infinite"
                    }} title={`Fire (${cell.fire.ticksRemaining} ticks)`}>
                      🔥
                    </div>
                  )}
                  
                  {/* Shelter indicator */}
                  {cell.shelter && (
                    <div style={{ 
                      position: "absolute", 
                      top: 1, 
                      left: 1, 
                      fontSize: 10,
                      opacity: 0.8
                    }} title={`Shelter (${cell.shelter.durability}% durability)`}>
                      🏠
                    </div>
                  )}
                  
                  {/* Food icon */}
                  {cell.food && !agent && (
                    <span style={{ fontSize: 14 }}>🍎</span>
                  )}
                  
                  {/* Agent avatar */}
                  {agent && (
                    <AgentAvatar 
                      agent={agent} 
                      size={28} 
                      isSelected={isSelected} 
                      isWatched={isWatched ?? false}
                    />
                  )}
                  
                  {/* Agent + food indicator */}
                  {agent && cell.food && (
                    <div style={{
                      position: "absolute",
                      bottom: 0,
                      right: 0,
                      fontSize: 8,
                      background: "rgba(0,0,0,0.5)",
                      borderRadius: 2,
                      padding: "0 2px",
                    }}>
                      🍎
                    </div>
                  )}
                </div>
              );
            })
          )}
          </div>

          <div style={{ marginTop: "16px", display: "flex", flexWrap: "wrap", gap: "8px" }}>
            <button 
              onClick={handleStep} 
              style={{
                padding: "8px 16px",
                fontSize: 13,
                fontWeight: 500,
                background: "linear-gradient(180deg, #3a4a6a 0%, #2a3a55 100%)",
                border: "1px solid #445",
                borderRadius: 6,
                color: "#fff",
                cursor: "pointer"
              }}
            >
              Step
            </button>
            <button
              onClick={() => setIsRunning(r => !r)}
              style={{
                padding: "8px 16px",
                fontSize: 13,
                fontWeight: 500,
                background: isRunning 
                  ? "linear-gradient(180deg, #c9a227 0%, #a88620 100%)" 
                  : "linear-gradient(180deg, #4a6cf7 0%, #3b5ce4 100%)",
                border: "none",
                borderRadius: 6,
                color: "#fff",
                cursor: "pointer"
              }}
            >
              {isRunning ? "Pause" : "Play"}
            </button>
            <button 
              onClick={handleReset}
              style={{
                padding: "8px 16px",
                fontSize: 13,
                fontWeight: 500,
                background: "linear-gradient(180deg, #3a4a6a 0%, #2a3a55 100%)",
                border: "1px solid #445",
                borderRadius: 6,
                color: "#fff",
                cursor: "pointer"
              }}
            >
              Reset
            </button>
            <button
              onClick={() => downloadWorld({ grid, agents, tick, history })}
              style={{
                padding: "8px 16px",
                fontSize: 13,
                fontWeight: 500,
                background: "linear-gradient(180deg, #2d5a3d 0%, #234530 100%)",
                border: "1px solid #3a5",
                borderRadius: 6,
                color: "#fff",
                cursor: "pointer"
              }}
            >
              Save
            </button>
            <button
              onClick={() => fileInputRef.current?.click()}
              style={{
                padding: "8px 16px",
                fontSize: 13,
                fontWeight: 500,
                background: "linear-gradient(180deg, #3a4a6a 0%, #2a3a55 100%)",
                border: "1px solid #445",
                borderRadius: 6,
                color: "#fff",
                cursor: "pointer"
              }}
            >
              Load
            </button>
            <button
              onClick={godsFinger}
              style={{
                padding: "8px 16px",
                fontSize: 13,
                fontWeight: 500,
                background: "linear-gradient(180deg, #6a3a6a 0%, #553055 100%)",
                border: "1px solid #858",
                borderRadius: 6,
                color: "#fff",
                cursor: "pointer"
              }}
            >
              God&apos;s Finger
            </button>
          </div>
          
          <div style={{ marginTop: 16, fontSize: 13 }}>
            <label style={{ display: "flex", alignItems: "center", gap: 10 }}>
              Speed:
              <input
                type="range"
                min={100}
                max={1200}
                step={100}
                value={speedMs}
                onChange={e => setSpeedMs(Number(e.target.value))}
                style={{ flex: 1, maxWidth: 150 }}
              />
              <span style={{ opacity: 0.7, minWidth: 70 }}>{speedMs} ms</span>
            </label>
          </div>
          
          <div style={{ 
            marginTop: 12, 
            padding: "10px 12px", 
            background: "rgba(255,255,255,0.05)", 
            borderRadius: 8,
            fontSize: 13
          }}>
            <div style={{ fontWeight: 600, marginBottom: 4 }}>
              Tick: {tick} &nbsp;|&nbsp; Agents: {agents.length}
            </div>
            <div style={{ opacity: 0.6, fontSize: 11 }}>
              Shortcuts: Space (Play/Pause), → (Step), R (Reset)
            </div>
          </div>
          
          {loadError && (
            <div style={{ 
              marginTop: 12, 
              padding: "10px 12px", 
              background: "linear-gradient(180deg, #6b1a1a 0%, #4a1212 100%)", 
                            borderRadius: 8, 
              fontSize: 12,
              border: "1px solid #933"
            }}>              Error: {loadError}
            </div>
          )}
        </div>        {/* RIGHT: Side panel */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0 }}>
        {/* Statistics Display */}
        <div
          style={{
            marginBottom: "12px",
            padding: "12px",
            background: "linear-gradient(180deg, #151a30 0%, #0f1422 100%)",
            borderRadius: "10px",
            border: "1px solid #2a3a5a"
          }}
        >
          <h3 style={{ margin: "0 0 10px 0", fontSize: 15 }}>📊 Statistics</h3>
          <div style={{ fontSize: "0.85em", display: "grid", gridTemplateColumns: "1fr 1fr", gap: "6px" }}>
            <div>
              <span style={{ opacity: 0.7 }}>Population:</span>{" "}
              <strong>{stats.maleCount}♂ + {stats.femaleCount}♀</strong>
            </div>
            <div>
              <span style={{ opacity: 0.7 }}>Pregnant:</span>{" "}
              <strong style={{ color: stats.pregnantCount > 0 ? "#ffd700" : "inherit" }}>
                {stats.pregnantCount} 🤰
              </strong>
            </div>
            <div>
              <span style={{ opacity: 0.7 }}>Avg Energy:</span>{" "}
              <strong style={{ color: stats.avgEnergy < 8 ? "#ff5252" : "#4caf50" }}>
                {stats.avgEnergy.toFixed(1)}
              </strong>
            </div>
            <div>
              <span style={{ opacity: 0.7 }}>Avg Age:</span>{" "}
              <strong>{stats.avgAge.toFixed(0)}y</strong>
            </div>
            <div>
              <span style={{ opacity: 0.7 }}>Mutation Rate:</span>{" "}
              <strong>{(stats.avgMutationRate * 100).toFixed(1)}%</strong>
            </div>
            <div>
              <span style={{ opacity: 0.7 }}>Unique Traits:</span>{" "}
              <strong>{stats.uniqueTraits}</strong>
            </div>
          </div>
        </div>

        {/* Simulation Settings */}
        <div
          style={{
            marginBottom: "12px",
            padding: "12px",
            background: "linear-gradient(180deg, #151a30 0%, #0f1422 100%)",
            borderRadius: "10px",
            border: "1px solid #2a3a5a"
          }}
        >
          <h3 style={{ margin: "0 0 10px 0", fontSize: 15 }}>⚙️ Settings</h3>
          <div style={{ fontSize: "0.85em", display: "grid", gridTemplateColumns: "1fr 1fr", gap: "8px" }}>
            <label>
              Initial Agents:
              <input
                type="number"
                min={1}
                max={50}
                value={settings.initialAgents}
                onChange={e => setSettings(s => ({ ...s, initialAgents: Number(e.target.value) }))}
                style={{ width: "60px", marginLeft: 8 }}
              />
            </label>
            <label>
              Initial Food:
              <input
                type="number"
                min={1}
                max={100}
                value={settings.initialFood}
                onChange={e => setSettings(s => ({ ...s, initialFood: Number(e.target.value) }))}
                style={{ width: "60px", marginLeft: 8 }}
              />
            </label>
            <label>
              Food Spawn %:
              <input
                type="number"
                min={0}
                max={100}
                step={5}
                value={Math.round(settings.foodSpawnChance * 100)}
                onChange={e => setSettings(s => ({ ...s, foodSpawnChance: Number(e.target.value) / 100 }))}
                style={{ width: "60px", marginLeft: 8 }}
              />
            </label>
            <label>
              Food/Spawn:
              <input
                type="number"
                min={1}
                max={10}
                value={settings.foodSpawnCount}
                onChange={e => setSettings(s => ({ ...s, foodSpawnCount: Number(e.target.value) }))}
                style={{ width: "60px", marginLeft: 8 }}
              />
            </label>
            <label style={{ display: "flex", alignItems: "center" }}>
              <input
                type="checkbox"
                checked={settings.foodRecycleEnabled}
                onChange={e => setSettings(s => ({ ...s, foodRecycleEnabled: e.target.checked }))}
                style={{ marginRight: 6 }}
              />
              Food Recycle
            </label>
            <label>
              Max Food/Agent:
              <input
                type="number"
                min={1}
                max={20}
                value={settings.maxFoodPerAgent}
                onChange={e => setSettings(s => ({ ...s, maxFoodPerAgent: Number(e.target.value) }))}
                style={{ width: "60px", marginLeft: 8 }}
                disabled={!settings.foodRecycleEnabled}
              />
            </label>
          </div>
          <button
            onClick={createWorldFromSettings}
            style={{ 
              marginTop: 10, 
              width: "100%",
              padding: "8px 16px",
              fontSize: 13,
              fontWeight: 500,
              background: "linear-gradient(180deg, #4a6cf7 0%, #3b5ce4 100%)",
              border: "none",
              borderRadius: 6,
              color: "#fff",
              cursor: "pointer"
            }}
          >
            Create New World with Settings
          </button>
        </div>        {/* Agent inspector */}
        <div
          style={{
            marginBottom: "12px",
            padding: "12px",
            background: "linear-gradient(180deg, #151a30 0%, #0f1422 100%)",
            borderRadius: "10px",
            border: "1px solid #2a3a5a"
          }}
        >
          <h3 style={{ margin: "0 0 10px 0", fontSize: 15 }}>🔍 Agent Inspector</h3>
          {selectedAgent ? (
            <>
              <p>
                <strong>ID:</strong> {selectedAgent.id} | <strong>Sex:</strong> {selectedAgent.sex}
                {selectedAgent.pregnantWith && <span style={{ marginLeft: 8, color: "#ffd700" }}>🤰 Pregnant</span>}
              </p>
              <p>
                <strong>Position:</strong> ({selectedAgent.x},{selectedAgent.y})
              </p>              <p>
                <strong>Energy:</strong> {selectedAgent.energy}
                {selectedAgent.energy <= 5 && <span style={{ marginLeft: 8, color: "#ff5252" }}>⚠️ Low</span>}
              </p>
              <p>
                <strong>Age:</strong> {selectedAgent.ageTicks} ticks (~
                {(selectedAgent.ageTicks * CONFIG.time.yearsPerTick).toFixed(1)} years)
                {(selectedAgent.ageTicks * CONFIG.time.yearsPerTick) < CONFIG.time.minReproAgeYears && 
                  <span style={{ marginLeft: 8, color: "#64b5f6" }}>👶 Young</span>}
                {(selectedAgent.ageTicks * CONFIG.time.yearsPerTick) > CONFIG.time.maxReproAgeYears && 
                  <span style={{ marginLeft: 8, color: "#9e9e9e" }}>👴 Elder</span>}
              </p>
              {/* Reproduction status */}
              {selectedAgent.pregnantWith && (
                <p style={{ background: "rgba(255,215,0,0.1)", padding: "4px 8px", borderRadius: 4 }}>
                  <strong>🤰 Gestation:</strong> {selectedAgent.pregnantWith.gestationTicks}/{CONFIG.reproduction.gestationTicks} ticks
                  <br />
                  <small>Father: Agent #{selectedAgent.pregnantWith.mateId}</small>
                </p>
              )}
              {selectedAgent.reproductionCooldown && selectedAgent.reproductionCooldown > 0 && (
                <p style={{ background: "rgba(255,100,100,0.1)", padding: "4px 8px", borderRadius: 4 }}>
                  <strong>⏳ Cooldown:</strong> {selectedAgent.reproductionCooldown} ticks
                </p>
              )}
              <p>
                <strong>Last Rule:</strong>{" "}
                {selectedAgent.lastRule}
              </p>              <p>
                <strong>Genes:</strong>
                <br />
                foodPref={selectedAgent.genes.foodPreference.toFixed(2)},{" "}
                explore={selectedAgent.genes.exploration.toFixed(2)},{" "}
                reproThresh={selectedAgent.genes.reproductionThreshold.toFixed(1)},{" "}
                mutationRate={selectedAgent.genes.mutationRate.toFixed(2)}
                <br />
                <strong>Intelligence:</strong> {selectedAgent.genes.intelligence.toFixed(2)},{" "}
                <strong>Creativity:</strong> {selectedAgent.genes.creativity.toFixed(2)}
                <br />
                <strong>Personality:</strong> riskTolerance={selectedAgent.genes.riskTolerance.toFixed(2)},{" "}
                socialDrive={selectedAgent.genes.socialDrive.toFixed(2)}
                <br />
                traitId={selectedAgent.genes.traitId}
              </p>
              {/* Tool and Inventory */}
              {(selectedAgent.tool || selectedAgent.storedFood) && (
                <p style={{ background: "rgba(100,200,255,0.1)", padding: "4px 8px", borderRadius: 4 }}>
                  <strong>🎒 Inventory:</strong>
                  {selectedAgent.tool && selectedAgent.tool.type !== "none" && (
                    <>
                      <br />
                      Tool: {selectedAgent.tool.type} ({selectedAgent.tool.durability}% durability)
                    </>
                  )}
                  {selectedAgent.storedFood && selectedAgent.storedFood > 0 && (
                    <>
                      <br />
                      Stored Food: {selectedAgent.storedFood}/{CONFIG.invention.basketCapacity}
                    </>
                  )}
                </p>
              )}
              {/* Knowledge */}
              {selectedAgent.memory.knowledge && (
                <p style={{ background: "rgba(255,200,100,0.1)", padding: "4px 8px", borderRadius: 4, fontSize: "0.85em" }}>
                  <strong>📚 Knowledge:</strong>
                  <br />
                  🔧 Toolmaking: {(selectedAgent.memory.knowledge.toolmaking * 100).toFixed(0)}%
                  {" | "}🔥 Fire: {(selectedAgent.memory.knowledge.firemaking * 100).toFixed(0)}%
                  <br />
                  🏠 Shelter: {(selectedAgent.memory.knowledge.shelterBuilding * 100).toFixed(0)}%
                  {" | "}🍖 Food Storage: {(selectedAgent.memory.knowledge.foodStorage * 100).toFixed(0)}%
                  <br />
                  🏹 Hunting: {(selectedAgent.memory.knowledge.hunting * 100).toFixed(0)}%
                </p>
              )}
            </>) : (
            <p>Click an agent in the grid to inspect its genetic memory.</p>
          )}
        </div>        {/* Rules + RL + evolution description */}
        <div
          style={{
            marginBottom: "12px",
            padding: "12px",
            background: "linear-gradient(180deg, #151a30 0%, #0f1422 100%)",
            borderRadius: "10px",
            border: "1px solid #2a3a5a"
          }}
        >          <h3 style={{ margin: "0 0 10px 0", fontSize: 15 }}>🧬 How It Works</h3>
          <ul style={{ fontSize: "0.8em", margin: 0, paddingLeft: 18, lineHeight: 1.6 }}>
            <li>
              <strong>Survival:</strong> Hungry agents seek food. Hazard terrain drains energy.
            </li>
            <li>
              <strong>Learning:</strong> Q-learning adapts behavior based on rewards.
            </li>
            <li>
              <strong>Mating:</strong> Adjacent ♂+♀ with enough energy may mate.
            </li>
            <li>
              <strong>Pregnancy:</strong> Females carry children for {CONFIG.reproduction.gestationTicks} ticks.
            </li>
            <li>
              <strong>Genetics:</strong> Children inherit blended genes with mutations.
            </li>
            <li>
              <strong>Aging:</strong> Fertility peaks at ~28y, death at {CONFIG.time.maxAgeYears}y.
            </li>
            <li>
              <strong>Inventions:</strong> Agents discover tools, fire, and shelter based on intelligence.
            </li>
            <li>
              <strong>Knowledge:</strong> Skills pass to children at {(CONFIG.invention.knowledgeTransferRate * 100)}% rate.
            </li>
          </ul>
        </div>{/* Charts */}
        <div
          style={{
            marginBottom: "12px",
            padding: "12px",
            background: "linear-gradient(180deg, #151a30 0%, #0f1422 100%)",
            borderRadius: "10px",
            border: "1px solid #2a3a5a"
          }}
        >
          <h3 style={{ margin: "0 0 10px 0", fontSize: 15 }}>📈 Population</h3>
          <PopulationChart history={history} />
        </div>        <div
          style={{
            marginBottom: "12px",
            padding: "12px",
            background: "linear-gradient(180deg, #151a30 0%, #0f1422 100%)",
            borderRadius: "10px",
            border: "1px solid #2a3a5a"
          }}
        >
          <h3 style={{ margin: "0 0 10px 0", fontSize: 15 }}>
            🧬 Trait Distribution
            {watchedTraitId !== null && (
              <span style={{ fontSize: 12, marginLeft: 8, opacity: 0.7 }}>
                Watching traitId {watchedTraitId}
              </span>
            )}
          </h3>
          <TraitChart
            last={lastHistory}
            watchedTraitId={watchedTraitId}
            onSelectTrait={tid => setWatchedTraitId(tid)}
          />
        </div>        {/* Log */}
        <div
          style={{
            flex: 1,
            padding: "12px",
            background: "linear-gradient(180deg, #151a30 0%, #0f1422 100%)",
            borderRadius: "10px",
            border: "1px solid #2a3a5a",
            overflowY: "auto",
            maxHeight: 200
          }}
        >
          <h3 style={{ margin: "0 0 10px 0", fontSize: 15 }}>📜 Action Log</h3>
          {log.length === 0 ? (
            <p style={{ opacity: 0.6, fontSize: 13 }}>No steps yet. Press "Step" or "Play" to advance the world.</p>
          ) : (
            <ul style={{ paddingLeft: "18px", margin: 0 }}>              {log.map((entry, i) => (
                <li key={i} style={{ fontSize: "0.75em", opacity: i === 0 ? 1 : 0.7, marginBottom: 2 }}>
                  {entry}
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
      </div>
    </div>
  );
};

export default App;