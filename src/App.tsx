// Contact: Name: dtay83 <dartey.banahene@gmail.com>
import React, { useState, useMemo, useEffect, useRef, useCallback } from "react";

type Direction = "up" | "down" | "left" | "right" | "stay";

interface Genes {
  foodPreference: number;        // 0–1: prioritize food when hungry
  exploration: number;           // 0–1: how often they wander
  reproductionThreshold: number; // energia needed to reproduce
  mutationRate: number;          // 0–1: chance each gene mutates
  traitId: number;               // lineage / random trait marker
  riskTolerance: number;         // 0–1: willingness to act with lower energy
  socialDrive: number;           // 0–1: preference for being near other agents
}

interface Memory {
  qTable: Record<string, number>; // RL value table: (state|action) -> Q
}

interface Agent {
  id: number;
  x: number;
  y: number;
  energy: number; // "energia"
  sex: "M" | "F";
  ageTicks: number;
  genes: Genes;
  memory: Memory;
  lastRule?: string;
}

interface Cell {
  food: boolean;
  agentId?: number;
  terrain: "plain" | "water" | "rich";
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
    foodSpawnChance: 0.75,
    foodSpawnCount: 2,
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
    reproductionThreshold: { min: 12, max: 20 },
    mutationRate: { min: 0.1, max: 0.3 },
    riskTolerance: { min: 0.2, max: 0.8 },
    socialDrive: { min: 0.2, max: 0.8 },
    mutation: {
      foodPreferenceMagnitude: 0.15,
      explorationMagnitude: 0.2,
      reproductionThresholdMagnitude: 3,
      mutationRateMagnitude: 0.05,
      riskToleranceMagnitude: 0.15,
      socialDriveMagnitude: 0.15,
      newTraitChance: 0.08,
      mutationRateEvolveMin: 0.01,
      mutationRateEvolveMax: 0.6,
      reproductionThresholdMin: 8,
      reproductionThresholdMax: 30,
    },
  },energy: {
    hungryCutoffRatio: 6,
    initialMin: 10,
    initialMax: 16,
  },
  time: {
    yearsPerTick: 0.5,       // conversion from ticks → years
    minReproAgeYears: 20,    // youngest reproductive age
    maxReproAgeYears: 120,   // oldest reproductive age
    maxAgeYears: 200         // lifespan limit
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
 * ~5% water cells, ~10% rich cells
 */
function addRandomTerrain(grid: Cell[][]): Cell[][] {
  const copy = grid.map(row => row.map(cell => ({ ...cell })));
  for (let y = 0; y < GRID_HEIGHT; y++) {
    for (let x = 0; x < GRID_WIDTH; x++) {
      const roll = Math.random();
      if (roll < 0.05) {
        copy[y][x].terrain = "water";
      } else if (roll < 0.15) {
        copy[y][x].terrain = "rich";
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
      memory: { qTable: {} },
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

  return {
    foodPreference,
    exploration,
    reproductionThreshold,
    mutationRate: newMutationRate,
    traitId,
    riskTolerance,
    socialDrive,
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
  };
  
  // Apply mutation to the combined genes
  return mutateGenes(combinedGenes);
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

    if (cell.food) {
      ateFood = true;
      cell.food = false;
      newEnergy += CONFIG.simulation.foodEnergyBonus;
      // Rich terrain gives extra energy when eating food
      if (cell.terrain === "rich") {
        richBonus = 2;
        newEnergy += richBonus;
      }
    }

    let reward = -1;
    if (ateFood) reward += CONFIG.simulation.foodEnergyBonus + richBonus;

    // Increment age
    const newAgeTicks = (agent.ageTicks ?? 0) + 1;
    const ageYears = newAgeTicks * CONFIG.time.yearsPerTick;

    let parentAgent: Agent = {
      ...agent,
      x: finalX,
      y: finalY,
      energy: newEnergy,
      lastRule: decision.rule,
      ageTicks: newAgeTicks,
    };

    // REPRODUCTION (Sexual - requires opposite-sex mate)
    let reproduced = false;

    // Age-based reproduction rules
    const ageYearsForRepro = parentAgent.ageTicks * CONFIG.time.yearsPerTick;
    const canReproduceByAge =
      ageYearsForRepro >= CONFIG.time.minReproAgeYears &&
      ageYearsForRepro <= CONFIG.time.maxReproAgeYears;

    // Risk tolerance affects effective reproduction threshold
    // Higher risk tolerance = lower effective threshold (willing to reproduce with less energy)
    const riskAdjustment = 1 - 0.3 * (parentAgent.genes.riskTolerance - 0.5);
    const effectiveReproThreshold = parentAgent.genes.reproductionThreshold * riskAdjustment;

    // Check if this agent hasn't already reproduced this tick
    const canAttemptRepro = !usedForRepro.has(parentAgent.id) && 
      canReproduceByAge && 
      parentAgent.energy > effectiveReproThreshold;

    if (canAttemptRepro) {
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
            
            if (
              potentialMate.sex !== parentAgent.sex &&
              mateCanReproByAge &&
              potentialMate.energy > mateEffectiveThreshold
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
          // Find empty spots for child (not water, not occupied)
          const neighborSpots = adjacentPositions.filter(
            p =>
              newGrid[p.y][p.x].agentId === undefined &&
              newGrid[p.y][p.x].terrain !== "water" &&
              !destinationMap.has(`${p.x},${p.y}`)
          );

          if (neighborSpots.length > 0) {
            const spot = neighborSpots[randomInt(neighborSpots.length)];

            // Both parents contribute energy (1/3 of total split between them)
            const totalEnergy = parentAgent.energy + mate.energy;
            const childEnergy = Math.floor(totalEnergy / 3);
            const parentCost = Math.floor(childEnergy / 2);
            const mateCost = childEnergy - parentCost;

            parentAgent = { ...parentAgent, energy: parentAgent.energy - parentCost };

            // Update mate's energy in the arrays
            const mateInUpdated = updatedAgents.findIndex(a => a.id === mate!.id);
            if (mateInUpdated !== -1) {
              updatedAgents[mateInUpdated] = { 
                ...updatedAgents[mateInUpdated], 
                energy: updatedAgents[mateInUpdated].energy - mateCost 
              };
            }

            // Mark both parents as having reproduced this tick
            usedForRepro.add(parentAgent.id);
            usedForRepro.add(mate.id);

            // Combine genes from both parents
            const childGenes = combineGenes(parentAgent.genes, mate.genes);

            // Child inherits qTable from first parent for genetic memory
            const child: Agent = {
              id: nextId++,
              x: spot.x,
              y: spot.y,
              energy: childEnergy,
              sex: Math.random() < 0.5 ? "M" : "F",
              ageTicks: 0,
              genes: childGenes,
              memory: { qTable: { ...parentAgent.memory.qTable } },
              lastRule: 'Born (sexual reproduction)'
            };

            newGrid[spot.y][spot.x].agentId = child.id;
            destinationMap.set(`${spot.x},${spot.y}`, child.id);
            updatedAgents.push(child);

            reward += CONFIG.simulation.reproductionReward;
            reproduced = true;

            logs.push(
              `Agent ${parentAgent.id}(${parentAgent.sex}) + ${mate.id}(${mate.sex}) → child ${child.id}(${child.sex}) at (${spot.x},${spot.y}), traitId ${child.genes.traitId}, energy ${childEnergy}`
            );
          }
        }
      }
    }

    // Determine if agent will die after this step (energy or old age)
    const isTooOld = ageYears > CONFIG.time.maxAgeYears;
    const isDeadAfterStep = parentAgent.energy <= 0 || isTooOld;
    const deathReason = isTooOld ? "old age" : "low energy";

    // Apply death penalty BEFORE Q-update so the agent learns from death
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

    parentAgent = { ...parentAgent, memory: { qTable: newQTable } };

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
    const gridWithFood = placeRandomFood(finalGrid, CONFIG.simulation.foodSpawnCount);
    return { agents: updatedAgents, grid: gridWithFood, log: logs };
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
 * Creates a fresh world state with terrain, food and agents to avoid duplication
 */
function initializeWorld(): { grid: Cell[][]; agents: Agent[] } {
  const empty = createEmptyGrid();
  const withTerrain = addRandomTerrain(empty);
  const withFood = placeRandomFood(withTerrain, INITIAL_FOOD);
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
    const newAgents = createInitialAgents(withFood, settings.initialAgents);
    setGrid(withFood);
    setAgents(newAgents);
    setLog([`New world created with ${settings.initialAgents} agents and ${settings.initialFood} food`]);
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
          memory: { qTable: {} },
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
          memory: { qTable: {} },
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
          memory: { qTable: {} },
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
              gridTemplateColumns: `repeat(${GRID_WIDTH}, 24px)`,
              gridTemplateRows: `repeat(${GRID_HEIGHT}, 24px)`,
              gap: "3px",
              border: "1px solid #334",
              padding: "8px",
              background: "linear-gradient(180deg, #151a30 0%, #0f1422 100%)",
              borderRadius: 12,
              boxShadow: "0 4px 20px rgba(0,0,0,0.3)"
            }}
          >
          {renderedGrid.map((row, y) =>
            row.map((cell, x) => {
              // PERFORMANCE IMPROVEMENT: Use agentMap for O(1) lookup instead of O(n) find
              const agent = cell.agentId ? agentMap.get(cell.agentId) : undefined;
              const isSelected = selectedAgentId === agent?.id;
              const isWatched =
                agent && watchedTraitId !== null && agent.genes.traitId === watchedTraitId;

              // Determine background color based on terrain, food, and agent
              let bg = "#1f2640"; // plain terrain default
              if (cell.terrain === "water") bg = "#1e3a5f"; // water - blue
              else if (cell.terrain === "rich") bg = "#3d2e1f"; // rich - brownish
              
              if (cell.food) bg = cell.terrain === "rich" ? "#4caf50" : "#2c9c3f"; // brighter green on rich
              if (agent) bg = colorForTrait(agent.genes.traitId);
              if (cell.food && agent) bg = "#ffb300";              return (
                <div
                  key={`${x}-${y}`}
                  onClick={() => agent && setSelectedAgentId(agent.id)}
                  style={{
                    width: "24px",
                    height: "24px",
                    borderRadius: "5px",
                    background: bg,
                    border: isSelected
                      ? "2px solid #ffeb3b"
                      : isWatched
                      ? "2px solid #ff5252"
                      : "1px solid rgba(255,255,255,0.1)",
                    boxSizing: "border-box",
                    cursor: agent ? "pointer" : "default",
                    transition: "transform 0.1s ease",
                    boxShadow: agent ? "0 2px 8px rgba(0,0,0,0.3)" : "none"
                  }}
                  title={
                    agent
                      ? `Agent ${agent.id} – energy ${agent.energy}, traitId ${agent.genes.traitId}`
                      : cell.food
                      ? "Food"
                      : ""
                  }                />
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
            padding: "10px",
            background: "#151a30",
            borderRadius: "8px",
            border: "1px solid #333"
          }}
        >
          <h3>Statistics</h3>
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

        {/* Simulation Settings */}
        <div
          style={{
            marginBottom: "12px",
            padding: "10px",
            background: "#151a30",
            borderRadius: "8px",
            border: "1px solid #333"
          }}
        >
          <h3>Simulation Settings</h3>
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
            style={{ marginTop: 10, width: "100%" }}
          >
            Create New World with Settings
          </button>
        </div>        {/* Agent inspector */}
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
                <strong>ID:</strong> {selectedAgent.id} | <strong>Sex:</strong> {selectedAgent.sex}
              </p>
              <p>
                <strong>Position:</strong> ({selectedAgent.x},{selectedAgent.y})
              </p>              <p>
                <strong>Energy:</strong> {selectedAgent.energy}
              </p>
              <p>
                <strong>Age:</strong> {selectedAgent.ageTicks} ticks (~
                {(selectedAgent.ageTicks * CONFIG.time.yearsPerTick).toFixed(1)} years)
              </p>
              <p>
                <strong>Last Rule:</strong>{" "}
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
                <strong>Personality:</strong> riskTolerance={selectedAgent.genes.riskTolerance.toFixed(2)},{" "}
                socialDrive={selectedAgent.genes.socialDrive.toFixed(2)}
                <br />
                traitId={selectedAgent.genes.traitId}
              </p>
            </>          ) : (
            <p>Click an agent in the grid to inspect its genetic memory.</p>
          )}
        </div>        {/* Rules + RL + evolution description */}
        <div
          style={{
            marginBottom: "12px",
            padding: "10px",
            background: "#151a30",
            borderRadius: "8px",
            border: "1px solid #333"
          }}
        >
          <h3>Behavior, Reinforcement Learning & Evolution</h3>
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
              <strong>Reproduction:</strong> if energy &gt; reproductionThreshold,
              agent may split energy with a child, whose genes are mutated.
            </li>
          </ul>
        </div>        {/* Charts */}
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
        </div>        <div
          style={{
            marginBottom: "12px",
            padding: "10px",
            background: "#151a30",
            borderRadius: "8px",
            border: "1px solid #333"
          }}
        >
          <h3>
            Trait Distribution
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
        </div>        {/* Log */}
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
          <h3>Action Log</h3>
          {log.length === 0 ? (
            <p>No steps yet. Press "Step" or "Play" to advance the world.</p>
          ) : (
            <ul style={{ paddingLeft: "18px" }}>              {log.map((entry, i) => (
                <li key={i} style={{ fontSize: "0.8em" }}>
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