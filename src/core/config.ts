/**
 * Configuration constants for the Living Agent Evolution Simulation
 * All magic numbers are organized here for better maintainability
 */

export const CONFIG = {
  grid: {
    initialWidth: 16,
    initialHeight: 10,
    maxWidth: 50,         // Maximum grid width before performance issues
    maxHeight: 30,        // Maximum grid height
    expandBy: 2,          // Number of cells to add when expanding
  },  simulation: {
    initialAgents: 6,
    initialFood: 24,
    // Food is spawned every tick; the amount scales with population so the
    // ecosystem can actually sustain the agents it produces.
    foodSpawnChance: 1.0,
    foodSpawnCount: 2,
    foodPerAgent: 0.35,        // extra food per living agent, per tick
    maxFoodOnGrid: 0.35,       // cap food at 35% of the cells
    minPopulation: 4,          // reseed floor so the world never flatlines
    baseEnergyCost: 0.5,       // metabolism is cheaper -> longer life course
    foodEnergyBonus: 8,
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
  invention: {
    discoveryEnergyCost: 15,      // Minimum energy needed to discover
    baseDiscoveryChance: 0.03,     // Base discovery probability
    pointGainMultiplier: 0.5,      // Multiplier for point accumulation
    inspirationBonus: 0.001,       // Bonus per accumulated point
    maxInspirationBonus: 0.02,     // Cap on inspiration bonus
    inheritanceRate: 0.8,          // Base rate for invention inheritance
    pointInheritanceRate: 0.3,     // Rate for inheriting invention points
  },
} as const;

// Backwards compatibility aliases (now dynamic - will be set by WorldState)
export let GRID_WIDTH = CONFIG.grid.initialWidth;
export let GRID_HEIGHT = CONFIG.grid.initialHeight;

/**
 * Update grid dimensions (used when grid expands)
 * @param width - New grid width
 * @param height - New grid height
 */
export function setGridDimensions(width: number, height: number) {
  GRID_WIDTH = width;
  GRID_HEIGHT = height;
}
export const INITIAL_AGENTS: number = CONFIG.simulation.initialAgents;
export const INITIAL_FOOD: number = CONFIG.simulation.initialFood;
export const ALPHA: number = CONFIG.rl.alpha;
export const GAMMA: number = CONFIG.rl.gamma;
export const EPSILON: number = CONFIG.rl.epsilon;
