/**
 * Configuration constants for the Living Agent Evolution Simulation
 * All magic numbers are organized here for better maintainability
 */

export const CONFIG = {
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

// Backwards compatibility aliases
export const GRID_WIDTH = CONFIG.grid.width;
export const GRID_HEIGHT = CONFIG.grid.height;
export const INITIAL_AGENTS = CONFIG.simulation.initialAgents;
export const INITIAL_FOOD = CONFIG.simulation.initialFood;
export const ALPHA = CONFIG.rl.alpha;
export const GAMMA = CONFIG.rl.gamma;
export const EPSILON = CONFIG.rl.epsilon;
