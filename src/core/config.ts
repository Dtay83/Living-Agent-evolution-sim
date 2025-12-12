/**
 * Configuration constants for the Living Agent Evolution Simulation
 * All magic numbers are organized here for better maintainability
 */

export const CONFIG = {
  grid: {
    initialWidth: 25,
    initialHeight: 25,
    maxWidth: 27,         // Maximum grid width before performance issues
    maxHeight: 30,        // Maximum grid height
    expandBy: 2,          // Number of cells to add when expanding
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
  },  genes: {
    foodPreference: { min: 0.6, max: 1.0 },
    exploration: { min: 0.3, max: 0.8 },
    reproductionThreshold: { min: 15, max: 23 },
    mutationRate: { min: 0.08, max: 0.25 },  // Increased for more genetic diversity
    curiosity: { min: 0.7, max: 1.6 },       // UNCAPPED: Can evolve beyond 1.0 for transcendent curiosity
    social: { min: 0.3, max: 0.9 },
    creativity: { min: 0.25, max: 0.8 },     // Boosted minimum for baseline creativity
    patience: { min: 0.2, max: 0.8 },
    mutation: {
      foodPreferenceMagnitude: 0.15,
      explorationMagnitude: 0.2,
      reproductionThresholdMagnitude: 3,
      mutationRateMagnitude: 0.08,           // Increased mutation magnitude
      curiosityMagnitude: 0.30,              // DOUBLED: 2x mutation magnitude for curiosity
      socialMagnitude: 0.15,
      creativityMagnitude: 0.18,             // Slightly higher creativity mutations
      patienceMagnitude: 0.15,
      newTraitChance: 0.12,                  // Higher chance for new trait lineages
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
    // Environmental Challenges - TUNED for faster evolution through selective pressure
  challenges: {
    enabled: true,
    
    // Periodic food scarcity events
    famine: {
      enabled: true,
      frequency: 120,              // More frequent (was 150)
      chance: 0.35,                // Higher chance (was 0.3)
      duration: 18,                // Slightly shorter (was 20)
      foodReduction: 0.4,          // Harsher reduction (was 0.5)
    },
    
    // Energy drain events (harsh conditions)
    harshWeather: {
      enabled: true,
      frequency: 160,              // More frequent (was 200)
      chance: 0.30,                // Higher chance (was 0.25)
      duration: 12,                // Shorter bursts (was 15)
      extraEnergyCost: 0.6,        // Slightly harsher (was 0.5)
    },
    
    // Population pressure - increases competition
    overcrowding: {
      enabled: true,
      threshold: 22,               // Lower threshold (was 25)
      energyPenalty: 0.35,         // Slightly higher penalty (was 0.3)
    },
    
    // Bonus events - rewards adapted agents
    abundance: {
      enabled: true,
      frequency: 200,              // More frequent rewards (was 250)
      chance: 0.2,                 // 20% chance
      duration: 30,                // Lasts N ticks
      foodBonus: 2.0,              // Food spawn rate multiplied by this
    },
  },
} as const;

// Backwards compatibility aliases (now dynamic - will be set by WorldState)
export let GRID_WIDTH: number = CONFIG.grid.initialWidth;
export let GRID_HEIGHT: number = CONFIG.grid.initialHeight;

/**
 * Update grid dimensions (used when grid expands)
 * @param width - New grid width
 * @param height - New grid height
 */
export function setGridDimensions(width: number, height: number) {
  GRID_WIDTH = width;
  GRID_HEIGHT = height;
}
export const INITIAL_AGENTS = CONFIG.simulation.initialAgents;
export const INITIAL_FOOD = CONFIG.simulation.initialFood;
export const ALPHA = CONFIG.rl.alpha;
export const GAMMA = CONFIG.rl.gamma;
export const EPSILON = CONFIG.rl.epsilon;
