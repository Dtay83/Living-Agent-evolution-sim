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
    mutationRate: { min: 0.08, max: 0.25 },  // Increased for more genetic diversity
    curiosity: { min: 0.2, max: 0.8 },
    social: { min: 0.3, max: 0.9 },
    creativity: { min: 0.25, max: 0.8 },     // Boosted minimum for baseline creativity
    patience: { min: 0.2, max: 0.8 },
    mutation: {
      foodPreferenceMagnitude: 0.15,
      explorationMagnitude: 0.2,
      reproductionThresholdMagnitude: 3,
      mutationRateMagnitude: 0.08,           // Increased mutation magnitude
      curiosityMagnitude: 0.15,
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
  
  // Environmental Challenges - accelerates evolution through selective pressure
  challenges: {
    enabled: true,
    
    // Periodic food scarcity events
    famine: {
      enabled: true,
      frequency: 150,              // Every N ticks, a famine may occur
      chance: 0.3,                 // 30% chance when frequency hits
      duration: 20,                // Lasts N ticks
      foodReduction: 0.5,          // Food spawn rate multiplied by this
    },
    
    // Energy drain events (harsh conditions)
    harshWeather: {
      enabled: true,
      frequency: 200,              // Every N ticks
      chance: 0.25,                // 25% chance
      duration: 15,                // Lasts N ticks
      extraEnergyCost: 0.5,        // Additional energy cost per tick
    },
    
    // Population pressure - increases competition
    overcrowding: {
      enabled: true,
      threshold: 25,               // When population exceeds this
      energyPenalty: 0.3,          // Extra energy cost per tick when overcrowded
    },
    
    // Bonus events - rewards adapted agents
    abundance: {
      enabled: true,
      frequency: 250,              // Every N ticks
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
