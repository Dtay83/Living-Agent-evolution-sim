/**
 * ENVIRONMENTAL CHALLENGES SYSTEM
 * 
 * Creates selective pressure to accelerate evolution through:
 * - Periodic famines (food scarcity)
 * - Harsh weather (increased energy costs)
 * - Overcrowding penalties
 * - Abundance events (rewards)
 * 
 * Challenges force agents to adapt or die, accelerating the evolution
 * of beneficial traits like efficiency, creativity, and social cooperation.
 */

import { CONFIG } from './config';

// Challenge state tracking
export interface ChallengeState {
  // Active challenges
  activeFamine: boolean;
  famineDuration: number;
  activeHarshWeather: boolean;
  harshWeatherDuration: number;
  activeAbundance: boolean;
  abundanceDuration: number;
  isOvercrowded: boolean;
  
  // Statistics
  totalFamines: number;
  totalHarshWeatherEvents: number;
  totalAbundanceEvents: number;
  
  // Current modifiers (computed each tick)
  foodSpawnModifier: number;
  energyCostModifier: number;
}

/**
 * Initialize challenge state
 */
export function initializeChallengeState(): ChallengeState {
  return {
    activeFamine: false,
    famineDuration: 0,
    activeHarshWeather: false,
    harshWeatherDuration: 0,
    activeAbundance: false,
    abundanceDuration: 0,
    isOvercrowded: false,
    totalFamines: 0,
    totalHarshWeatherEvents: 0,
    totalAbundanceEvents: 0,
    foodSpawnModifier: 1.0,
    energyCostModifier: 0,
  };
}

/**
 * Process challenges for the current tick
 * Returns updated challenge state and any log messages
 */
export function processChallenges(
  state: ChallengeState,
  tick: number,
  populationCount: number
): { state: ChallengeState; logs: string[] } {
  if (!CONFIG.challenges.enabled) {
    return { state, logs: [] };
  }

  const logs: string[] = [];
  let newState = { ...state };
  
  // Reset modifiers
  newState.foodSpawnModifier = 1.0;
  newState.energyCostModifier = 0;

  // ========================================
  // FAMINE EVENTS
  // ========================================
  if (CONFIG.challenges.famine.enabled) {
    if (newState.activeFamine) {
      // Famine in progress
      newState.famineDuration--;
      newState.foodSpawnModifier *= CONFIG.challenges.famine.foodReduction;
      
      if (newState.famineDuration <= 0) {
        newState.activeFamine = false;
        logs.push(`🌾 The famine has ended. Food supplies returning to normal.`);
      }
    } else if (tick > 0 && tick % CONFIG.challenges.famine.frequency === 0) {
      // Check for new famine
      if (Math.random() < CONFIG.challenges.famine.chance) {
        newState.activeFamine = true;
        newState.famineDuration = CONFIG.challenges.famine.duration;
        newState.totalFamines++;
        logs.push(`⚠️ FAMINE! Food scarcity for ${CONFIG.challenges.famine.duration} ticks. Only the efficient will survive.`);
      }
    }
  }

  // ========================================
  // HARSH WEATHER EVENTS
  // ========================================
  if (CONFIG.challenges.harshWeather.enabled) {
    if (newState.activeHarshWeather) {
      // Harsh weather in progress
      newState.harshWeatherDuration--;
      newState.energyCostModifier += CONFIG.challenges.harshWeather.extraEnergyCost;
      
      if (newState.harshWeatherDuration <= 0) {
        newState.activeHarshWeather = false;
        logs.push(`☀️ The harsh weather has passed. Conditions normalizing.`);
      }
    } else if (tick > 0 && tick % CONFIG.challenges.harshWeather.frequency === 0) {
      // Check for new harsh weather
      if (Math.random() < CONFIG.challenges.harshWeather.chance) {
        newState.activeHarshWeather = true;
        newState.harshWeatherDuration = CONFIG.challenges.harshWeather.duration;
        newState.totalHarshWeatherEvents++;
        logs.push(`🌪️ HARSH WEATHER! Energy costs increased for ${CONFIG.challenges.harshWeather.duration} ticks.`);
      }
    }
  }

  // ========================================
  // OVERCROWDING
  // ========================================
  if (CONFIG.challenges.overcrowding.enabled) {
    const wasOvercrowded = newState.isOvercrowded;
    newState.isOvercrowded = populationCount > CONFIG.challenges.overcrowding.threshold;
    
    if (newState.isOvercrowded) {
      newState.energyCostModifier += CONFIG.challenges.overcrowding.energyPenalty;
      
      if (!wasOvercrowded) {
        logs.push(`👥 OVERCROWDING! Population (${populationCount}) exceeds capacity. Competition intensifies.`);
      }
    } else if (wasOvercrowded) {
      logs.push(`📉 Population stabilized. Overcrowding pressure relieved.`);
    }
  }

  // ========================================
  // ABUNDANCE EVENTS
  // ========================================
  if (CONFIG.challenges.abundance.enabled) {
    if (newState.activeAbundance) {
      // Abundance in progress
      newState.abundanceDuration--;
      newState.foodSpawnModifier *= CONFIG.challenges.abundance.foodBonus;
      
      if (newState.abundanceDuration <= 0) {
        newState.activeAbundance = false;
        logs.push(`🍂 The abundance period has ended.`);
      }
    } else if (tick > 0 && tick % CONFIG.challenges.abundance.frequency === 0) {
      // Check for new abundance (only if not in famine)
      if (!newState.activeFamine && Math.random() < CONFIG.challenges.abundance.chance) {
        newState.activeAbundance = true;
        newState.abundanceDuration = CONFIG.challenges.abundance.duration;
        newState.totalAbundanceEvents++;
        logs.push(`🌸 ABUNDANCE! Food plentiful for ${CONFIG.challenges.abundance.duration} ticks. Time to grow!`);
      }
    }
  }

  return { state: newState, logs };
}

/**
 * Get the effective food spawn chance considering challenges
 */
export function getEffectiveFoodSpawnChance(state: ChallengeState): number {
  return CONFIG.simulation.foodSpawnChance * state.foodSpawnModifier;
}

/**
 * Get the effective food spawn count considering challenges
 */
export function getEffectiveFoodSpawnCount(state: ChallengeState): number {
  const baseCount = CONFIG.simulation.foodSpawnCount;
  if (state.activeAbundance) {
    return Math.ceil(baseCount * CONFIG.challenges.abundance.foodBonus);
  }
  if (state.activeFamine) {
    return Math.max(1, Math.floor(baseCount * CONFIG.challenges.famine.foodReduction));
  }
  return baseCount;
}

/**
 * Get additional energy cost from challenges
 */
export function getChallengeEnergyCost(state: ChallengeState): number {
  return state.energyCostModifier;
}

/**
 * Get a summary of active challenges for display
 */
export function getActiveChallengesSummary(state: ChallengeState): string[] {
  const active: string[] = [];
  
  if (state.activeFamine) {
    active.push(`🌾 Famine (${state.famineDuration} ticks)`);
  }
  if (state.activeHarshWeather) {
    active.push(`🌪️ Harsh Weather (${state.harshWeatherDuration} ticks)`);
  }
  if (state.isOvercrowded) {
    active.push(`👥 Overcrowded`);
  }
  if (state.activeAbundance) {
    active.push(`🌸 Abundance (${state.abundanceDuration} ticks)`);
  }
  
  return active;
}

/**
 * Check if any negative challenge is active
 */
export function hasNegativeChallenge(state: ChallengeState): boolean {
  return state.activeFamine || state.activeHarshWeather || state.isOvercrowded;
}

/**
 * Check if any positive event is active
 */
export function hasPositiveEvent(state: ChallengeState): boolean {
  return state.activeAbundance;
}
