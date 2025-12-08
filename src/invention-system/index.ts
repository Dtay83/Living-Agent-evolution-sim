/**
 * Limitless Invention System for the Living Agent Evolution Simulation
 * Provides procedural generation of unlimited inventions based on agent capabilities
 */

import type { Agent, Invention, InventionEffect } from '../types';
import { CONFIG } from '../core/config';

/**
 * Invention name components for procedural generation
 */
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
 * @param agent - Agent discovering the invention
 * @param tick - Current simulation tick
 * @param inventionNumber - Number of inventions this agent has
 * @returns Procedurally generated invention
 */
export function generateInvention(agent: Agent, tick: number, inventionNumber: number): Invention {
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
 * Check if an agent discovers a new invention this tick
 * Discovery is based on:
 * - Having enough "cognitive surplus" (energy)
 * - Curiosity and creativity genes
 * - Accumulated invention points (inspiration)
 * - Random chance
 * 
 * @param agent - Agent potentially discovering
 * @param tick - Current simulation tick
 * @returns Object with invention (if discovered) and updated agent with new points
 */
export function checkForDiscovery(
  agent: Agent, 
  tick: number
): { invention: Invention | null; updatedAgent: Agent } {
  // Agents gain invention points based on curiosity and exploration
  // Points represent accumulated knowledge, experience, and inspiration
  const pointGain = agent.genes.curiosity * agent.genes.exploration * CONFIG.invention.pointGainMultiplier;
  let newPoints = agent.inventionPoints + pointGain;
  
  // Only agents with enough energy can invent (cognitive surplus)
  if (agent.energy < CONFIG.invention.discoveryEnergyCost) {
    return { 
      invention: null, 
      updatedAgent: { ...agent, inventionPoints: newPoints }
    };
  }
  
  // Discovery chance increases with curiosity and creativity
  const creativityBoost = agent.genes.creativity;
  const discoveryChance = agent.genes.curiosity * CONFIG.invention.baseDiscoveryChance * (1 + creativityBoost);
  
  // Also consider accumulated invention points as inspiration
  const inspirationBonus = Math.min(
    CONFIG.invention.maxInspirationBonus, 
    newPoints * CONFIG.invention.inspirationBonus
  );
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
 * Calculate the actual energy cost for movement based on inventions
 * @param agent - Agent to calculate cost for
 * @returns Modified energy cost (minimum 0.5)
 */
export function getMovementCost(agent: Agent): number {
  let cost = CONFIG.simulation.baseEnergyCost;
  
  // Check for energy efficiency inventions
  for (const inv of agent.inventions) {
    if (inv.effect.type === 'energy_efficiency') {
      cost *= inv.effect.multiplier;
    }
  }
  
  return Math.max(0.5, cost); // Minimum cost of 0.5
}

/**
 * Get the food detection range for an agent
 * FUTURE: This helper is ready but not yet integrated into decision-making logic
 * @param agent - Agent to check
 * @returns Maximum food detection range
 */
export function getFoodDetectionRange(agent: Agent): number {
  let range = 1; // Base range (adjacent cells)
  
  for (const inv of agent.inventions) {
    if (inv.effect.type === 'food_detection_range') {
      range = Math.max(range, inv.effect.range);
    }
  }
  
  return range;
}

/**
 * Get reproduction bonus from inventions
 * @param agent - Agent to check
 * @returns Total reproduction energy bonus
 */
export function getReproductionBonus(agent: Agent): number {
  let bonus = 0;
  
  for (const inv of agent.inventions) {
    if (inv.effect.type === 'reproduction_boost') {
      bonus += inv.effect.bonus;
    }
  }
  
  return bonus;
}
