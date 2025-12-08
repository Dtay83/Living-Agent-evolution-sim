/**
 * Limitless Invention System for the Living Agent Evolution Simulation
 * Provides procedural generation of unlimited inventions based on agent capabilities
 * 
 * NEW REQUIREMENT: Inventions can supersede modern physics, quantum mechanics, and quantum physics
 * when supported by documentation and evidence. This creates breakthrough discoveries that
 * go beyond conventional scientific understanding.
 */

import type { Agent, Invention, InventionEffect } from '../types';
import { CONFIG } from '../core/config';

/**
 * Invention name components for procedural generation
 */
const INVENTION_PREFIXES = [
  'Efficient', 'Advanced', 'Enhanced', 'Optimized', 'Swift', 'Powerful',
  'Refined', 'Masterful', 'Superior', 'Elite', 'Expert', 'Precise',
  'Strategic', 'Tactical', 'Innovative', 'Revolutionary', 'Adaptive', 'Dynamic',
  // NEW: Beyond-physics breakthrough prefixes
  'Quantum-Breaking', 'Post-Relativistic', 'Hyperdimensional', 'Meta-Physical',
  'Trans-Quantum', 'Neo-Classical', 'Ultra-Modern', 'Paradigm-Shifting'
];

const INVENTION_THEMES = [
  'Foraging', 'Hunting', 'Gathering', 'Navigation', 'Communication',
  'Defense', 'Offense', 'Survival', 'Cooperation', 'Efficiency',
  'Awareness', 'Adaptation', 'Endurance', 'Speed', 'Strength',
  'Intelligence', 'Memory', 'Reflexes', 'Instinct', 'Wisdom',
  // NEW: Beyond-physics breakthrough themes
  'Entanglement', 'Superposition', 'Tunneling', 'Unification',
  'Emergence', 'Transcendence', 'Singularity', 'Manifestation'
];

const INVENTION_TYPES: Array<'tool' | 'technique' | 'structure'> = [
  'tool', 'technique', 'structure'
];

/**
 * Generate a unique invention based on agent's capabilities and random inspiration
 * 
 * BREAKTHROUGH CAPABILITY: Agents with high invention points and creativity can discover
 * inventions that supersede conventional physics through emergent phenomena.
 * These "beyond-physics" inventions are documented as breakthroughs.
 * 
 * @param agent - Agent discovering the invention
 * @param tick - Current simulation tick
 * @param inventionNumber - Number of inventions this agent has
 * @param scienceLevel - Optional science level for breakthrough determination
 * @returns Procedurally generated invention
 */
export function generateInvention(
  agent: Agent, 
  tick: number, 
  inventionNumber: number,
  scienceLevel: number = 0
): Invention {
  // Use agent's creativity to influence invention quality
  const creativityFactor = agent.genes.curiosity * agent.genes.creativity;
  const inspirationFactor = Math.min(1, agent.inventionPoints / 100); // Normalize points
  
  // BREAKTHROUGH CHECK: High creativity + inspiration + advanced science can supersede physics
  // Documentation: When agents accumulate sufficient knowledge (invention points > 50),
  // combined with high creativity (> 0.7) and advanced science (level > 5),
  // they can discover phenomena that go beyond conventional physics limits.
  const breakthroughChance = creativityFactor * inspirationFactor * (scienceLevel / 10);
  const isBreakthrough = breakthroughChance > 0.5 && Math.random() < breakthroughChance * 0.1;
  
  // Random invention type
  const type = INVENTION_TYPES[Math.floor(Math.random() * INVENTION_TYPES.length)];
  
  // Generate unique name based on invention number and random elements
  const prefixPool = isBreakthrough && Math.random() < 0.3
    ? INVENTION_PREFIXES.slice(-8) // Use breakthrough prefixes for 30% of breakthroughs
    : INVENTION_PREFIXES;
  const themePool = isBreakthrough && Math.random() < 0.3
    ? INVENTION_THEMES.slice(-8) // Use breakthrough themes for 30% of breakthroughs
    : INVENTION_THEMES;
    
  const prefix = prefixPool[Math.floor(Math.random() * prefixPool.length)];
  const theme = themePool[Math.floor(Math.random() * themePool.length)];
  const name = `${prefix} ${theme}`;
  const id = `invention_${agent.id}_${inventionNumber}_${tick}`;
  
  // Determine effect type based on creativity and randomness
  const effectRoll = Math.random();
  let effect: InventionEffect;
  let description: string;
  
  // Breakthrough inventions can exceed normal limits (documented as emergent phenomena)
  const breakthroughMultiplier = isBreakthrough ? 1.5 : 1.0;
  
  if (effectRoll < 0.4) {
    // Energy efficiency - scales with creativity (ACTIVE EFFECT)
    // Breakthrough documentation: Emergent energy optimization can exceed thermodynamic limits
    // through quantum tunneling effects and zero-point energy exploitation
    const baseMultiplier = 0.95 - (creativityFactor * 0.25); // 0.7 to 0.95 range
    const multiplier = isBreakthrough 
      ? Math.max(0.3, baseMultiplier * 0.6) // Breakthroughs: 0.3 to 0.7 (30-70% efficiency!)
      : Math.max(0.5, baseMultiplier);
    effect = { type: 'energy_efficiency', multiplier };
    description = isBreakthrough
      ? `⚡ BREAKTHROUGH: Reduces energy cost by ${Math.round((1 - multiplier) * 100)}% through post-quantum optimization!`
      : `Reduces energy cost by ${Math.round((1 - multiplier) * 100)}%`;
  } else if (effectRoll < 0.7) {
    // Reproduction boost - scales with creativity (ACTIVE EFFECT)
    // Breakthrough documentation: Advanced bio-engineering and resource optimization
    // can dramatically improve reproductive success beyond natural limits
    const baseBonus = Math.ceil(2 + creativityFactor * 8); // 2 to 10 bonus
    const bonus = isBreakthrough 
      ? Math.ceil(baseBonus * breakthroughMultiplier) // Breakthroughs: up to 15 bonus!
      : baseBonus;
    effect = { type: 'reproduction_boost', bonus };
    description = isBreakthrough
      ? `⚡ BREAKTHROUGH: +${bonus} bonus energy for reproduction via bio-optimization!`
      : `+${bonus} bonus energy for reproduction`;
  } else if (effectRoll < 0.85) {
    // Food detection range - scales with exploration
    // Breakthrough documentation: Quantum entanglement sensing and hyperdimensional
    // awareness can detect resources far beyond normal perception
    const baseRange = Math.ceil(1 + agent.genes.exploration * 3); // 1 to 4 range
    const range = isBreakthrough 
      ? Math.ceil(baseRange * breakthroughMultiplier) // Breakthroughs: up to 6 cells!
      : baseRange;
    effect = { type: 'food_detection_range', range };
    description = isBreakthrough
      ? `⚡ BREAKTHROUGH: Detect food ${range} cells away via quantum sensing!`
      : `Detect food ${range} cells away`;
  } else if (effectRoll < 0.95) {
    // Storage capacity - scales with patience
    // Breakthrough documentation: Hyperdimensional storage and matter compression
    // techniques can store far more than conventional limits
    const baseCapacity = Math.ceil(5 + agent.genes.patience * 20); // 5 to 25 capacity
    const capacity = isBreakthrough 
      ? Math.ceil(baseCapacity * breakthroughMultiplier) // Breakthroughs: up to 37 capacity!
      : baseCapacity;
    effect = { type: 'storage', capacity };
    description = isBreakthrough
      ? `⚡ BREAKTHROUGH: Store ${capacity} extra energy in hyperdimensional cache!`
      : `Store ${capacity} extra energy`;
  } else {
    // Defense - scales with both genes
    // Breakthrough documentation: Probability manipulation and quantum superposition
    // can provide protection beyond classical physics limits
    const baseProtection = Math.min(0.5, 0.1 + creativityFactor * 0.4); // 0.1 to 0.5
    const protection = isBreakthrough 
      ? Math.min(0.75, baseProtection * breakthroughMultiplier) // Breakthroughs: up to 75% protection!
      : baseProtection;
    effect = { type: 'defense', protection };
    description = isBreakthrough
      ? `⚡ BREAKTHROUGH: ${Math.round(protection * 100)}% protection via quantum shielding!`
      : `${Math.round(protection * 100)}% chance to avoid energy loss`;
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
 * @param scienceLevel - Current civilization science level (for breakthrough inventions)
 * @returns Object with invention (if discovered) and updated agent with new points
 */
export function checkForDiscovery(
  agent: Agent, 
  tick: number,
  scienceLevel: number = 0
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
  
  // Discovery! Generate a unique invention (with science level for breakthrough potential)
  const inventionNumber = agent.inventions.length + 1;
  const invention = generateInvention(agent, tick, inventionNumber, scienceLevel);
  
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
