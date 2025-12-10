/**
 * PHYSICS INTEGRATION SYSTEM
 * 
 * Integrates physics concepts with agent actions:
 * - Movement costs based on "mass" (energy/inventions)
 * - Physics bonuses from discovered concepts
 * - Collaborative physics discovery when agents are nearby
 * - Applied physics inventions
 */

import type { Agent } from '../types';
import type { PhysicsConcept } from '../science-system/physics';
import { PHYSICS_CONCEPTS, calculatePhysicsBonuses, generateAdvancedPhysicsConcept, getAllPhysicsConcepts } from '../science-system/physics';

/**
 * Physics state tracked per agent
 */
export interface AgentPhysicsState {
  mass: number;                    // Calculated from energy + inventions
  friction: number;                // Movement resistance (0-1)
  discoveredConcepts: string[];    // IDs of concepts this agent discovered
  appliedPhysics: string[];        // IDs of concepts actively applied
}

/**
 * Global physics state for the civilization
 */
export interface CivilizationPhysics {
  unlockedConcepts: PhysicsConcept[];
  totalDiscoveries: number;
  lastDiscoveryTick: number;
  collaborationBonuses: Map<string, number>; // agentPair -> bonus
}

/**
 * Initialize civilization physics state
 */
export function initializeCivilizationPhysics(): CivilizationPhysics {
  return {
    unlockedConcepts: [],
    totalDiscoveries: 0,
    lastDiscoveryTick: 0,
    collaborationBonuses: new Map()
  };
}

/**
 * Calculate an agent's effective "mass" for physics calculations
 * Higher mass = more energy cost but more momentum
 */
export function calculateAgentMass(agent: Agent): number {
  // Base mass from energy (heavier when well-fed)
  const energyMass = Math.sqrt(agent.energy) * 0.5;
  
  // Inventions add mass (carrying tools/knowledge)
  const inventionMass = agent.inventions.length * 0.2;
  
  // Total mass with minimum of 1.0
  return Math.max(1.0, energyMass + inventionMass);
}

/**
 * Calculate friction based on agent's movement style
 * Exploration gene reduces friction (more agile)
 */
export function calculateFriction(agent: Agent): number {
  // Base friction
  let friction = 0.5;
  
  // Exploration gene reduces friction
  friction -= agent.genes.exploration * 0.2;
  
  // Patience increases friction (more deliberate movement)
  friction += agent.genes.patience * 0.1;
  
  return Math.max(0.1, Math.min(0.9, friction));
}

/**
 * Apply physics-based movement cost
 * Integrates with existing getMovementCost but adds physics bonuses
 */
export function getPhysicsMovementCost(
  agent: Agent,
  baseMovementCost: number,
  unlockedPhysics: PhysicsConcept[]
): number {
  // Get physics bonuses from civilization knowledge
  const bonuses = calculatePhysicsBonuses(unlockedPhysics);
  
  // Calculate mass and friction
  const mass = calculateAgentMass(agent);
  const friction = calculateFriction(agent);
  
  // Physics formula: cost = base * mass * friction * energyEfficiency
  let cost = baseMovementCost * (mass / 3) * (1 + friction * 0.5);
  
  // Apply energy efficiency bonus from physics discoveries
  cost *= bonuses.energyEfficiency;
  
  // Minimum cost
  return Math.max(0.1, cost);
}

/**
 * Check if agents can collaborate on physics discovery
 * Nearby agents with high social genes boost discovery chance
 */
export function getCollaborationBonus(
  agent: Agent,
  nearbyAgents: Agent[],
  unlockedPhysics: PhysicsConcept[]
): number {
  if (nearbyAgents.length === 0) return 0;
  
  // Sum social genes of nearby agents
  const socialSum = nearbyAgents.reduce((sum, a) => sum + a.genes.social, 0);
  
  // Average curiosity of the group
  const avgCuriosity = (agent.genes.curiosity + 
    nearbyAgents.reduce((sum, a) => sum + a.genes.curiosity, 0)) / (nearbyAgents.length + 1);
  
  // Knowledge synergy: more discovered concepts = higher collaboration value
  const knowledgeFactor = Math.sqrt(unlockedPhysics.length + 1) * 0.1;
  
  // Collaboration bonus: social * curiosity * knowledge * group size factor
  const groupSizeFactor = Math.min(1.5, 1 + Math.log2(nearbyAgents.length + 1) * 0.3);
  
  return socialSum * avgCuriosity * knowledgeFactor * groupSizeFactor;
}

/**
 * Check if an agent discovers a physics concept this tick
 * UNLIMITED EVOLUTION - procedurally generates new concepts when base set exhausted
 */
export function checkPhysicsDiscovery(
  agent: Agent,
  nearbyAgents: Agent[],
  unlockedPhysics: PhysicsConcept[],
  tick: number,
  generatedPhysicsLevel: number = 0
): { concept: PhysicsConcept | null; log: string | null; nextGeneratedLevel?: number } {
  // Need cognitive surplus (energy > 10) to think about physics
  if (agent.energy < 10) {
    return { concept: null, log: null };
  }
  
  // Find discoverable concepts (prerequisites met, not already unlocked)
  const unlockedIds = new Set(unlockedPhysics.map(p => p.id));
  let discoverableConcepts = PHYSICS_CONCEPTS.filter(concept => {
    if (unlockedIds.has(concept.id)) return false;
    return concept.prerequisiteIds.every(id => unlockedIds.has(id) || id === '');
  });
  
  // Include concepts with empty prerequisites for initial discoveries
  let availableConcepts = discoverableConcepts.length > 0 
    ? discoverableConcepts 
    : PHYSICS_CONCEPTS.filter(c => c.prerequisiteIds.length === 0 && !unlockedIds.has(c.id));
  
  // UNLIMITED EVOLUTION: If all base concepts discovered, generate procedural ones
  let nextGeneratedLevel = generatedPhysicsLevel;
  if (availableConcepts.length === 0) {
    // Check if last concept in chain is unlocked to allow next procedural
    const lastConceptId = generatedPhysicsLevel > 0 
      ? `advanced_physics_${generatedPhysicsLevel - 1}` 
      : 'omega_physics';
    
    if (unlockedIds.has(lastConceptId)) {
      // Generate next procedural concept
      const proceduralConcept = generateAdvancedPhysicsConcept(generatedPhysicsLevel);
      availableConcepts = [proceduralConcept];
      nextGeneratedLevel = generatedPhysicsLevel + 1;
    } else {
      return { concept: null, log: null };
    }
  }
  
  if (availableConcepts.length === 0) {
    return { concept: null, log: null };
  }
  
  // Calculate discovery chance
  const baseChance = agent.genes.curiosity * agent.genes.creativity * 0.02;
  const collaborationBonus = getCollaborationBonus(agent, nearbyAgents, unlockedPhysics);
  const inventionPointBonus = Math.min(0.03, agent.inventionPoints * 0.001);
  
  const totalChance = baseChance + collaborationBonus * 0.01 + inventionPointBonus;
  
  if (Math.random() > totalChance) {
    return { concept: null, log: null };
  }
  
  // Select concept to discover (prefer lower complexity)
  const sortedConcepts = [...availableConcepts].sort((a, b) => a.complexity - b.complexity);
  
  // Weight towards simpler concepts but allow lucky breakthroughs
  const indexWeight = Math.random() * Math.random(); // Bias towards lower indices
  const selectedIndex = Math.floor(indexWeight * sortedConcepts.length);
  const concept = sortedConcepts[selectedIndex];
  
  // Mark as discovered
  const discoveredConcept: PhysicsConcept = {
    ...concept,
    discoveredAt: tick,
    discoveredBy: agent.id
  };
    // Generate log message
  const collabMsg = nearbyAgents.length > 0 
    ? ` (collaborated with ${nearbyAgents.length} nearby agent${nearbyAgents.length > 1 ? 's' : ''}!)` 
    : '';
  const proceduralMsg = concept.id.startsWith('advanced_physics_') ? ' 🌟 TRANSCENDENT DISCOVERY!' : '';
  const log = `🔬 Agent ${agent.id} discovered physics concept: ${concept.name}!${collabMsg}${proceduralMsg}`;
  
  return { concept: discoveredConcept, log, nextGeneratedLevel };
}

/**
 * Get physics bonuses that affect agent actions
 * Returns multipliers/bonuses to apply to various agent capabilities
 * NO CAPS - bonuses grow infinitely with discoveries
 */
export function getAgentPhysicsBonuses(unlockedPhysics: PhysicsConcept[]): {
  energyEfficiency: number;
  movementSpeed: number;
  learningRate: number;
  inventionChance: number;
  spaceManipulation: number;
  timePerception: number;
} {
  return calculatePhysicsBonuses(unlockedPhysics);
}

/**
 * Apply physics-enhanced learning rate to Q-learning
 */
export function getPhysicsEnhancedLearningRate(
  baseLearningRate: number,
  unlockedPhysics: PhysicsConcept[]
): number {
  const bonuses = calculatePhysicsBonuses(unlockedPhysics);
  return baseLearningRate * bonuses.learningRate;
}

/**
 * Get physics-enhanced invention discovery chance
 */
export function getPhysicsInventionBonus(unlockedPhysics: PhysicsConcept[]): number {
  const bonuses = calculatePhysicsBonuses(unlockedPhysics);
  return bonuses.inventionChance;
}

/**
 * Summarize physics state for UI display
 * Shows all bonuses including new space manipulation and time perception
 */
export function getPhysicsSummary(unlockedPhysics: PhysicsConcept[]): {
  totalConcepts: number;
  byCategory: Record<string, number>;
  topBonuses: { name: string; value: string }[];
  proceduralLevel: number;
} {
  const byCategory: Record<string, number> = {};
  let proceduralLevel = 0;
  
  for (const concept of unlockedPhysics) {
    byCategory[concept.category] = (byCategory[concept.category] || 0) + 1;
    // Track procedural concepts
    if (concept.id.startsWith('advanced_physics_')) {
      const level = parseInt(concept.id.replace('advanced_physics_', ''));
      proceduralLevel = Math.max(proceduralLevel, level + 1);
    }
  }
  
  const bonuses = calculatePhysicsBonuses(unlockedPhysics);
  const topBonuses: { name: string; value: string }[] = [];
  
  if (bonuses.energyEfficiency < 1) {
    topBonuses.push({ 
      name: 'Energy Efficiency', 
      value: `${Math.round((1 - bonuses.energyEfficiency) * 100)}% reduced cost` 
    });
  }
  if (bonuses.movementSpeed > 1) {
    topBonuses.push({ 
      name: 'Movement Speed', 
      value: `${Math.round((bonuses.movementSpeed - 1) * 100)}% faster` 
    });
  }
  if (bonuses.learningRate > 1) {
    topBonuses.push({ 
      name: 'Learning Rate', 
      value: `${Math.round((bonuses.learningRate - 1) * 100)}% bonus` 
    });
  }
  if (bonuses.inventionChance > 0) {
    topBonuses.push({ 
      name: 'Invention Chance', 
      value: `+${(bonuses.inventionChance * 100).toFixed(1)}%` 
    });
  }
  // NEW BONUSES - NO CAPS
  if (bonuses.spaceManipulation > 1) {
    topBonuses.push({ 
      name: 'Space Manipulation', 
      value: `${Math.round((bonuses.spaceManipulation - 1) * 100)}% power` 
    });
  }
  if (bonuses.timePerception > 1) {
    topBonuses.push({ 
      name: 'Time Perception', 
      value: `${Math.round((bonuses.timePerception - 1) * 100)}% enhanced` 
    });
  }
  
  return {
    totalConcepts: unlockedPhysics.length,
    byCategory,
    topBonuses,
    proceduralLevel
  };
}
