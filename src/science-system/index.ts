/**
 * Science System Coordinator
 * Manages physics, mathematics, and era progression with unlimited scaling
 */

import type { Agent } from '../types';
import type { ScientificEra, ScientificDiscovery, ProgressionMetrics } from './eras';
import type { PhysicsConcept } from './physics';
import type { MathConcept } from './mathematics';
import { BASE_ERAS, generateNextEra, shouldAdvanceEra } from './eras';
import { PHYSICS_CONCEPTS, generateAdvancedPhysicsConcept, calculatePhysicsBonuses } from './physics';
import { MATH_CONCEPTS, generateAdvancedMathConcept, calculateMathBonuses } from './mathematics';

export interface ScienceState {
  currentEra: ScientificEra;
  allEras: ScientificEra[];              // Complete history of eras
  unlockedPhysics: PhysicsConcept[];
  unlockedMath: MathConcept[];
  allDiscoveries: ScientificDiscovery[];
  progressionMetrics: ProgressionMetrics;
  civilizationKnowledge: {
    physics: Record<string, PhysicsConcept>;
    mathematics: Record<string, MathConcept>;
  };
}

/**
 * Initialize the science system
 */
export function initializeScienceSystem(tick: number): ScienceState {
  const initialEra: ScientificEra = {
    ...BASE_ERAS[0],
    startTick: tick,
    discoveries: [],
    physicsUnlocked: [],
    mathUnlocked: []
  };
  
  return {
    currentEra: initialEra,
    allEras: [initialEra],
    unlockedPhysics: [],
    unlockedMath: [],
    allDiscoveries: [],
    progressionMetrics: {
      totalDiscoveries: 0,
      discoveriesPerEra: { 0: 0 },
      averageDiscoveryRate: 0,
      currentEraLevel: 0,
      ticksSinceLastEra: 0,
      scientificAcceleration: 0
    },
    civilizationKnowledge: {
      physics: {},
      mathematics: {}
    }
  };
}

/**
 * Check if agent can discover a physics concept
 */
export function canDiscoverPhysics(
  agent: Agent,
  concept: PhysicsConcept,
  unlockedPhysics: PhysicsConcept[]
): boolean {
  // Check prerequisites
  const unlockedIds = new Set(unlockedPhysics.map(p => p.id));
  const prereqsMet = concept.prerequisiteIds.every(id => unlockedIds.has(id));
  
  if (!prereqsMet) return false;
  
  // Already unlocked?
  if (unlockedIds.has(concept.id)) return false;
  
  // Curiosity and creativity affect discovery chance
  const discoveryChance = agent.genes.curiosity * agent.genes.creativity * 0.01;
  
  // Complexity affects difficulty
  const complexityPenalty = Math.pow(0.9, concept.complexity);
  
  return Math.random() < discoveryChance * complexityPenalty;
}

/**
 * Check if agent can discover a math concept
 */
export function canDiscoverMath(
  agent: Agent,
  concept: MathConcept,
  unlockedMath: MathConcept[]
): boolean {
  // Check prerequisites
  const unlockedIds = new Set(unlockedMath.map(m => m.id));
  const prereqsMet = concept.prerequisiteIds.every(id => unlockedIds.has(id));
  
  if (!prereqsMet) return false;
  
  // Already unlocked?
  if (unlockedIds.has(concept.id)) return false;
  
  // Curiosity and patience affect discovery chance
  const discoveryChance = agent.genes.curiosity * agent.genes.patience * 0.01;
  
  // Complexity affects difficulty
  const complexityPenalty = Math.pow(0.9, concept.complexity);
  
  return Math.random() < discoveryChance * complexityPenalty;
}

/**
 * Attempt scientific discovery for an agent
 */
export function attemptScientificDiscovery(
  agent: Agent,
  scienceState: ScienceState,
  tick: number
): {
  scienceState: ScienceState;
  discoveries: ScientificDiscovery[];
} {
  const newDiscoveries: ScientificDiscovery[] = [];
  
  // Only agents with sufficient energy can do research
  if (agent.energy < 20) {
    return { scienceState, discoveries: [] };
  }
  
  // Try to discover physics
  const availablePhysics = PHYSICS_CONCEPTS.filter(
    concept => !scienceState.unlockedPhysics.find(p => p.id === concept.id)
  );
  
  for (const concept of availablePhysics) {
    if (canDiscoverPhysics(agent, concept, scienceState.unlockedPhysics)) {
      const discovered: PhysicsConcept = {
        ...concept,
        discoveredAt: tick,
        discoveredBy: agent.id
      };
      
      scienceState.unlockedPhysics.push(discovered);
      scienceState.civilizationKnowledge.physics[concept.id] = discovered;
      scienceState.currentEra.physicsUnlocked.push(concept.id);
      
      const discovery: ScientificDiscovery = {
        id: `physics_${concept.id}_${tick}`,
        name: concept.name,
        category: 'physics',
        discoveredAt: tick,
        discoveredBy: agent.id,
        eraLevel: scienceState.currentEra.level,
        significance: concept.complexity / 10,
        description: concept.description,
        prerequisiteIds: concept.prerequisiteIds,
        enablesIds: []
      };
      
      newDiscoveries.push(discovery);
      break; // One discovery per tick
    }
  }
  
  // Try to discover mathematics (if no physics discovered)
  if (newDiscoveries.length === 0) {
    const availableMath = MATH_CONCEPTS.filter(
      concept => !scienceState.unlockedMath.find(m => m.id === concept.id)
    );
    
    for (const concept of availableMath) {
      if (canDiscoverMath(agent, concept, scienceState.unlockedMath)) {
        const discovered: MathConcept = {
          ...concept,
          discoveredAt: tick,
          discoveredBy: agent.id
        };
        
        scienceState.unlockedMath.push(discovered);
        scienceState.civilizationKnowledge.mathematics[concept.id] = discovered;
        scienceState.currentEra.mathUnlocked.push(concept.id);
        
        const discovery: ScientificDiscovery = {
          id: `math_${concept.id}_${tick}`,
          name: concept.name,
          category: 'mathematics',
          discoveredAt: tick,
          discoveredBy: agent.id,
          eraLevel: scienceState.currentEra.level,
          significance: concept.complexity / 10,
          description: concept.description,
          prerequisiteIds: concept.prerequisiteIds,
          enablesIds: []
        };
        
        newDiscoveries.push(discovery);
        break; // One discovery per tick
      }
    }
  }
  
  return { scienceState, discoveries: newDiscoveries };
}

/**
 * Update science state each tick
 */
export function updateScienceSystem(
  scienceState: ScienceState,
  agents: Agent[],
  tick: number
): {
  scienceState: ScienceState;
  discoveries: ScientificDiscovery[];
  eraAdvanced: boolean;
  newEra?: ScientificEra;
} {
  const allDiscoveries: ScientificDiscovery[] = [];
  let eraAdvanced = false;
  let newEra: ScientificEra | undefined;
  
  // Each agent has a chance to make discoveries
  for (const agent of agents) {
    const result = attemptScientificDiscovery(agent, scienceState, tick);
    scienceState = result.scienceState;
    allDiscoveries.push(...result.discoveries);
  }
  
  // Update metrics
  scienceState.progressionMetrics.totalDiscoveries += allDiscoveries.length;
  scienceState.allDiscoveries.push(...allDiscoveries);
  
  const currentEraLevel = scienceState.currentEra.level;
  scienceState.progressionMetrics.discoveriesPerEra[currentEraLevel] = 
    (scienceState.progressionMetrics.discoveriesPerEra[currentEraLevel] || 0) + allDiscoveries.length;
  
  scienceState.progressionMetrics.ticksSinceLastEra++;
  
  // Calculate discovery rate
  if (tick > 0) {
    scienceState.progressionMetrics.averageDiscoveryRate = 
      scienceState.progressionMetrics.totalDiscoveries / tick;
  }
  
  // Check for era advancement
  if (shouldAdvanceEra(
    scienceState.currentEra,
    scienceState.progressionMetrics,
    scienceState.unlockedPhysics.length,
    scienceState.unlockedMath.length
  )) {
    const nextLevel = currentEraLevel + 1;
    
    // Get next era (from base or generate)
    const nextEraTemplate = nextLevel < BASE_ERAS.length
      ? BASE_ERAS[nextLevel]
      : generateNextEra(nextLevel);
    
    newEra = {
      ...nextEraTemplate,
      startTick: tick,
      discoveries: [],
      physicsUnlocked: [],
      mathUnlocked: []
    };
    
    scienceState.currentEra = newEra;
    scienceState.allEras.push(newEra);
    scienceState.progressionMetrics.currentEraLevel = nextLevel;
    scienceState.progressionMetrics.ticksSinceLastEra = 0;
    scienceState.progressionMetrics.discoveriesPerEra[nextLevel] = 0;
    
    eraAdvanced = true;
  }
  
  return {
    scienceState,
    discoveries: allDiscoveries,
    eraAdvanced,
    newEra
  };
}

/**
 * Get science bonuses for an agent based on civilization knowledge
 */
export function getScienceBonuses(scienceState: ScienceState) {
  const physicsBonuses = calculatePhysicsBonuses(scienceState.unlockedPhysics);
  const mathBonuses = calculateMathBonuses(scienceState.unlockedMath);
  
  return {
    physics: physicsBonuses,
    math: mathBonuses,
    combined: {
      energyEfficiency: physicsBonuses.energyEfficiency,
      movementSpeed: physicsBonuses.movementSpeed,
      learningRate: physicsBonuses.learningRate * mathBonuses.decisionQuality,
      inventionChance: physicsBonuses.inventionChance,
      exploration: mathBonuses.explorationBonus,
      optimization: mathBonuses.optimizationPower
    }
  };
}

/**
 * Export all necessary types and functions
 */
export * from './eras';
export * from './physics';
export * from './mathematics';
