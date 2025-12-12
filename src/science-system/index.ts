/**
 * Science System Coordinator
 * Manages physics, mathematics, and era progression with unlimited scaling
 * 
 * Updated to integrate expanded physics system with new bonus types:
 * - curiosity: Drive to explore unknown areas and concepts
 * - cooperationBonus: Benefits when near other agents
 * - dimensionalAccess: Ability to interact with parallel simulation layers
 * - informationProcessing: Faster decision-making and pattern recognition
 * - entropicResistance: Slower energy decay / resistance to disorder
 */

import type { Agent } from '../types';
import type { ScientificEra, ScientificDiscovery, ProgressionMetrics } from './eras';
import type { PhysicsConcept } from './physics';
import type { MathConcept } from './mathematics';
import { BASE_ERAS, generateNextEra, shouldAdvanceEra } from './eras';
import { 
  PHYSICS_CONCEPTS, 
  generateAdvancedPhysicsConcept, 
  calculatePhysicsBonuses,
  getDiscoverableConcepts,
  calculateDiscoveryProbability,
  getPrerequisiteChain,
  getAllPhysicsConcepts
} from './physics';
import { 
  MATH_CONCEPTS, 
  generateAdvancedMathConcept, 
  calculateMathBonuses 
} from './mathematics';

/**
 * Extended science state with tracking for all bonus types
 */
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
  // Track procedural generation levels
  generatedPhysicsLevel: number;
  generatedMathLevel: number;
  // Track discovery streaks and momentum
  discoveryMomentum: number;
  lastDiscoveryTick: number;
  // Track collaborative discoveries
  collaborativeDiscoveries: number;
}

/**
 * Combined bonuses from all scientific knowledge
 */
export interface CombinedScienceBonuses {
  // Core bonuses
  energyEfficiency: number;
  movementSpeed: number;
  learningRate: number;
  inventionChance: number;
  exploration: number;
  optimization: number;
  // New expanded bonuses
  curiosity: number;
  cooperationBonus: number;
  dimensionalAccess: number;
  informationProcessing: number;
  entropicResistance: number;
  // Derived bonuses
  spaceManipulation: number;
  timePerception: number;
  decisionQuality: number;
  patternRecognition: number;
  abstractThinking: number;
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
    allDiscoveries: [],    progressionMetrics: {
      totalDiscoveries: 0,
      discoveriesPerEra: { 0: 0 },
      averageDiscoveryRate: 0,
      currentEraLevel: 0,
      ticksSinceLastEra: 0,
      scientificAcceleration: 0,
      // Extended metrics
      collaborativeDiscoveries: 0,
      categoryBreakthroughs: 0,
      peakMomentum: 1.0,
      averageComplexity: 0,
      physicsCategories: 0,
      mathCategories: 0,
      transcendentDiscoveries: 0
    },
    civilizationKnowledge: {
      physics: {},
      mathematics: {}
    },
    generatedPhysicsLevel: 0,
    generatedMathLevel: 0,
    discoveryMomentum: 1.0,
    lastDiscoveryTick: tick,
    collaborativeDiscoveries: 0
  };
}

/**
 * Calculate the curiosity-driven discovery modifier
 * Curiosity compounds with existing knowledge - the more you know, the more curious you become
 */
function calculateCuriosityModifier(
  agent: Agent,
  scienceState: ScienceState,
  bonuses: ReturnType<typeof calculatePhysicsBonuses>
): number {
  // Base curiosity from agent genes
  const baseCuriosity = agent.genes.curiosity || 0.5;
  
  // Curiosity bonus from unlocked physics concepts
  const physicsCuriosityBonus = bonuses.curiosity || 1.0;
  
  // Knowledge breadth bonus - knowing diverse fields increases curiosity
  const physicsCategories = new Set(scienceState.unlockedPhysics.map(p => p.category));
  const breadthBonus = 1 + (physicsCategories.size * 0.05);
  
  // Discovery momentum - recent discoveries fuel more curiosity
  const momentumBonus = scienceState.discoveryMomentum;
  
  return baseCuriosity * physicsCuriosityBonus * breadthBonus * momentumBonus;
}

/**
 * Calculate cooperation modifier for collaborative discoveries
 */
function calculateCooperationModifier(
  agent: Agent,
  nearbyAgents: Agent[],
  bonuses: ReturnType<typeof calculatePhysicsBonuses>
): number {
  if (nearbyAgents.length === 0) return 1.0;
  
  // Base cooperation from physics bonuses
  const baseCooperation = bonuses.cooperationBonus || 1.0;
  
  // More nearby agents = more potential for collaboration
  const proximityBonus = 1 + (Math.min(nearbyAgents.length, 5) * 0.1);
  
  // Average creativity of nearby agents boosts collaborative potential
  const avgCreativity = nearbyAgents.reduce((sum, a) => sum + (a.genes.creativity || 0.5), 0) / nearbyAgents.length;
  const creativityBonus = 1 + (avgCreativity * 0.2);
  
  return baseCooperation * proximityBonus * creativityBonus;
}

/**
 * Calculate information processing advantage for discovery
 */
function calculateInformationAdvantage(
  agent: Agent,
  concept: PhysicsConcept,
  bonuses: ReturnType<typeof calculatePhysicsBonuses>
): number {
  // Base information processing from bonuses
  const baseProcessing = bonuses.informationProcessing || 1.0;
  
  // More complex concepts benefit more from information processing
  const complexityFactor = 1 + (concept.complexity / 20);
  
  // Agent's pattern recognition ability (if available)
  const patternBonus = 1 + ((agent.genes.creativity || 0.5) * 0.3);
  
  return baseProcessing * complexityFactor * patternBonus;
}

/**
 * Check if agent can discover a physics concept - ENHANCED with new bonuses
 */
export function canDiscoverPhysics(
  agent: Agent,
  concept: PhysicsConcept,
  scienceState: ScienceState,
  nearbyAgents: Agent[] = []
): boolean {
  const unlockedIds = new Set(scienceState.unlockedPhysics.map(p => p.id));
  
  // Check prerequisites
  const prereqsMet = concept.prerequisiteIds.every(id => unlockedIds.has(id));
  if (!prereqsMet) return false;
  
  // Already unlocked?
  if (unlockedIds.has(concept.id)) return false;
  
  // Calculate current bonuses
  const bonuses = calculatePhysicsBonuses(scienceState.unlockedPhysics);
  
  // CURIOSITY is now a primary driver of discovery
  const curiosityMod = calculateCuriosityModifier(agent, scienceState, bonuses);
  
  // Cooperation bonus for nearby agents
  const cooperationMod = calculateCooperationModifier(agent, nearbyAgents, bonuses);
  
  // Information processing helps with complex concepts
  const infoMod = calculateInformationAdvantage(agent, concept, bonuses);
  
  // Base discovery chance from agent genes
  const baseChance = agent.genes.curiosity * agent.genes.creativity * 0.008;
  
  // Learning rate bonus
  const learningMod = bonuses.learningRate || 1.0;
  
  // Invention chance bonus (additive)
  const inventionBonus = bonuses.inventionChance || 0;
  
  // Complexity penalty (softer with high information processing)
  const effectiveComplexity = concept.complexity / Math.sqrt(infoMod);
  const complexityPenalty = Math.pow(0.92, effectiveComplexity);
  
  // Dimensional access can help discover exotic/transcendent concepts
  const dimensionalBonus = (concept.category === 'exotic' || concept.category === 'transcendent')
    ? (bonuses.dimensionalAccess || 1.0)
    : 1.0;
  
  // Time perception helps with relativity/cosmology concepts
  const timeBonus = (concept.category === 'relativity' || concept.category === 'cosmology')
    ? (bonuses.timePerception || 1.0)
    : 1.0;
  
  // Final discovery probability
  const finalChance = (baseChance + inventionBonus) 
    * curiosityMod 
    * cooperationMod 
    * learningMod 
    * complexityPenalty 
    * dimensionalBonus 
    * timeBonus
    * (infoMod > 1 ? Math.sqrt(infoMod) : 1);
  
  return Math.random() < finalChance;
}

/**
 * Check if agent can discover a math concept - ENHANCED
 */
export function canDiscoverMath(
  agent: Agent,
  concept: MathConcept,
  scienceState: ScienceState,
  nearbyAgents: Agent[] = []
): boolean {
  const unlockedIds = new Set(scienceState.unlockedMath.map(m => m.id));
  
  // Check prerequisites
  const prereqsMet = concept.prerequisiteIds.every(id => unlockedIds.has(id));
  if (!prereqsMet) return false;
  
  // Already unlocked?
  if (unlockedIds.has(concept.id)) return false;
  
  // Get bonuses from both physics and math
  const physicsBonuses = calculatePhysicsBonuses(scienceState.unlockedPhysics);
  const mathBonuses = calculateMathBonuses(scienceState.unlockedMath);
  
  // Curiosity drives mathematical exploration too
  const curiosityMod = calculateCuriosityModifier(agent, scienceState, physicsBonuses);
  
  // Information processing is crucial for mathematics
  const infoProcessing = physicsBonuses.informationProcessing || 1.0;
  
  // Base discovery chance
  const baseChance = agent.genes.curiosity * agent.genes.patience * 0.008;
  
  // Pattern recognition from math helps discover more math
  const patternBonus = mathBonuses.patternRecognition || 1.0;
  
  // Abstract thinking helps with higher-level math
  const abstractBonus = concept.complexity > 8 
    ? (mathBonuses.abstractThinking || 1.0) 
    : 1.0;
  
  // Complexity penalty (softer with good info processing and pattern recognition)
  const effectiveComplexity = concept.complexity / Math.sqrt(infoProcessing * patternBonus);
  const complexityPenalty = Math.pow(0.92, effectiveComplexity);
  
  // Final probability
  const finalChance = baseChance 
    * curiosityMod 
    * complexityPenalty 
    * patternBonus 
    * abstractBonus
    * infoProcessing;
  
  return Math.random() < finalChance;
}

/**
 * Get all currently discoverable physics concepts
 */
export function getAvailablePhysicsDiscoveries(scienceState: ScienceState): PhysicsConcept[] {
  const unlockedIds = new Set(scienceState.unlockedPhysics.map(p => p.id));
  
  // Get base concepts plus any generated ones
  const allConcepts = getAllPhysicsConcepts(scienceState.generatedPhysicsLevel);
  
  return allConcepts.filter(concept => {
    if (unlockedIds.has(concept.id)) return false;
    return concept.prerequisiteIds.every(id => unlockedIds.has(id));
  });
}

/**
 * Attempt scientific discovery for an agent - ENHANCED
 */
export function attemptScientificDiscovery(
  agent: Agent,
  scienceState: ScienceState,
  tick: number,
  nearbyAgents: Agent[] = []
): {
  scienceState: ScienceState;
  discoveries: ScientificDiscovery[];
  wasCollaborative: boolean;
} {
  const newDiscoveries: ScientificDiscovery[] = [];
  let wasCollaborative = false;
  
  // Energy requirement - but entropic resistance reduces it
  const physicsBonuses = calculatePhysicsBonuses(scienceState.unlockedPhysics);
  const entropicResist = physicsBonuses.entropicResistance || 1.0;
  const energyRequired = Math.max(5, 20 / entropicResist);
  
  if (agent.energy < energyRequired) {
    return { scienceState, discoveries: [], wasCollaborative: false };
  }
  
  // Get available physics concepts to discover
  const availablePhysics = getAvailablePhysicsDiscoveries(scienceState);
  
  // Sort by complexity (try easier ones first, but curiosity can override)
  const curiosityMod = calculateCuriosityModifier(agent, scienceState, physicsBonuses);
  const sortedPhysics = [...availablePhysics].sort((a, b) => {
    // High curiosity? Try harder concepts
    if (curiosityMod > 2) {
      return b.complexity - a.complexity;
    }
    return a.complexity - b.complexity;
  });
  
  // Try to discover physics
  for (const concept of sortedPhysics) {
    if (canDiscoverPhysics(agent, concept, scienceState, nearbyAgents)) {
      const discovered: PhysicsConcept = {
        ...concept,
        discoveredAt: tick,
        discoveredBy: agent.id
      };
      
      scienceState.unlockedPhysics.push(discovered);
      scienceState.civilizationKnowledge.physics[concept.id] = discovered;
      scienceState.currentEra.physicsUnlocked.push(concept.id);
      
      // Check if this was a collaborative discovery
      wasCollaborative = nearbyAgents.length > 0 && 
        (physicsBonuses.cooperationBonus || 1.0) > 1.0;
      
      if (wasCollaborative) {
        scienceState.collaborativeDiscoveries++;
      }
      
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
        enablesIds: findEnabledConcepts(concept.id, scienceState),
        wasCollaborative,
        collaboratorCount: wasCollaborative ? nearbyAgents.length : 0
      };
      
      newDiscoveries.push(discovery);
      
      // Update discovery momentum
      const ticksSinceLast = tick - scienceState.lastDiscoveryTick;
      if (ticksSinceLast < 100) {
        scienceState.discoveryMomentum = Math.min(3.0, scienceState.discoveryMomentum * 1.1);
      } else {
        scienceState.discoveryMomentum = Math.max(1.0, scienceState.discoveryMomentum * 0.95);
      }
      scienceState.lastDiscoveryTick = tick;
      
      // Check if we need to generate more advanced physics concepts
      if (scienceState.unlockedPhysics.length >= PHYSICS_CONCEPTS.length - 5) {
        scienceState.generatedPhysicsLevel++;
      }
      
      break; // One discovery per tick
    }
  }
  
  // Try to discover mathematics (if no physics discovered)
  if (newDiscoveries.length === 0) {
    const availableMath = MATH_CONCEPTS.filter(
      concept => !scienceState.unlockedMath.find(m => m.id === concept.id)
    );
    
    for (const concept of availableMath) {
      if (canDiscoverMath(agent, concept, scienceState, nearbyAgents)) {
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
        
        // Update discovery momentum
        scienceState.discoveryMomentum = Math.min(3.0, scienceState.discoveryMomentum * 1.05);
        scienceState.lastDiscoveryTick = tick;
        
        break; // One discovery per tick
      }
    }
  }
  
  return { scienceState, discoveries: newDiscoveries, wasCollaborative };
}

/**
 * Find concepts that become discoverable after unlocking a concept
 */
function findEnabledConcepts(conceptId: string, scienceState: ScienceState): string[] {
  const allConcepts = getAllPhysicsConcepts(scienceState.generatedPhysicsLevel);
  const unlockedIds = new Set(scienceState.unlockedPhysics.map(p => p.id));
  unlockedIds.add(conceptId);
  
  return allConcepts
    .filter(c => {
      if (unlockedIds.has(c.id)) return false;
      // Check if this concept was blocked only by the newly discovered one
      const prereqsWithout = c.prerequisiteIds.filter(id => id !== conceptId);
      const allOthersMet = prereqsWithout.every(id => unlockedIds.has(id));
      const includesNew = c.prerequisiteIds.includes(conceptId);
      return allOthersMet && includesNew;
    })
    .map(c => c.id);
}

/**
 * Update science state each tick - ENHANCED
 */
export function updateScienceSystem(
  scienceState: ScienceState,
  agents: Agent[],
  tick: number,
  agentProximityMap?: Map<number, Agent[]> // Optional map of agent ID to nearby agents
): {
  scienceState: ScienceState;
  discoveries: ScientificDiscovery[];
  eraAdvanced: boolean;
  newEra?: ScientificEra;
  collaborativeCount: number;
} {
  const allDiscoveries: ScientificDiscovery[] = [];
  let eraAdvanced = false;
  let newEra: ScientificEra | undefined;
  let collaborativeCount = 0;
  
  // Decay discovery momentum over time
  scienceState.discoveryMomentum = Math.max(
    1.0, 
    scienceState.discoveryMomentum - 0.001
  );
  
  // Each agent has a chance to make discoveries
  for (const agent of agents) {
    const nearbyAgents = agentProximityMap?.get(agent.id) || [];
    const result = attemptScientificDiscovery(agent, scienceState, tick, nearbyAgents);
    scienceState = result.scienceState;
    allDiscoveries.push(...result.discoveries);
    if (result.wasCollaborative) collaborativeCount++;
  }
  
  // Update metrics
  scienceState.progressionMetrics.totalDiscoveries += allDiscoveries.length;
  scienceState.allDiscoveries.push(...allDiscoveries);
  
  const currentEraLevel = scienceState.currentEra.level;
  scienceState.progressionMetrics.discoveriesPerEra[currentEraLevel] = 
    (scienceState.progressionMetrics.discoveriesPerEra[currentEraLevel] || 0) + allDiscoveries.length;
  
  scienceState.progressionMetrics.ticksSinceLastEra++;
  
  // Calculate discovery rate with acceleration
  if (tick > 0) {
    const oldRate = scienceState.progressionMetrics.averageDiscoveryRate;
    const newRate = scienceState.progressionMetrics.totalDiscoveries / tick;
    scienceState.progressionMetrics.averageDiscoveryRate = newRate;
    
    // Track acceleration (is discovery rate increasing?)
    if (oldRate > 0) {
      scienceState.progressionMetrics.scientificAcceleration = (newRate - oldRate) / oldRate;
    }
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
    
    // Era advancement boosts discovery momentum
    scienceState.discoveryMomentum = Math.min(5.0, scienceState.discoveryMomentum * 1.5);
    
    eraAdvanced = true;
  }
  
  return {
    scienceState,
    discoveries: allDiscoveries,
    eraAdvanced,
    newEra,
    collaborativeCount
  };
}

/**
 * Get comprehensive science bonuses for an agent based on civilization knowledge
 * Now includes all expanded physics bonuses
 */
export function getScienceBonuses(scienceState: ScienceState): {
  physics: ReturnType<typeof calculatePhysicsBonuses>;
  math: ReturnType<typeof calculateMathBonuses>;
  combined: CombinedScienceBonuses;
} {
  const physicsBonuses = calculatePhysicsBonuses(scienceState.unlockedPhysics);
  const mathBonuses = calculateMathBonuses(scienceState.unlockedMath);
  
  return {
    physics: physicsBonuses,
    math: mathBonuses,
    combined: {
      // Core bonuses (physics-driven)
      energyEfficiency: physicsBonuses.energyEfficiency,
      movementSpeed: physicsBonuses.movementSpeed,
      learningRate: physicsBonuses.learningRate * (mathBonuses.decisionQuality || 1.0),
      inventionChance: physicsBonuses.inventionChance,
      exploration: mathBonuses.explorationBonus || 1.0,
      optimization: mathBonuses.optimizationPower || 1.0,
      
      // NEW expanded bonuses from physics
      curiosity: physicsBonuses.curiosity,
      cooperationBonus: physicsBonuses.cooperationBonus,
      dimensionalAccess: physicsBonuses.dimensionalAccess,
      informationProcessing: physicsBonuses.informationProcessing * (mathBonuses.patternRecognition || 1.0),
      entropicResistance: physicsBonuses.entropicResistance,
      
      // Spatial/temporal from physics
      spaceManipulation: physicsBonuses.spaceManipulation,
      timePerception: physicsBonuses.timePerception,
      
      // Math-driven bonuses
      decisionQuality: mathBonuses.decisionQuality || 1.0,
      patternRecognition: mathBonuses.patternRecognition || 1.0,
      abstractThinking: mathBonuses.abstractThinking || 1.0
    }
  };
}

/**
 * Apply entropic resistance to an agent's energy decay
 */
export function applyEntropicResistance(
  baseDecay: number,
  scienceState: ScienceState
): number {
  const bonuses = calculatePhysicsBonuses(scienceState.unlockedPhysics);
  const resistance = bonuses.entropicResistance || 1.0;
  
  // Higher resistance = lower decay
  return baseDecay / resistance;
}

/**
 * Calculate dimensional access effects (for future multi-layer simulations)
 */
export function getDimensionalAccessLevel(scienceState: ScienceState): {
  level: number;
  canAccessParallel: boolean;
  canPerceiveAlternates: boolean;
  canManipulateLayers: boolean;
} {
  const bonuses = calculatePhysicsBonuses(scienceState.unlockedPhysics);
  const access = bonuses.dimensionalAccess || 1.0;
  
  return {
    level: access,
    canAccessParallel: access >= 1.5,
    canPerceiveAlternates: access >= 2.0,
    canManipulateLayers: access >= 2.5
  };
}

/**
 * Get time perception effects (extra ticks, preview ability)
 */
export function getTimePerceptionEffects(scienceState: ScienceState): {
  level: number;
  bonusTicks: number;
  canPreviewOutcomes: boolean;
  temporalAwareness: number;
} {
  const bonuses = calculatePhysicsBonuses(scienceState.unlockedPhysics);
  const perception = bonuses.timePerception || 1.0;
  
  // Bonus ticks scale with time perception
  const bonusTicks = Math.floor(Math.log2(perception));
  
  return {
    level: perception,
    bonusTicks: bonusTicks,
    canPreviewOutcomes: perception >= 1.5,
    temporalAwareness: Math.min(1.0, (perception - 1) / 2)
  };
}

/**
 * Get cooperation effects for multi-agent interactions
 */
export function getCooperationEffects(scienceState: ScienceState): {
  level: number;
  proximityBonus: number;
  knowledgeShareRate: number;
  collaborativeDiscoveryBoost: number;
} {
  const bonuses = calculatePhysicsBonuses(scienceState.unlockedPhysics);
  const cooperation = bonuses.cooperationBonus || 1.0;
  
  return {
    level: cooperation,
    proximityBonus: 1 + (cooperation - 1) * 0.5,
    knowledgeShareRate: Math.min(1.0, cooperation / 2),
    collaborativeDiscoveryBoost: cooperation
  };
}

/**
 * Get curiosity-driven exploration parameters
 */
export function getCuriosityDrivenBehavior(scienceState: ScienceState): {
  level: number;
  explorationBias: number;       // Preference for unknown areas
  noveltySeekingStrength: number; // Drive toward new experiences
  complexityPreference: number;   // Attraction to complex problems
  discoveryDrive: number;         // Overall discovery motivation
} {
  const bonuses = calculatePhysicsBonuses(scienceState.unlockedPhysics);
  const curiosity = bonuses.curiosity || 1.0;
  
  return {
    level: curiosity,
    explorationBias: Math.min(0.9, 0.3 + (curiosity - 1) * 0.2),
    noveltySeekingStrength: curiosity,
    complexityPreference: 1 + Math.log2(curiosity),
    discoveryDrive: curiosity * (bonuses.learningRate || 1.0)
  };
}

/**
 * Get information processing capabilities
 */
export function getInformationProcessingCapabilities(scienceState: ScienceState): {
  level: number;
  decisionSpeed: number;
  patternComplexity: number;    // Max pattern complexity that can be recognized
  parallelProcessing: number;   // Number of simultaneous considerations
  memoryEfficiency: number;     // How well information is retained
} {
  const physicsBonuses = calculatePhysicsBonuses(scienceState.unlockedPhysics);
  const mathBonuses = calculateMathBonuses(scienceState.unlockedMath);
  
  const infoProc = physicsBonuses.informationProcessing || 1.0;
  const patternRec = mathBonuses.patternRecognition || 1.0;
  
  return {
    level: infoProc * patternRec,
    decisionSpeed: infoProc,
    patternComplexity: Math.floor(5 + Math.log2(patternRec) * 3),
    parallelProcessing: Math.floor(1 + Math.log2(infoProc)),
    memoryEfficiency: Math.min(1.0, 0.5 + (infoProc - 1) * 0.25)
  };
}

/**
 * Get a summary of the science system state for UI/debugging
 */
export function getScienceSystemSummary(scienceState: ScienceState): {
  era: string;
  eraLevel: number;
  physicsUnlocked: number;
  physicsTotal: number;
  mathUnlocked: number;
  mathTotal: number;
  totalDiscoveries: number;
  discoveryRate: number;
  momentum: number;
  collaborativeDiscoveries: number;
  topBonuses: { name: string; value: number }[];
} {
  const bonuses = getScienceBonuses(scienceState);
  
  // Find top bonuses
  const allBonuses = Object.entries(bonuses.combined)
    .map(([name, value]) => ({ name, value: value as number }))
    .filter(b => b.value !== 1.0 && b.value !== 0)
    .sort((a, b) => {
      // Sort by how far from 1.0 (neutral) the value is
      const aDiff = Math.abs(Math.log(a.value));
      const bDiff = Math.abs(Math.log(b.value));
      return bDiff - aDiff;
    })
    .slice(0, 5);
  
  return {
    era: scienceState.currentEra.name,
    eraLevel: scienceState.currentEra.level,
    physicsUnlocked: scienceState.unlockedPhysics.length,
    physicsTotal: PHYSICS_CONCEPTS.length + scienceState.generatedPhysicsLevel,
    mathUnlocked: scienceState.unlockedMath.length,
    mathTotal: MATH_CONCEPTS.length + scienceState.generatedMathLevel,
    totalDiscoveries: scienceState.progressionMetrics.totalDiscoveries,
    discoveryRate: scienceState.progressionMetrics.averageDiscoveryRate,
    momentum: scienceState.discoveryMomentum,
    collaborativeDiscoveries: scienceState.collaborativeDiscoveries,
    topBonuses: allBonuses
  };
}

/**
 * Export all necessary types and functions
 * Note: Using explicit exports to avoid naming conflicts between physics and mathematics
 */
export * from './eras';

// Physics exports - prefix conflicting names
export { 
  PHYSICS_CONCEPTS, 
  generateAdvancedPhysicsConcept, 
  calculatePhysicsBonuses,
  getDiscoverableConcepts as getDiscoverablePhysicsConcepts,
  calculateDiscoveryProbability as calculatePhysicsDiscoveryProbability,
  getPrerequisiteChain as getPhysicsPrerequisiteChain,
  getAllPhysicsConcepts,
  getConceptsByCategory as getPhysicsConceptsByCategory,
  validatePrerequisites as validatePhysicsPrerequisites
} from './physics';
export type { PhysicsConcept } from './physics';

// Mathematics exports - prefix conflicting names  
export { 
  MATH_CONCEPTS, 
  generateAdvancedMathConcept, 
  calculateMathBonuses,
  getConceptsByCategory as getMathConceptsByCategory,
  getPrerequisiteChain as getMathPrerequisiteChain,
  validatePrerequisites as validateMathPrerequisites
} from './mathematics';
export type { MathConcept } from './mathematics';