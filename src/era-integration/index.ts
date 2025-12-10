/**
 * ERA INTEGRATION SYSTEM
 * 
 * Integrates scientific eras with civilization progression:
 * - Era advancement based on physics/math discoveries and inventions
 * - Era-specific agent abilities and bonuses
 * - Scientific milestones and achievements
 * - Knowledge transmission to offspring based on current era
 */

import type { Agent } from '../types';
import type { PhysicsConcept } from '../science-system/physics';
import type { MathConcept } from '../science-system/mathematics';
import { 
  ScientificEra, 
  ProgressionMetrics, 
  BASE_ERAS, 
  generateNextEra,
  shouldAdvanceEra 
} from '../science-system/eras';

/**
 * Civilization era state
 */
export interface CivilizationEraState {
  currentEra: ScientificEra;
  allEras: ScientificEra[];           // History of all eras reached
  metrics: ProgressionMetrics;
  milestones: EpochMilestone[];       // Achievements unlocked
  lastEraAdvanceTick: number;
}

/**
 * Milestones that can be achieved
 */
export interface EpochMilestone {
  id: string;
  name: string;
  description: string;
  achievedAt: number;                 // Tick when achieved
  eraLevel: number;                   // Era when achieved
  type: 'discovery' | 'population' | 'invention' | 'collaboration' | 'evolution';
}

/**
 * Era-specific bonuses for agents
 */
export interface EraAgentBonuses {
  learningMultiplier: number;         // Q-learning boost
  discoveryChance: number;            // Additional discovery chance
  energyEfficiency: number;           // Energy cost multiplier (< 1 = more efficient)
  reproductionBonus: number;          // Flat bonus to reproduction threshold
  knowledgeRetention: number;         // 0-1: How much knowledge offspring inherit
  collaborationRange: number;         // Range for collaboration bonuses
}

/**
 * Predefined milestones the civilization can achieve
 */
export const EPOCH_MILESTONES: Omit<EpochMilestone, 'achievedAt'>[] = [
  // Discovery milestones
  { id: 'first_physics', name: 'First Physicist', description: 'First physics concept discovered', eraLevel: 0, type: 'discovery' },
  { id: 'first_math', name: 'First Mathematician', description: 'First math concept discovered', eraLevel: 0, type: 'discovery' },
  { id: 'physics_master', name: 'Physics Mastery', description: '10 physics concepts discovered', eraLevel: 1, type: 'discovery' },
  { id: 'math_master', name: 'Mathematics Mastery', description: '10 math concepts discovered', eraLevel: 1, type: 'discovery' },
  { id: 'unified_science', name: 'Unified Science', description: '25+ physics and math concepts each', eraLevel: 3, type: 'discovery' },
  
  // Population milestones
  { id: 'first_village', name: 'First Village', description: 'Population reached 10', eraLevel: 0, type: 'population' },
  { id: 'growing_town', name: 'Growing Town', description: 'Population reached 25', eraLevel: 1, type: 'population' },
  { id: 'bustling_city', name: 'Bustling City', description: 'Population reached 50', eraLevel: 2, type: 'population' },
  { id: 'metropolis', name: 'Metropolis', description: 'Population reached 100', eraLevel: 3, type: 'population' },
  
  // Invention milestones
  { id: 'first_invention', name: 'First Inventor', description: 'First invention created', eraLevel: 0, type: 'invention' },
  { id: 'inventor_society', name: 'Inventor Society', description: '50 total inventions', eraLevel: 1, type: 'invention' },
  { id: 'innovation_age', name: 'Innovation Age', description: '200 total inventions', eraLevel: 3, type: 'invention' },
  
  // Collaboration milestones
  { id: 'first_collab', name: 'First Collaboration', description: 'Discovery made through collaboration', eraLevel: 0, type: 'collaboration' },
  { id: 'research_team', name: 'Research Team', description: '5 collaborative discoveries', eraLevel: 1, type: 'collaboration' },
  { id: 'scientific_community', name: 'Scientific Community', description: '20 collaborative discoveries', eraLevel: 2, type: 'collaboration' },
  
  // Era advancement milestones
  { id: 'bronze_age_reached', name: 'Bronze Age Pioneer', description: 'Advanced to Bronze Age', eraLevel: 1, type: 'evolution' },
  { id: 'iron_age_reached', name: 'Iron Age Pioneer', description: 'Advanced to Iron Age', eraLevel: 2, type: 'evolution' },
  { id: 'classical_age_reached', name: 'Classical Age Pioneer', description: 'Advanced to Classical Age', eraLevel: 3, type: 'evolution' },
  { id: 'renaissance_reached', name: 'Renaissance Pioneer', description: 'Advanced to Renaissance', eraLevel: 4, type: 'evolution' },
  { id: 'industrial_age_reached', name: 'Industrial Pioneer', description: 'Advanced to Industrial Age', eraLevel: 5, type: 'evolution' },
  { id: 'information_age_reached', name: 'Information Pioneer', description: 'Advanced to Information Age', eraLevel: 6, type: 'evolution' },
  { id: 'quantum_age_reached', name: 'Quantum Pioneer', description: 'Advanced to Quantum Age', eraLevel: 7, type: 'evolution' },
  { id: 'singularity_reached', name: 'Singularity Pioneer', description: 'Advanced to Singularity Age', eraLevel: 8, type: 'evolution' }
];

/**
 * Initialize civilization era state
 */
export function initializeCivilizationEra(): CivilizationEraState {
  const initialEra: ScientificEra = {
    ...BASE_ERAS[0],
    startTick: 0,
    discoveries: [],
    physicsUnlocked: [],
    mathUnlocked: []
  };
  
  return {
    currentEra: initialEra,
    allEras: [initialEra],
    metrics: {
      totalDiscoveries: 0,
      discoveriesPerEra: { 0: 0 },
      averageDiscoveryRate: 0,
      currentEraLevel: 0,
      ticksSinceLastEra: 0,
      scientificAcceleration: 1.0
    },
    milestones: [],
    lastEraAdvanceTick: 0
  };
}

/**
 * Get era-specific bonuses based on current era level
 */
export function getEraBonuses(eraLevel: number): EraAgentBonuses {
  // Base bonuses that scale with era
  const baseLearning = 1.0 + (eraLevel * 0.05);           // +5% per era
  const baseDiscovery = eraLevel * 0.002;                 // +0.2% per era
  const baseEfficiency = Math.max(0.5, 1.0 - eraLevel * 0.02); // -2% cost per era, min 50%
  const baseReproBonus = Math.min(5, eraLevel * 0.5);     // +0.5 per era, max 5
  const baseKnowledge = Math.min(0.9, 0.1 + eraLevel * 0.1); // 10% + 10% per era, max 90%
  const baseCollabRange = Math.min(5, 2 + Math.floor(eraLevel / 2)); // 2 + 1 per 2 eras, max 5
  
  return {
    learningMultiplier: baseLearning,
    discoveryChance: baseDiscovery,
    energyEfficiency: baseEfficiency,
    reproductionBonus: baseReproBonus,
    knowledgeRetention: baseKnowledge,
    collaborationRange: baseCollabRange
  };
}

/**
 * Apply era-enhanced learning rate to Q-learning
 */
export function getEraEnhancedLearningRate(
  baseLearningRate: number,
  eraLevel: number
): number {
  const bonuses = getEraBonuses(eraLevel);
  return baseLearningRate * bonuses.learningMultiplier;
}

/**
 * Get era-enhanced discovery chance bonus
 */
export function getEraDiscoveryBonus(eraLevel: number): number {
  return getEraBonuses(eraLevel).discoveryChance;
}

/**
 * Get era-enhanced energy efficiency
 */
export function getEraEnergyEfficiency(eraLevel: number): number {
  return getEraBonuses(eraLevel).energyEfficiency;
}

/**
 * Get knowledge retention for offspring based on era
 * Higher eras = more knowledge passed to children
 */
export function getKnowledgeRetentionRate(
  eraLevel: number,
  parentSocialGene: number
): number {
  const eraBonus = getEraBonuses(eraLevel).knowledgeRetention;
  // Social gene influences how well knowledge is transmitted
  const socialBonus = parentSocialGene * 0.2;
  return Math.min(0.95, eraBonus + socialBonus);
}

/**
 * Calculate inherited inventions for offspring based on era
 */
export function calculateInheritedInventions(
  parentInventions: Agent['inventions'],
  eraLevel: number,
  parentSocialGene: number
): Agent['inventions'] {
  if (parentInventions.length === 0) return [];
  
  const retentionRate = getKnowledgeRetentionRate(eraLevel, parentSocialGene);
  const numToInherit = Math.floor(parentInventions.length * retentionRate);
  
  if (numToInherit === 0) return [];
  
  // Prefer inheriting "better" inventions (higher effect values)
  const sortedInventions = [...parentInventions].sort((a, b) => {
    const aValue = getInventionValue(a);
    const bValue = getInventionValue(b);
    return bValue - aValue;
  });
  
  return sortedInventions.slice(0, numToInherit);
}

/**
 * Helper to get a numeric value for an invention (for sorting)
 */
function getInventionValue(invention: Agent['inventions'][0]): number {
  switch (invention.effect.type) {
    case 'energy_efficiency':
      return invention.effect.multiplier * 10;
    case 'food_detection_range':
      return invention.effect.range * 5;
    case 'reproduction_boost':
      return invention.effect.bonus * 3;
    case 'defense':
      return invention.effect.protection * 4;
    case 'storage':
      return invention.effect.capacity * 2;
    default:
      return 1;
  }
}

/**
 * Check if the civilization should advance to the next era
 */
export function checkEraAdvancement(
  currentState: CivilizationEraState,
  physicsCount: number,
  mathCount: number,
  totalInventions: number,
  currentTick: number
): { shouldAdvance: boolean; nextEra: ScientificEra | null; log: string | null } {
  const { currentEra, metrics } = currentState;
  
  // Get requirements for next era
  let nextEraTemplate;
  const nextLevel = currentEra.level + 1;
  
  if (nextLevel < BASE_ERAS.length) {
    nextEraTemplate = BASE_ERAS[nextLevel];
  } else {
    nextEraTemplate = generateNextEra(nextLevel);
  }
  
  // Check if requirements are met
  const meetsDiscoveryReq = metrics.totalDiscoveries >= nextEraTemplate.requirements.minDiscoveries;
  const meetsPhysicsReq = physicsCount >= nextEraTemplate.requirements.minPhysicsConcepts;
  const meetsMathReq = mathCount >= nextEraTemplate.requirements.minMathConcepts;
  
  if (meetsDiscoveryReq && meetsPhysicsReq && meetsMathReq) {
    const nextEra: ScientificEra = {
      ...nextEraTemplate,
      startTick: currentTick,
      discoveries: [],
      physicsUnlocked: [],
      mathUnlocked: []
    };
    
    const log = `🏛️ CIVILIZATION ADVANCED TO ${nextEra.name.toUpperCase()}! (Era ${nextEra.level})`;
    
    return { shouldAdvance: true, nextEra, log };
  }
  
  return { shouldAdvance: false, nextEra: null, log: null };
}

/**
 * Check for newly achieved milestones
 */
export function checkMilestones(
  currentState: CivilizationEraState,
  physicsCount: number,
  mathCount: number,
  totalInventions: number,
  agentCount: number,
  collaborativeDiscoveries: number,
  currentTick: number
): EpochMilestone[] {
  const achievedIds = new Set(currentState.milestones.map(m => m.id));
  const newMilestones: EpochMilestone[] = [];
  
  // Check each potential milestone
  for (const milestone of EPOCH_MILESTONES) {
    if (achievedIds.has(milestone.id)) continue;
    
    let achieved = false;
    
    switch (milestone.id) {
      // Discovery milestones
      case 'first_physics':
        achieved = physicsCount >= 1;
        break;
      case 'first_math':
        achieved = mathCount >= 1;
        break;
      case 'physics_master':
        achieved = physicsCount >= 10;
        break;
      case 'math_master':
        achieved = mathCount >= 10;
        break;
      case 'unified_science':
        achieved = physicsCount >= 25 && mathCount >= 25;
        break;
        
      // Population milestones
      case 'first_village':
        achieved = agentCount >= 10;
        break;
      case 'growing_town':
        achieved = agentCount >= 25;
        break;
      case 'bustling_city':
        achieved = agentCount >= 50;
        break;
      case 'metropolis':
        achieved = agentCount >= 100;
        break;
        
      // Invention milestones
      case 'first_invention':
        achieved = totalInventions >= 1;
        break;
      case 'inventor_society':
        achieved = totalInventions >= 50;
        break;
      case 'innovation_age':
        achieved = totalInventions >= 200;
        break;
        
      // Collaboration milestones
      case 'first_collab':
        achieved = collaborativeDiscoveries >= 1;
        break;
      case 'research_team':
        achieved = collaborativeDiscoveries >= 5;
        break;
      case 'scientific_community':
        achieved = collaborativeDiscoveries >= 20;
        break;
        
      // Era advancement milestones
      case 'bronze_age_reached':
        achieved = currentState.currentEra.level >= 1;
        break;
      case 'iron_age_reached':
        achieved = currentState.currentEra.level >= 2;
        break;
      case 'classical_age_reached':
        achieved = currentState.currentEra.level >= 3;
        break;
      case 'renaissance_reached':
        achieved = currentState.currentEra.level >= 4;
        break;
      case 'industrial_age_reached':
        achieved = currentState.currentEra.level >= 5;
        break;
      case 'information_age_reached':
        achieved = currentState.currentEra.level >= 6;
        break;
      case 'quantum_age_reached':
        achieved = currentState.currentEra.level >= 7;
        break;
      case 'singularity_reached':
        achieved = currentState.currentEra.level >= 8;
        break;
    }
    
    if (achieved) {
      newMilestones.push({
        ...milestone,
        achievedAt: currentTick
      });
    }
  }
  
  return newMilestones;
}

/**
 * Update progression metrics
 */
export function updateProgressionMetrics(
  currentMetrics: ProgressionMetrics,
  newDiscoveryCount: number,
  currentTick: number
): ProgressionMetrics {
  const totalDiscoveries = currentMetrics.totalDiscoveries + newDiscoveryCount;
  const ticksSinceStart = Math.max(1, currentTick);
  
  // Calculate discovery rate
  const averageDiscoveryRate = totalDiscoveries / ticksSinceStart;
  
  // Update discoveries per era
  const discoveriesPerEra = { ...currentMetrics.discoveriesPerEra };
  const eraLevel = currentMetrics.currentEraLevel;
  discoveriesPerEra[eraLevel] = (discoveriesPerEra[eraLevel] || 0) + newDiscoveryCount;
  
  // Calculate scientific acceleration (are discoveries speeding up?)
  const recentRate = newDiscoveryCount; // Just this tick
  const scientificAcceleration = averageDiscoveryRate > 0 
    ? Math.min(3.0, Math.max(0.5, recentRate / averageDiscoveryRate))
    : 1.0;
  
  return {
    ...currentMetrics,
    totalDiscoveries,
    discoveriesPerEra,
    averageDiscoveryRate,
    ticksSinceLastEra: currentMetrics.ticksSinceLastEra + 1,
    scientificAcceleration
  };
}

/**
 * Get era summary for UI display
 */
export function getEraSummary(state: CivilizationEraState): {
  currentEraName: string;
  currentEraLevel: number;
  currentEraDescription: string;
  totalMilestones: number;
  recentMilestones: EpochMilestone[];
  progressToNextEra: number;          // 0-1 percentage
  nextEraRequirements: {
    discoveries: { current: number; required: number };
    physics: { current: number; required: number };
    math: { current: number; required: number };
  };
  bonuses: EraAgentBonuses;
} {
  const { currentEra, milestones, metrics } = state;
  
  // Get next era requirements
  let nextEraTemplate;
  const nextLevel = currentEra.level + 1;
  if (nextLevel < BASE_ERAS.length) {
    nextEraTemplate = BASE_ERAS[nextLevel];
  } else {
    nextEraTemplate = generateNextEra(nextLevel);
  }
  
  // Calculate progress to next era (average of all requirements)
  const discProgress = Math.min(1, metrics.totalDiscoveries / nextEraTemplate.requirements.minDiscoveries);
  // Note: We'll need to pass actual physics/math counts for accurate progress
  // For now, use discovery progress as approximation
  const progressToNextEra = discProgress;
  
  // Get recent milestones (last 3)
  const recentMilestones = [...milestones]
    .sort((a, b) => b.achievedAt - a.achievedAt)
    .slice(0, 3);
  
  const bonuses = getEraBonuses(currentEra.level);
  
  return {
    currentEraName: currentEra.name,
    currentEraLevel: currentEra.level,
    currentEraDescription: currentEra.description,
    totalMilestones: milestones.length,
    recentMilestones,
    progressToNextEra,
    nextEraRequirements: {
      discoveries: { 
        current: metrics.totalDiscoveries, 
        required: nextEraTemplate.requirements.minDiscoveries 
      },
      physics: { 
        current: 0, // Will be updated with actual count
        required: nextEraTemplate.requirements.minPhysicsConcepts 
      },
      math: { 
        current: 0, // Will be updated with actual count
        required: nextEraTemplate.requirements.minMathConcepts 
      }
    },
    bonuses
  };
}

/**
 * Get full era summary with actual physics/math counts
 */
export function getFullEraSummary(
  state: CivilizationEraState,
  physicsCount: number,
  mathCount: number
): ReturnType<typeof getEraSummary> {
  const summary = getEraSummary(state);
  
  // Update with actual counts
  summary.nextEraRequirements.physics.current = physicsCount;
  summary.nextEraRequirements.math.current = mathCount;
  
  // Recalculate progress with all metrics
  const discProgress = Math.min(1, summary.nextEraRequirements.discoveries.current / summary.nextEraRequirements.discoveries.required);
  const physProgress = Math.min(1, physicsCount / summary.nextEraRequirements.physics.required);
  const mathProgress = Math.min(1, mathCount / summary.nextEraRequirements.math.required);
  
  summary.progressToNextEra = (discProgress + physProgress + mathProgress) / 3;
  
  return summary;
}
