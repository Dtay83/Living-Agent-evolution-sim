/**
 * MATHEMATICS INTEGRATION SYSTEM
 * 
 * Integrates mathematical concepts with agent behavior:
 * - Resource optimization using math knowledge
 * - Pattern recognition for food prediction
 * - Statistical learning improvements to Q-learning
 * - Geometric awareness for pathfinding
 */

import type { Agent } from '../types';
import type { MathConcept } from '../science-system/mathematics';
import { MATH_CONCEPTS, calculateMathBonuses, generateAdvancedMathConcept, getAllMathConcepts } from '../science-system/mathematics';

/**
 * Mathematics state tracked for the civilization
 */
export interface CivilizationMath {
  unlockedConcepts: MathConcept[];
  totalDiscoveries: number;
  lastDiscoveryTick: number;
  foodPredictionAccuracy: number; // Improves with pattern recognition
  resourceOptimizationLevel: number; // Improves with optimization power
}

/**
 * Initialize civilization mathematics state
 */
export function initializeCivilizationMath(): CivilizationMath {
  return {
    unlockedConcepts: [],
    totalDiscoveries: 0,
    lastDiscoveryTick: 0,
    foodPredictionAccuracy: 0,
    resourceOptimizationLevel: 0
  };
}

/**
 * Get mathematics bonuses for agent behavior
 * NO CAPS - bonuses grow infinitely with discoveries
 */
export function getAgentMathBonuses(unlockedMath: MathConcept[]): {
  decisionQuality: number;
  explorationBonus: number;
  patternRecognition: number;
  optimizationPower: number;
  computationalSpeed: number;
  abstractionLevel: number;
} {
  return calculateMathBonuses(unlockedMath);
}

/**
 * Apply mathematics to Q-learning updates
 * Better math = better learning rates and decision quality
 */
export function getMathEnhancedLearningRate(
  baseLearningRate: number,
  unlockedMath: MathConcept[]
): number {
  const bonuses = calculateMathBonuses(unlockedMath);
  // Decision quality improves learning rate
  // Optimization power provides additional boost
  return baseLearningRate * Math.sqrt(bonuses.decisionQuality) * Math.pow(bonuses.optimizationPower, 0.3);
}

/**
 * Apply mathematics to Q-value calculations
 * Pattern recognition helps identify better states
 */
export function getMathEnhancedQValue(
  baseQValue: number,
  unlockedMath: MathConcept[]
): number {
  const bonuses = calculateMathBonuses(unlockedMath);
  // Pattern recognition helps distinguish state values
  return baseQValue * Math.pow(bonuses.patternRecognition, 0.2);
}

/**
 * Calculate food prediction bonus based on math knowledge
 * Agents with pattern recognition can better predict food spawns
 */
export function getFoodPredictionBonus(
  agent: Agent,
  unlockedMath: MathConcept[],
  recentFoodPositions: { x: number; y: number; tick: number }[]
): { x: number; y: number; confidence: number } | null {
  if (unlockedMath.length === 0 || recentFoodPositions.length < 3) {
    return null;
  }

  const bonuses = calculateMathBonuses(unlockedMath);
  
  // Need sufficient pattern recognition to predict
  if (bonuses.patternRecognition < 1.1) {
    return null;
  }

  // Simple pattern: find center of recent food positions
  const avgX = recentFoodPositions.reduce((sum, p) => sum + p.x, 0) / recentFoodPositions.length;
  const avgY = recentFoodPositions.reduce((sum, p) => sum + p.y, 0) / recentFoodPositions.length;

  // Confidence scales with pattern recognition and agent curiosity
  const confidence = Math.min(0.8, 
    (bonuses.patternRecognition - 1) * 0.5 + 
    agent.genes.curiosity * 0.3
  );

  return {
    x: Math.round(avgX),
    y: Math.round(avgY),
    confidence
  };
}

/**
 * Calculate geometric pathfinding bonus
 * Agents with geometry knowledge can find shorter paths
 */
export function getGeometricPathBonus(
  agentX: number,
  agentY: number,
  targetX: number,
  targetY: number,
  unlockedMath: MathConcept[]
): { preferredDirection: 'up' | 'down' | 'left' | 'right' | null; efficiency: number } {
  // Check for geometry concepts
  const hasBasicShapes = unlockedMath.some(c => c.id === 'basic_shapes');
  const hasPythagorean = unlockedMath.some(c => c.id === 'pythagorean_theorem');
  const hasTrigonometry = unlockedMath.some(c => c.id === 'trigonometry');

  if (!hasBasicShapes) {
    return { preferredDirection: null, efficiency: 1.0 };
  }

  const dx = targetX - agentX;
  const dy = targetY - agentY;

  // Determine preferred direction based on geometry knowledge
  let preferredDirection: 'up' | 'down' | 'left' | 'right' | null = null;
  
  if (Math.abs(dx) > Math.abs(dy)) {
    preferredDirection = dx > 0 ? 'right' : 'left';
  } else if (dy !== 0) {
    preferredDirection = dy > 0 ? 'down' : 'up';
  }

  // Calculate efficiency bonus
  let efficiency = 1.05; // Basic shapes bonus
  if (hasPythagorean) efficiency += 0.08;
  if (hasTrigonometry) efficiency += 0.12;

  return { preferredDirection, efficiency };
}

/**
 * Calculate resource optimization bonus
 * Better math = more efficient energy usage
 */
export function getResourceOptimizationBonus(unlockedMath: MathConcept[]): number {
  const bonuses = calculateMathBonuses(unlockedMath);
  
  // Optimization power directly improves resource efficiency
  // Returns a multiplier < 1 for energy cost reduction
  const reduction = (bonuses.optimizationPower - 1) * 0.3;
  return Math.max(0.7, 1 - reduction); // Cap at 30% reduction
}

/**
 * Check if an agent discovers a math concept this tick
 * UNLIMITED EVOLUTION - procedurally generates new concepts when base set exhausted
 */
export function checkMathDiscovery(
  agent: Agent,
  nearbyAgents: Agent[],
  unlockedMath: MathConcept[],
  tick: number,
  generatedMathLevel: number = 0
): { concept: MathConcept | null; log: string | null; nextGeneratedLevel?: number } {
  // Need cognitive surplus (energy > 8) to think about math
  if (agent.energy < 8) {
    return { concept: null, log: null };
  }

  // Find discoverable concepts (prerequisites met, not already unlocked)
  const unlockedIds = new Set(unlockedMath.map(m => m.id));
  let discoverableConcepts = MATH_CONCEPTS.filter(concept => {
    if (unlockedIds.has(concept.id)) return false;
    return concept.prerequisiteIds.every(id => unlockedIds.has(id) || id === '');
  });

  // Include concepts with empty prerequisites for initial discoveries
  let availableConcepts = discoverableConcepts.length > 0
    ? discoverableConcepts
    : MATH_CONCEPTS.filter(c => c.prerequisiteIds.length === 0 && !unlockedIds.has(c.id));

  // UNLIMITED EVOLUTION: If all base concepts discovered, generate procedural ones
  let nextGeneratedLevel = generatedMathLevel;
  if (availableConcepts.length === 0) {
    // Check if last concept in chain is unlocked to allow next procedural
    const lastConceptId = generatedMathLevel > 0 
      ? `advanced_math_${generatedMathLevel - 1}` 
      : 'hypercomputation';
    
    if (unlockedIds.has(lastConceptId)) {
      // Generate next procedural concept
      const proceduralConcept = generateAdvancedMathConcept(generatedMathLevel);
      availableConcepts = [proceduralConcept];
      nextGeneratedLevel = generatedMathLevel + 1;
    } else {
      return { concept: null, log: null };
    }
  }

  if (availableConcepts.length === 0) {
    return { concept: null, log: null };
  }

  // Calculate discovery chance - math discovery favors patience and curiosity
  const baseChance = agent.genes.curiosity * agent.genes.patience * 0.025;
  
  // Collaboration bonus from nearby agents
  const collaborationBonus = nearbyAgents.length > 0
    ? nearbyAgents.reduce((sum, a) => sum + a.genes.social * a.genes.patience, 0) * 0.005
    : 0;
  
  // Invention points provide insight bonus
  const insightBonus = Math.min(0.02, agent.inventionPoints * 0.0008);

  const totalChance = baseChance + collaborationBonus + insightBonus;

  if (Math.random() > totalChance) {
    return { concept: null, log: null };
  }

  // Select concept to discover (prefer lower complexity)
  const sortedConcepts = [...availableConcepts].sort((a, b) => a.complexity - b.complexity);
  
  // Weight towards simpler concepts
  const indexWeight = Math.random() * Math.random();
  const selectedIndex = Math.floor(indexWeight * sortedConcepts.length);
  const concept = sortedConcepts[selectedIndex];

  // Mark as discovered
  const discoveredConcept: MathConcept = {
    ...concept,
    discoveredAt: tick,
    discoveredBy: agent.id
  };
  // Generate log message
  const collabMsg = nearbyAgents.length > 0
    ? ` (studied with ${nearbyAgents.length} nearby agent${nearbyAgents.length > 1 ? 's' : ''}!)`
    : '';
  const proceduralMsg = concept.id.startsWith('advanced_math_') ? ' 🌟 TRANSCENDENT DISCOVERY!' : '';
  const log = `📐 Agent ${agent.id} discovered math concept: ${concept.name}!${collabMsg}${proceduralMsg}`;

  return { concept: discoveredConcept, log, nextGeneratedLevel };
}

/**
 * Get exploration bonus from math knowledge
 * Statistical thinking encourages better exploration
 */
export function getMathExplorationBonus(unlockedMath: MathConcept[]): number {
  const bonuses = calculateMathBonuses(unlockedMath);
  return bonuses.explorationBonus;
}

/**
 * Summarize math state for UI display
 * Shows all bonuses including new computational and abstraction bonuses
 */
export function getMathSummary(unlockedMath: MathConcept[]): {
  totalConcepts: number;
  byCategory: Record<string, number>;
  topBonuses: { name: string; value: string }[];
  proceduralLevel: number;
} {
  const byCategory: Record<string, number> = {};
  let proceduralLevel = 0;

  for (const concept of unlockedMath) {
    byCategory[concept.category] = (byCategory[concept.category] || 0) + 1;
    // Track procedural concepts
    if (concept.id.startsWith('advanced_math_')) {
      const level = parseInt(concept.id.replace('advanced_math_', ''));
      proceduralLevel = Math.max(proceduralLevel, level + 1);
    }
  }

  const bonuses = calculateMathBonuses(unlockedMath);
  const topBonuses: { name: string; value: string }[] = [];

  if (bonuses.decisionQuality > 1) {
    topBonuses.push({
      name: 'Decision Quality',
      value: `${Math.round((bonuses.decisionQuality - 1) * 100)}% better`
    });
  }
  if (bonuses.patternRecognition > 1) {
    topBonuses.push({
      name: 'Pattern Recognition',
      value: `${Math.round((bonuses.patternRecognition - 1) * 100)}% improved`
    });
  }
  if (bonuses.optimizationPower > 1) {
    topBonuses.push({
      name: 'Optimization',
      value: `${Math.round((bonuses.optimizationPower - 1) * 100)}% more efficient`
    });
  }
  if (bonuses.explorationBonus > 0) {
    topBonuses.push({
      name: 'Exploration Bonus',
      value: `+${(bonuses.explorationBonus * 100).toFixed(0)}%`
    });
  }
  // NEW BONUSES - NO CAPS
  if (bonuses.computationalSpeed > 1) {
    topBonuses.push({
      name: 'Computational Speed',
      value: `${Math.round((bonuses.computationalSpeed - 1) * 100)}% faster`
    });
  }
  if (bonuses.abstractionLevel > 1) {
    topBonuses.push({
      name: 'Abstraction Level',
      value: `${Math.round((bonuses.abstractionLevel - 1) * 100)}% higher`
    });
  }

  return {
    totalConcepts: unlockedMath.length,
    byCategory,
    topBonuses,
    proceduralLevel
  };
}
