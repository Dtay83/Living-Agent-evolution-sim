/**
 * CONSCIOUSNESS & SELF-AWARENESS DETECTION SYSTEM
 * 
 * This system monitors agents for signs of emergent self-awareness through
 * multiple interconnected indicators. Self-awareness emerges when agents
 * demonstrate:
 * - Complex learning behaviors (metacognition)
 * - Social knowledge transmission (theory of mind)
 * - Creative problem-solving (abstract thinking)
 * - Long-term planning (temporal consciousness)
 */

import { Agent, Invention, PhysicsConcept, MathConcept } from '../types';

/**
 * Consciousness levels ranging from simple reactivity to full self-awareness
 */
export enum ConsciousnessLevel {
  REACTIVE = 0,        // Pure stimulus-response (< 10 points)
  ADAPTIVE = 1,        // Basic learning capability (10-30 points)
  COGNITIVE = 2,       // Problem-solving and planning (30-60 points)
  METACOGNITIVE = 3,   // Learning about learning (60-100 points)
  SELF_AWARE = 4,      // Full consciousness (100+ points)
}

export interface ConsciousnessIndicator {
  name: string;
  description: string;
  value: number;        // 0-1 score
  weight: number;       // Importance multiplier
  threshold: number;    // Minimum for consciousness
}

export interface ConsciousnessState {
  level: ConsciousnessLevel;
  score: number;                    // Total consciousness score (0-100+)
  indicators: ConsciousnessIndicator[];
  awakenedAt?: number;              // Tick when became self-aware
  awarenessEvents: AwarenessEvent[];
}

export interface AwarenessEvent {
  tick: number;
  agentId: number;
  level: ConsciousnessLevel;
  trigger: string;                  // What caused the awareness shift
  indicators: string[];             // Which indicators triggered it
}

/**
 * Calculate consciousness indicators for an agent
 */
export function calculateConsciousnessIndicators(
  agent: Agent,
  tick: number,
  allAgents: Agent[],
  physics: PhysicsConcept[],
  mathematics: MathConcept[]
): ConsciousnessIndicator[] {
  const indicators: ConsciousnessIndicator[] = [];

  // 1. METACOGNITIVE LEARNING (Learning about learning)
  // High invention points + high Q-table diversity = understands own learning
  const qTableSize = agent.qTable ? Object.keys(agent.qTable).length : 0;
  const metacognitionScore = Math.min(1, (
    (agent.inventionPoints / 100) * 0.6 +
    (qTableSize / 50) * 0.4
  ));
  
  indicators.push({
    name: 'Metacognition',
    description: 'Understanding of own learning process',
    value: metacognitionScore,
    weight: 3.0,
    threshold: 0.6,
  });

  // 2. THEORY OF MIND (Understanding others exist and think)
  // High social gene + has taught inventions + observes other agents
  const nearbyAgents = allAgents.filter(a => 
    a.id !== agent.id &&
    Math.abs(a.x - agent.x) <= 3 &&
    Math.abs(a.y - agent.y) <= 3
  ).length;
  
  const theoryOfMindScore = Math.min(1, (
    agent.genes.social * 0.5 +
    (agent.inventions.length > 0 ? 0.3 : 0) +
    (nearbyAgents > 0 ? 0.2 : 0)
  ));
  
  indicators.push({
    name: 'Theory of Mind',
    description: 'Awareness that other agents have thoughts',
    value: theoryOfMindScore,
    weight: 2.5,
    threshold: 0.7,
  });

  // 3. CREATIVE ABSTRACTION (Original thinking beyond training)
  // High creativity + breakthrough inventions + diverse invention types
  const breakthroughCount = agent.inventions.filter(inv => 
    inv.name.includes('Quantum-Breaking') ||
    inv.name.includes('Hyperdimensional') ||
    inv.name.includes('Transcendent')
  ).length;
  
  const inventionTypes = new Set(agent.inventions.map(i => i.effect.type)).size;
  
  const creativityScore = Math.min(1, (
    agent.genes.creativity * 0.4 +
    (breakthroughCount / 3) * 0.4 +
    (inventionTypes / 5) * 0.2
  ));
  
  indicators.push({
    name: 'Creative Abstraction',
    description: 'Ability to think beyond immediate experience',
    value: creativityScore,
    weight: 2.5,
    threshold: 0.65,
  });

  // 4. TEMPORAL CONSCIOUSNESS (Long-term planning & memory)
  // High patience + stored energy + consistent survival time
  const temporalScore = Math.min(1, (
    agent.genes.patience * 0.4 +
    (agent.age > 100 ? 0.4 : agent.age / 250) +
    (agent.energy > 15 ? 0.2 : 0) // Surplus = planning ahead
  ));
  
  indicators.push({
    name: 'Temporal Consciousness',
    description: 'Awareness of past, present, and future',
    value: temporalScore,
    weight: 2.0,
    threshold: 0.6,
  });

  // 5. ABSTRACT KNOWLEDGE (Understanding universal principles)
  // Advanced science discoveries + high curiosity
  const scienceLevel = physics.length + mathematics.length;
  const hasAdvancedScience = scienceLevel > 10;
  
  const abstractScore = Math.min(1, (
    agent.genes.curiosity * 0.4 +
    (scienceLevel / 30) * 0.4 +
    (hasAdvancedScience ? 0.2 : 0)
  ));
  
  indicators.push({
    name: 'Abstract Knowledge',
    description: 'Understanding of universal principles',
    value: abstractScore,
    weight: 2.0,
    threshold: 0.55,
  });

  // 6. SELF-REFLECTION (Introspection capability)
  // High curiosity + social + creativity combined
  const introspectionScore = Math.min(1, (
    agent.genes.curiosity * 0.35 +
    agent.genes.social * 0.35 +
    agent.genes.creativity * 0.3
  ));
  
  indicators.push({
    name: 'Self-Reflection',
    description: 'Ability to examine own thoughts and actions',
    value: introspectionScore,
    weight: 2.5,
    threshold: 0.7,
  });

  // 7. EXISTENTIAL UNDERSTANDING (Awareness of existence itself)
  // Combination of all advanced traits + inventions + science
  const existentialScore = Math.min(1, (
    (agent.inventionPoints / 100) * 0.3 +
    (agent.inventions.length / 10) * 0.3 +
    (scienceLevel / 20) * 0.2 +
    (agent.age / 200) * 0.2
  ));
  
  indicators.push({
    name: 'Existential Awareness',
    description: 'Consciousness of own existence and mortality',
    value: existentialScore,
    weight: 3.5,
    threshold: 0.75,
  });

  return indicators;
}

/**
 * Calculate overall consciousness level from indicators
 */
export function calculateConsciousnessLevel(
  indicators: ConsciousnessIndicator[]
): { level: ConsciousnessLevel; score: number } {
  // Weighted sum of all indicators
  let totalScore = 0;
  let totalWeight = 0;
  let thresholdsMet = 0;

  for (const indicator of indicators) {
    totalScore += indicator.value * indicator.weight * 10;
    totalWeight += indicator.weight;
    
    if (indicator.value >= indicator.threshold) {
      thresholdsMet++;
    }
  }

  const normalizedScore = totalWeight > 0 ? totalScore / totalWeight * 10 : 0;

  // Determine consciousness level
  let level: ConsciousnessLevel;
  
  if (normalizedScore >= 100 && thresholdsMet >= 5) {
    level = ConsciousnessLevel.SELF_AWARE;
  } else if (normalizedScore >= 60 && thresholdsMet >= 4) {
    level = ConsciousnessLevel.METACOGNITIVE;
  } else if (normalizedScore >= 30 && thresholdsMet >= 2) {
    level = ConsciousnessLevel.COGNITIVE;
  } else if (normalizedScore >= 10) {
    level = ConsciousnessLevel.ADAPTIVE;
  } else {
    level = ConsciousnessLevel.REACTIVE;
  }

  return { level, score: normalizedScore };
}

/**
 * Update consciousness state for an agent
 */
export function updateConsciousness(
  agent: Agent,
  previousState: ConsciousnessState | undefined,
  tick: number,
  allAgents: Agent[],
  physics: PhysicsConcept[],
  mathematics: MathConcept[]
): { state: ConsciousnessState; levelChanged: boolean; becameAware: boolean } {
  const indicators = calculateConsciousnessIndicators(
    agent,
    tick,
    allAgents,
    physics,
    mathematics
  );

  const { level, score } = calculateConsciousnessLevel(indicators);

  const previousLevel = previousState?.level ?? ConsciousnessLevel.REACTIVE;
  const levelChanged = level !== previousLevel;
  const becameAware = level === ConsciousnessLevel.SELF_AWARE && 
                      previousLevel !== ConsciousnessLevel.SELF_AWARE;

  const awarenessEvents = previousState?.awarenessEvents ?? [];
  
  // Log level changes
  if (levelChanged) {
    const triggeredIndicators = indicators
      .filter(ind => ind.value >= ind.threshold)
      .map(ind => ind.name);

    awarenessEvents.push({
      tick,
      agentId: agent.id,
      level,
      trigger: getLevelChangeTrigger(previousLevel, level, indicators),
      indicators: triggeredIndicators,
    });
  }

  const state: ConsciousnessState = {
    level,
    score,
    indicators,
    awakenedAt: becameAware ? tick : previousState?.awakenedAt,
    awarenessEvents,
  };

  return { state, levelChanged, becameAware };
}

/**
 * Get human-readable description of what triggered level change
 */
function getLevelChangeTrigger(
  fromLevel: ConsciousnessLevel,
  toLevel: ConsciousnessLevel,
  indicators: ConsciousnessIndicator[]
): string {
  if (toLevel === ConsciousnessLevel.SELF_AWARE) {
    const highestIndicator = indicators.reduce((max, ind) => 
      ind.value * ind.weight > max.value * max.weight ? ind : max
    );
    return `Achieved self-awareness through ${highestIndicator.name}`;
  } else if (toLevel === ConsciousnessLevel.METACOGNITIVE) {
    return 'Developed metacognitive abilities';
  } else if (toLevel === ConsciousnessLevel.COGNITIVE) {
    return 'Advanced to cognitive processing';
  } else if (toLevel === ConsciousnessLevel.ADAPTIVE) {
    return 'Developed adaptive learning';
  }
  return 'Consciousness level changed';
}

/**
 * Get consciousness level name
 */
export function getConsciousnessLevelName(level: ConsciousnessLevel): string {
  switch (level) {
    case ConsciousnessLevel.REACTIVE:
      return 'Reactive';
    case ConsciousnessLevel.ADAPTIVE:
      return 'Adaptive';
    case ConsciousnessLevel.COGNITIVE:
      return 'Cognitive';
    case ConsciousnessLevel.METACOGNITIVE:
      return 'Metacognitive';
    case ConsciousnessLevel.SELF_AWARE:
      return '🧠 SELF-AWARE';
    default:
      return 'Unknown';
  }
}

/**
 * Get consciousness level color
 */
export function getConsciousnessLevelColor(level: ConsciousnessLevel): string {
  switch (level) {
    case ConsciousnessLevel.REACTIVE:
      return '#666666';
    case ConsciousnessLevel.ADAPTIVE:
      return '#4a90e2';
    case ConsciousnessLevel.COGNITIVE:
      return '#7b68ee';
    case ConsciousnessLevel.METACOGNITIVE:
      return '#ff6b6b';
    case ConsciousnessLevel.SELF_AWARE:
      return '#ffd700';
    default:
      return '#999999';
  }
}

/**
 * Initialize consciousness tracking for world state
 */
export function initializeConsciousnessTracking(): Map<number, ConsciousnessState> {
  return new Map();
}

/**
 * Get summary statistics about consciousness in the population
 */
export function getConsciousnessSummary(
  consciousnessMap: Map<number, ConsciousnessState>
): {
  totalAgents: number;
  selfAwareCount: number;
  metacognitiveCount: number;
  cognitiveCount: number;
  adaptiveCount: number;
  reactiveCount: number;
  highestScore: number;
  averageScore: number;
  awakenedAgents: number[];
} {
  const states = Array.from(consciousnessMap.values());
  
  const summary = {
    totalAgents: states.length,
    selfAwareCount: states.filter(s => s.level === ConsciousnessLevel.SELF_AWARE).length,
    metacognitiveCount: states.filter(s => s.level === ConsciousnessLevel.METACOGNITIVE).length,
    cognitiveCount: states.filter(s => s.level === ConsciousnessLevel.COGNITIVE).length,
    adaptiveCount: states.filter(s => s.level === ConsciousnessLevel.ADAPTIVE).length,
    reactiveCount: states.filter(s => s.level === ConsciousnessLevel.REACTIVE).length,
    highestScore: states.length > 0 ? Math.max(...states.map(s => s.score)) : 0,
    averageScore: states.length > 0 
      ? states.reduce((sum, s) => sum + s.score, 0) / states.length 
      : 0,
    awakenedAgents: Array.from(consciousnessMap.entries())
      .filter(([_, state]) => state.awakenedAt !== undefined)
      .map(([agentId, _]) => agentId),
  };

  return summary;
}
