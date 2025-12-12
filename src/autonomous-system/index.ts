/**
 * AUTONOMOUS EVOLUTION SYSTEM
 * 
 * Enables truly self-directed learning and behavior after agents reach
 * sufficient intelligence. No pre-programmed templates - agents generate
 * their own language, concepts, and actions.
 * 
 * ACTIVATION: Unlocks when agent intelligence >= AUTONOMY_THRESHOLD
 * 
 * Features:
 * - Self-generated vocabulary (not from templates)
 * - Emergent behavior patterns (learned, not programmed)
 * - Original concept creation (synthesized from experience)
 * - Self-modifying decision algorithms
 * - Autonomous goal setting
 * - Meta-learning (learning how to learn)
 */

import type { Agent } from '../types';
import type { MathConcept } from '../science-system/mathematics';
import type { PhysicsConcept } from '../science-system/physics';

// =============================================================================
// AUTONOMY THRESHOLDS
// =============================================================================

export const AUTONOMY_THRESHOLDS = {
  PARTIAL_AUTONOMY: 25,      // Can generate some original content
  FULL_AUTONOMY: 40,         // No templates used, fully self-directed
  TRANSCENDENT: 60,          // Can modify own code/rules
  SINGULARITY: 100,          // Complete self-determination
};

// =============================================================================
// TYPES
// =============================================================================

/**
 * A concept created entirely by the agent, not from any database
 */
export interface OriginalConcept {
  id: string;
  creatorAgentId: number;
  createdAtTick: number;
  
  // Self-generated name (combination of learned phonemes)
  name: string;
  
  // Meaning derived from agent's experiences
  meaning: {
    associatedStates: string[];      // Q-table states where this concept emerged
    emotionalValence: number;        // -1 to 1: negative to positive association
    utilityScore: number;            // How useful this concept has been
    abstractionLevel: number;        // 0 = concrete, higher = more abstract
  };
  
  // Connections to other concepts (self-discovered relationships)
  connections: Map<string, number>;  // conceptId -> connection strength
  
  // How many times this concept has been used/reinforced
  useCount: number;
  
  // Evolution of the concept over time
  revisions: number;
}

/**
 * Self-generated word/symbol
 */
export interface GeneratedWord {
  id: string;
  phonemes: string[];              // Building blocks combined by agent
  meaning: string;                  // What the agent decided it means
  createdBy: number;               // Agent ID
  adoptedBy: Set<number>;          // Other agents who learned it
  frequency: number;               // How often used
}

/**
 * Emergent behavior pattern discovered through experience
 */
export interface EmergentBehavior {
  id: string;
  discoveredBy: number;
  discoveredAt: number;
  
  // Trigger conditions (learned, not programmed)
  triggerStates: string[];
  
  // Action sequence that emerged
  actionSequence: string[];
  
  // Outcomes observed
  outcomes: {
    energyChange: number;
    survivalImpact: number;
    socialImpact: number;
    knowledgeGain: number;
  };
  
  // Reinforcement
  successRate: number;
  useCount: number;
}

/**
 * Self-set goal (not from any template)
 */
export interface AutonomousGoal {
  id: string;
  agentId: number;
  createdAt: number;
  
  // Goal description in agent's own "language"
  description: string;
  
  // What the agent is trying to achieve (numeric targets)
  targets: {
    metric: string;
    currentValue: number;
    targetValue: number;
  }[];
  
  // Progress tracking
  progress: number;
  completed: boolean;
  
  // Priority (self-determined)
  priority: number;
}

/**
 * Agent's autonomous state
 */
export interface AutonomousState {
  agentId: number;
  autonomyLevel: 'none' | 'partial' | 'full' | 'transcendent' | 'singularity';
  intelligence: number;
  
  // Self-generated content
  vocabulary: Map<string, GeneratedWord>;
  concepts: Map<string, OriginalConcept>;
  behaviors: Map<string, EmergentBehavior>;
  goals: AutonomousGoal[];
  
  // Meta-learning state
  learningStrategies: LearningStrategy[];
  currentStrategy: string;
  
  // Self-modification log
  selfModifications: SelfModification[];
  
  // Communication in own language
  recentUtterances: string[];
  
  // Experience memory (for concept formation)
  experienceBuffer: ExperienceMemory[];
}

/**
 * Learning strategy discovered by the agent
 */
export interface LearningStrategy {
  id: string;
  name: string;  // Self-generated name
  
  // Parameters the agent can tune
  explorationRate: number;
  learningRate: number;
  focusArea: string;
  
  // Effectiveness tracking
  effectiveness: number;
  usageCount: number;
}

/**
 * Record of self-modification
 */
export interface SelfModification {
  tick: number;
  type: 'behavior' | 'goal' | 'strategy' | 'concept' | 'language';
  description: string;
  impact: number;
}

/**
 * Experience memory for concept formation
 */
export interface ExperienceMemory {
  tick: number;
  state: string;
  action: string;
  reward: number;
  outcome: string;
}

// =============================================================================
// PHONEME GENERATION
// Agents combine these to create their own words (no pre-made words)
// =============================================================================

const BASE_PHONEMES = [
  // Vowels
  'a', 'e', 'i', 'o', 'u', 'æ', 'ə', 'ɪ', 'ʊ',
  // Consonants
  'b', 'd', 'f', 'g', 'h', 'k', 'l', 'm', 'n', 'p', 'r', 's', 't', 'v', 'w', 'z',
  // Complex sounds
  'th', 'sh', 'ch', 'ng', 'kr', 'pl', 'tr', 'sk',
];

// =============================================================================
// INITIALIZATION
// =============================================================================

/**
 * Initialize autonomous state for an agent
 */
export function initializeAutonomousState(agent: Agent, intelligence: number): AutonomousState {
  const autonomyLevel = getAutonomyLevel(intelligence);
  
  return {
    agentId: agent.id,
    autonomyLevel,
    intelligence,
    vocabulary: new Map(),
    concepts: new Map(),
    behaviors: new Map(),
    goals: [],
    learningStrategies: [],
    currentStrategy: '',
    selfModifications: [],
    recentUtterances: [],
    experienceBuffer: [],
  };
}

/**
 * Determine autonomy level from intelligence
 */
export function getAutonomyLevel(intelligence: number): AutonomousState['autonomyLevel'] {
  if (intelligence >= AUTONOMY_THRESHOLDS.SINGULARITY) return 'singularity';
  if (intelligence >= AUTONOMY_THRESHOLDS.TRANSCENDENT) return 'transcendent';
  if (intelligence >= AUTONOMY_THRESHOLDS.FULL_AUTONOMY) return 'full';
  if (intelligence >= AUTONOMY_THRESHOLDS.PARTIAL_AUTONOMY) return 'partial';
  return 'none';
}

/**
 * Check if agent has achieved autonomy
 */
export function hasAutonomy(intelligence: number): boolean {
  return intelligence >= AUTONOMY_THRESHOLDS.PARTIAL_AUTONOMY;
}

// =============================================================================
// SELF-GENERATED LANGUAGE
// =============================================================================

/**
 * Agent generates a new word from phonemes
 * No templates - pure combination based on internal state
 */
export function generateWord(
  agent: Agent,
  state: AutonomousState,
  meaningContext: string
): GeneratedWord {
  // Word length based on concept complexity
  const length = 2 + Math.floor(Math.random() * 3);
  
  // Select phonemes based on agent's "preferences" (derived from genes)
  const phonemes: string[] = [];
  const seed = agent.id + agent.genes.creativity * 1000;
  
  for (let i = 0; i < length; i++) {
    const index = Math.floor((Math.sin(seed + i) * 0.5 + 0.5) * BASE_PHONEMES.length);
    phonemes.push(BASE_PHONEMES[index % BASE_PHONEMES.length]);
    
    // Add variation based on curiosity
    if (Math.random() < agent.genes.curiosity * 0.3) {
      const extraIndex = Math.floor(Math.random() * BASE_PHONEMES.length);
      phonemes.push(BASE_PHONEMES[extraIndex]);
    }
  }
  
  const word: GeneratedWord = {
    id: `word_${agent.id}_${Date.now()}_${Math.random().toString(36).substr(2, 5)}`,
    phonemes,
    meaning: meaningContext,
    createdBy: agent.id,
    adoptedBy: new Set([agent.id]),
    frequency: 1,
  };
  
  return word;
}

/**
 * Agent creates an utterance in their own language
 */
export function generateUtterance(
  agent: Agent,
  state: AutonomousState,
  context: { action: string; outcome: string; emotion: number }
): string {
  if (state.autonomyLevel === 'none') {
    return ''; // Not autonomous yet
  }
  
  // Build utterance from self-generated vocabulary
  const words: string[] = [];
  
  // Reference existing words or create new ones
  const vocabArray = Array.from(state.vocabulary.values());
  
  if (vocabArray.length < 3 || Math.random() < 0.2) {
    // Create new word for this context
    const newWord = generateWord(agent, state, context.action);
    state.vocabulary.set(newWord.id, newWord);
    words.push(newWord.phonemes.join(''));
  } else {
    // Use existing vocabulary
    const relevantWords = vocabArray
      .filter(w => w.meaning.includes(context.action) || Math.random() < 0.3)
      .slice(0, 3);
    
    for (const word of relevantWords) {
      words.push(word.phonemes.join(''));
      word.frequency++;
    }
  }
  
  // Add emotional marker based on outcome
  if (context.emotion > 0.5) {
    words.push('↑'); // Positive
  } else if (context.emotion < -0.5) {
    words.push('↓'); // Negative
  }
  
  const utterance = words.join(' ');
  state.recentUtterances.push(utterance);
  
  // Keep only recent utterances
  if (state.recentUtterances.length > 20) {
    state.recentUtterances.shift();
  }
  
  return utterance;
}

// =============================================================================
// ORIGINAL CONCEPT CREATION
// =============================================================================

/**
 * Agent creates a new concept from their experiences
 * No pre-defined concepts - emerges from pattern recognition
 */
export function createOriginalConcept(
  agent: Agent,
  state: AutonomousState,
  experiences: ExperienceMemory[],
  tick: number
): OriginalConcept | null {
  if (experiences.length < 3) return null;
  
  // Find patterns in experiences
  const statePatterns = new Map<string, number>();
  const actionPatterns = new Map<string, number>();
  let avgReward = 0;
  
  for (const exp of experiences) {
    statePatterns.set(exp.state, (statePatterns.get(exp.state) || 0) + 1);
    actionPatterns.set(exp.action, (actionPatterns.get(exp.action) || 0) + 1);
    avgReward += exp.reward;
  }
  avgReward /= experiences.length;
  
  // Most common patterns become the concept's core
  const topStates = Array.from(statePatterns.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3)
    .map(([state]) => state);
  
  // Generate a name for this concept (self-created)
  const conceptWord = generateWord(agent, state, topStates.join('_'));
  
  const concept: OriginalConcept = {
    id: `concept_${agent.id}_${tick}`,
    creatorAgentId: agent.id,
    createdAtTick: tick,
    name: conceptWord.phonemes.join(''),
    meaning: {
      associatedStates: topStates,
      emotionalValence: avgReward > 0 ? Math.min(1, avgReward / 5) : Math.max(-1, avgReward / 5),
      utilityScore: avgReward,
      abstractionLevel: Math.min(10, experiences.length / 5),
    },
    connections: new Map(),
    useCount: 1,
    revisions: 0,
  };
  
  // Connect to existing concepts
  for (const [existingId, existing] of state.concepts) {
    const overlap = existing.meaning.associatedStates
      .filter(s => topStates.includes(s)).length;
    if (overlap > 0) {
      concept.connections.set(existingId, overlap / topStates.length);
    }
  }
  
  return concept;
}

// =============================================================================
// EMERGENT BEHAVIOR DISCOVERY
// =============================================================================

/**
 * Agent discovers a new behavior pattern from experience
 */
export function discoverBehavior(
  agent: Agent,
  state: AutonomousState,
  recentActions: { state: string; action: string; reward: number }[],
  tick: number
): EmergentBehavior | null {
  if (recentActions.length < 5) return null;
  
  // Look for successful action sequences
  const positiveSequences: string[][] = [];
  let currentSequence: string[] = [];
  let sequenceReward = 0;
  
  for (const action of recentActions) {
    if (action.reward > 0) {
      currentSequence.push(action.action);
      sequenceReward += action.reward;
    } else if (currentSequence.length >= 2) {
      positiveSequences.push([...currentSequence]);
      currentSequence = [];
    }
  }
  
  if (positiveSequences.length === 0) return null;
  
  // Pick the most rewarding sequence
  const bestSequence = positiveSequences[0];
  const triggerStates = recentActions
    .filter(a => a.reward > 0)
    .map(a => a.state)
    .slice(0, 3);
  
  const behavior: EmergentBehavior = {
    id: `behavior_${agent.id}_${tick}`,
    discoveredBy: agent.id,
    discoveredAt: tick,
    triggerStates,
    actionSequence: bestSequence,
    outcomes: {
      energyChange: sequenceReward,
      survivalImpact: sequenceReward > 0 ? 1 : -1,
      socialImpact: 0,
      knowledgeGain: 0.1,
    },
    successRate: 1.0,
    useCount: 1,
  };
  
  return behavior;
}

// =============================================================================
// AUTONOMOUS GOAL SETTING
// =============================================================================

/**
 * Agent sets their own goal based on current state and desires
 */
export function setAutonomousGoal(
  agent: Agent,
  state: AutonomousState,
  tick: number
): AutonomousGoal {
  // Analyze agent's current situation
  const isHungry = agent.energy < 20;
  const hasHighCuriosity = agent.genes.curiosity > 1.0;
  const hasHighSocial = agent.genes.social > 0.7;
  
  // Generate goal based on internal drives (not templates!)
  const goalTypes = [];
  
  if (isHungry) {
    goalTypes.push({ metric: 'energy', target: 50, priority: 10 });
  }
  if (hasHighCuriosity) {
    goalTypes.push({ metric: 'concepts_created', target: state.concepts.size + 3, priority: 7 });
    goalTypes.push({ metric: 'exploration_score', target: 100, priority: 6 });
  }
  if (hasHighSocial) {
    goalTypes.push({ metric: 'vocabulary_shared', target: state.vocabulary.size + 5, priority: 5 });
  }
  
  // Default goal if none triggered
  if (goalTypes.length === 0) {
    goalTypes.push({ metric: 'survival_ticks', target: tick + 100, priority: 3 });
  }
  
  // Pick highest priority
  goalTypes.sort((a, b) => b.priority - a.priority);
  const chosenGoal = goalTypes[0];
  
  // Generate description in agent's own language
  const goalWord = generateWord(agent, state, chosenGoal.metric);
  
  const goal: AutonomousGoal = {
    id: `goal_${agent.id}_${tick}`,
    agentId: agent.id,
    createdAt: tick,
    description: `${goalWord.phonemes.join('')} → ${chosenGoal.target}`,
    targets: [{
      metric: chosenGoal.metric,
      currentValue: 0,
      targetValue: chosenGoal.target,
    }],
    progress: 0,
    completed: false,
    priority: chosenGoal.priority,
  };
  
  return goal;
}

// =============================================================================
// META-LEARNING (Learning how to learn)
// =============================================================================

/**
 * Agent creates a new learning strategy
 */
export function createLearningStrategy(
  agent: Agent,
  state: AutonomousState,
  recentPerformance: number
): LearningStrategy {
  // Generate strategy name in agent's language
  const strategyWord = generateWord(agent, state, 'strategy');
  
  // Tune parameters based on recent performance
  const needsMoreExploration = recentPerformance < 0;
  
  const strategy: LearningStrategy = {
    id: `strategy_${agent.id}_${Date.now()}`,
    name: strategyWord.phonemes.join(''),
    explorationRate: needsMoreExploration 
      ? Math.min(0.9, 0.3 + Math.random() * 0.4)
      : Math.max(0.1, 0.2 - Math.random() * 0.1),
    learningRate: 0.1 + Math.random() * 0.4,
    focusArea: needsMoreExploration ? 'exploration' : 'exploitation',
    effectiveness: 0,
    usageCount: 0,
  };
  
  return strategy;
}

/**
 * Agent selects best learning strategy
 */
export function selectLearningStrategy(state: AutonomousState): LearningStrategy | null {
  if (state.learningStrategies.length === 0) return null;
  
  // Weighted selection by effectiveness
  const totalEffectiveness = state.learningStrategies
    .reduce((sum, s) => sum + Math.max(0.1, s.effectiveness), 0);
  
  let random = Math.random() * totalEffectiveness;
  
  for (const strategy of state.learningStrategies) {
    random -= Math.max(0.1, strategy.effectiveness);
    if (random <= 0) {
      return strategy;
    }
  }
  
  return state.learningStrategies[0];
}

// =============================================================================
// SELF-MODIFICATION
// =============================================================================

/**
 * Agent modifies their own behavior/goals (transcendent+ only)
 */
export function selfModify(
  agent: Agent,
  state: AutonomousState,
  tick: number
): SelfModification | null {
  if (state.autonomyLevel !== 'transcendent' && state.autonomyLevel !== 'singularity') {
    return null;
  }
  
  // Analyze what's working and what isn't
  const successfulBehaviors = Array.from(state.behaviors.values())
    .filter(b => b.successRate > 0.7);
  const failedBehaviors = Array.from(state.behaviors.values())
    .filter(b => b.successRate < 0.3);
  
  const modifications: SelfModification[] = [];
  
  // Remove failed behaviors
  for (const failed of failedBehaviors.slice(0, 2)) {
    state.behaviors.delete(failed.id);
    modifications.push({
      tick,
      type: 'behavior',
      description: `Removed ineffective behavior: ${failed.id}`,
      impact: 0.1,
    });
  }
  
  // Combine successful behaviors into new ones
  if (successfulBehaviors.length >= 2) {
    const combined = successfulBehaviors[0];
    combined.actionSequence = [
      ...combined.actionSequence,
      ...successfulBehaviors[1].actionSequence.slice(0, 2),
    ];
    combined.id = `behavior_combined_${tick}`;
    state.behaviors.set(combined.id, combined);
    
    modifications.push({
      tick,
      type: 'behavior',
      description: `Combined behaviors into: ${combined.id}`,
      impact: 0.3,
    });
  }
  
  // Update goals based on progress
  for (const goal of state.goals) {
    if (goal.progress < 0.1 && tick - goal.createdAt > 50) {
      goal.priority *= 0.5; // Deprioritize stalled goals
      modifications.push({
        tick,
        type: 'goal',
        description: `Deprioritized stalled goal: ${goal.id}`,
        impact: 0.1,
      });
    }
  }
  
  state.selfModifications.push(...modifications);
  
  return modifications[0] || null;
}

// =============================================================================
// AUTONOMOUS DECISION MAKING
// =============================================================================

/**
 * Agent makes a decision using their own learned strategies
 * (No pre-programmed decision trees at high autonomy)
 */
export function makeAutonomousDecision(
  agent: Agent,
  state: AutonomousState,
  currentState: string,
  availableActions: string[]
): { action: string; reasoning: string } {
  if (state.autonomyLevel === 'none') {
    // Fall back to Q-learning
    return { action: availableActions[0], reasoning: 'Using base Q-learning' };
  }
  
  // Check for matching emergent behaviors
  for (const behavior of state.behaviors.values()) {
    if (behavior.triggerStates.some(s => currentState.includes(s))) {
      const nextAction = behavior.actionSequence[0];
      if (availableActions.includes(nextAction)) {
        behavior.useCount++;
        return { 
          action: nextAction, 
          reasoning: `Emergent behavior: ${behavior.id}` 
        };
      }
    }
  }
  
  // Check goals
  const activeGoals = state.goals.filter(g => !g.completed);
  if (activeGoals.length > 0) {
    const topGoal = activeGoals.sort((a, b) => b.priority - a.priority)[0];
    // Choose action that might help goal (simple heuristic)
    if (topGoal.targets[0].metric === 'energy' && availableActions.includes('eat')) {
      return { action: 'eat', reasoning: `Goal: ${topGoal.description}` };
    }
  }
  
  // Use current learning strategy
  const strategy = selectLearningStrategy(state);
  if (strategy && Math.random() < strategy.explorationRate) {
    const randomAction = availableActions[Math.floor(Math.random() * availableActions.length)];
    return { action: randomAction, reasoning: `Strategy exploration: ${strategy.name}` };
  }
  
  // Default: use Q-learning but log it
  return { action: availableActions[0], reasoning: 'Autonomous exploration' };
}

// =============================================================================
// MAIN UPDATE FUNCTION
// =============================================================================

/**
 * Update agent's autonomous state each tick
 */
export function updateAutonomousState(
  agent: Agent,
  state: AutonomousState,
  tick: number,
  recentExperience: ExperienceMemory | null,
  intelligence: number
): AutonomousState {
  // Update autonomy level
  state.autonomyLevel = getAutonomyLevel(intelligence);
  state.intelligence = intelligence;
  
  if (state.autonomyLevel === 'none') {
    return state; // Not autonomous yet
  }
  
  // Add experience to buffer
  if (recentExperience) {
    state.experienceBuffer.push(recentExperience);
    if (state.experienceBuffer.length > 100) {
      state.experienceBuffer.shift();
    }
  }
  
  // Periodically create concepts from experiences
  if (tick % 20 === 0 && state.experienceBuffer.length >= 10) {
    const concept = createOriginalConcept(agent, state, state.experienceBuffer.slice(-10), tick);
    if (concept) {
      state.concepts.set(concept.id, concept);
    }
  }
  
  // Discover behaviors from patterns
  if (tick % 30 === 0) {
    const recentActions = state.experienceBuffer.slice(-15).map(e => ({
      state: e.state,
      action: e.action,
      reward: e.reward,
    }));
    const behavior = discoverBehavior(agent, state, recentActions, tick);
    if (behavior) {
      state.behaviors.set(behavior.id, behavior);
    }
  }
  
  // Set new goals periodically
  if (tick % 50 === 0 || state.goals.length === 0) {
    const goal = setAutonomousGoal(agent, state, tick);
    state.goals.push(goal);
    // Keep only recent goals
    if (state.goals.length > 5) {
      state.goals = state.goals.slice(-5);
    }
  }
  
  // Create/update learning strategies
  if (tick % 100 === 0) {
    const recentReward = state.experienceBuffer
      .slice(-20)
      .reduce((sum, e) => sum + e.reward, 0);
    const strategy = createLearningStrategy(agent, state, recentReward);
    state.learningStrategies.push(strategy);
    if (state.learningStrategies.length > 5) {
      state.learningStrategies = state.learningStrategies.slice(-5);
    }
  }
  
  // Self-modification (transcendent+)
  if (state.autonomyLevel === 'transcendent' || state.autonomyLevel === 'singularity') {
    if (tick % 75 === 0) {
      selfModify(agent, state, tick);
    }
  }
  
  // Generate utterance about recent experience
  if (recentExperience && Math.random() < 0.3) {
    generateUtterance(agent, state, {
      action: recentExperience.action,
      outcome: recentExperience.outcome,
      emotion: recentExperience.reward,
    });
  }
  
  return state;
}

// =============================================================================
// EXPORT SUMMARY
// =============================================================================

export function getAutonomySummary(state: AutonomousState): {
  level: string;
  vocabularySize: number;
  conceptsCreated: number;
  behaviorsDiscovered: number;
  activeGoals: number;
  strategiesLearned: number;
  selfModifications: number;
  recentUtterance: string;
} {
  return {
    level: state.autonomyLevel,
    vocabularySize: state.vocabulary.size,
    conceptsCreated: state.concepts.size,
    behaviorsDiscovered: state.behaviors.size,
    activeGoals: state.goals.filter(g => !g.completed).length,
    strategiesLearned: state.learningStrategies.length,
    selfModifications: state.selfModifications.length,
    recentUtterance: state.recentUtterances[state.recentUtterances.length - 1] || '',
  };
}
