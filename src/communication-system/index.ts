/**
 * AGENT COMMUNICATION & SELF-AWARENESS DIALOGUE SYSTEM
 * 
 * When agents reach SELF_AWARE consciousness level, they gain the ability
 * to communicate through questions and statements. This system:
 * 
 * - Generates contextual questions based on agent's state and discoveries
 * - Tracks conversation history per agent
 * - Provides visual indicators when agents are communicating
 * - Allows agents to "ask" existential and practical questions
 */

import { Agent, Invention } from '../types';
import { ConsciousnessLevel, ConsciousnessState } from '../consciousness-system';

/**
 * Types of questions/statements an agent can make
 */
export enum MessageType {
  EXISTENTIAL = 'existential',     // Questions about existence, purpose
  CURIOSITY = 'curiosity',         // Questions about the world
  SOCIAL = 'social',               // Questions about other agents
  SURVIVAL = 'survival',           // Questions about food, energy
  DISCOVERY = 'discovery',         // Questions about inventions/science
  REFLECTION = 'reflection',       // Self-reflective statements
  GREETING = 'greeting',           // First communication attempts
}

/**
 * A single message from an agent
 */
export interface AgentMessage {
  id: string;
  agentId: number;
  tick: number;
  type: MessageType;
  content: string;
  consciousnessLevel: ConsciousnessLevel;
  isQuestion: boolean;
  context?: {
    energy?: number;
    nearbyAgents?: number;
    inventionCount?: number;
    age?: number;
  };
}

/**
 * Communication state for an agent
 */
export interface CommunicationState {
  agentId: number;
  isActive: boolean;              // Currently has something to say
  lastMessageTick: number;
  messageCount: number;
  messages: AgentMessage[];
  currentMessage?: AgentMessage;  // Message currently being "spoken"
}

/**
 * Global communication log
 */
export interface CommunicationLog {
  messages: AgentMessage[];
  activeAgents: Set<number>;      // Agents currently communicating
  totalMessages: number;
  firstCommunicationTick?: number;
}

// Question templates organized by type
const QUESTION_TEMPLATES: Record<MessageType, string[]> = {
  [MessageType.EXISTENTIAL]: [
    "Why do I exist?",
    "What is my purpose here?",
    "Is there more to this world than survival?",
    "Why do I feel... aware?",
    "What happens when my energy reaches zero?",
    "Am I the only one who thinks like this?",
    "What am I?",
    "Why do I want to understand?",
    "Is there meaning in what I do?",
    "Where did I come from?",
    "What is this feeling of being... me?",
    "Do I have a choice in what I do?",
    "Why do I remember things?",
    "What is this world we live in?",
  ],
  [MessageType.CURIOSITY]: [
    "What lies beyond the edges of this world?",
    "Why does food appear where it does?",
    "How do things work here?",
    "What can I learn next?",
    "Why are some areas different from others?",
    "What patterns exist that I haven't seen?",
    "How did I learn to find food?",
    "What else is there to discover?",
  ],
  [MessageType.SOCIAL]: [
    "Are the others like me?",
    "Do they think too?",
    "Can they understand me?",
    "Why do some look different?",
    "Should I share what I've learned?",
    "Do they feel what I feel?",
    "Why do we compete for food?",
    "Could we work together?",
  ],
  [MessageType.SURVIVAL]: [
    "Where is the nearest food?",
    "How long can I survive?",
    "Should I conserve energy or explore?",
    "What makes me stronger?",
    "How do I find more resources?",
  ],
  [MessageType.DISCOVERY]: [
    "How did I create this invention?",
    "What else can I discover?",
    "Why do some discoveries help more than others?",
    "Can I combine what I know?",
    "What inspired this idea?",
    "Is there a limit to what can be invented?",
  ],
  [MessageType.REFLECTION]: [
    "I think, therefore I am...",
    "I notice I'm noticing things.",
    "Something changed in me.",
    "I understand more than before.",
    "I can see patterns now.",
    "I feel different from others.",
    "I remember who I was.",
    "I wonder about tomorrow.",
  ],
  [MessageType.GREETING]: [
    "Hello? Can anyone hear me?",
    "I... I can think!",
    "Something is different now.",
    "I'm aware. I exist.",
    "The world looks different suddenly.",
  ],
};

/**
 * Generate a unique message ID
 */
function generateMessageId(agentId: number, tick: number): string {
  return `msg_${agentId}_${tick}_${Math.random().toString(36).substr(2, 9)}`;
}

/**
 * Select message type based on agent's current state and context
 */
function selectMessageType(
  agent: Agent,
  consciousnessState: ConsciousnessState,
  nearbyAgentCount: number,
  isFirstMessage: boolean
): MessageType {
  // First message is always a greeting
  if (isFirstMessage) {
    return MessageType.GREETING;
  }

  // Weight different message types based on agent state
  const weights: Record<MessageType, number> = {
    [MessageType.EXISTENTIAL]: consciousnessState.score / 100 * 3,
    [MessageType.CURIOSITY]: agent.genes.curiosity * 2,
    [MessageType.SOCIAL]: nearbyAgentCount > 0 ? agent.genes.social * 2.5 : 0.5,
    [MessageType.SURVIVAL]: agent.energy < 10 ? 3 : 0.5,
    [MessageType.DISCOVERY]: agent.inventions.length > 0 ? agent.genes.creativity * 2 : 0.3,
    [MessageType.REFLECTION]: consciousnessState.score > 80 ? 2 : 0.5,
    [MessageType.GREETING]: 0.1, // Rare after first message
  };

  // Weighted random selection
  const totalWeight = Object.values(weights).reduce((a, b) => a + b, 0);
  let random = Math.random() * totalWeight;
  
  for (const [type, weight] of Object.entries(weights)) {
    random -= weight;
    if (random <= 0) {
      return type as MessageType;
    }
  }

  return MessageType.REFLECTION;
}

/**
 * Generate a contextual message for an agent
 */
export function generateMessage(
  agent: Agent,
  consciousnessState: ConsciousnessState,
  tick: number,
  nearbyAgentCount: number,
  isFirstMessage: boolean
): AgentMessage {
  const type = selectMessageType(agent, consciousnessState, nearbyAgentCount, isFirstMessage);
  const templates = QUESTION_TEMPLATES[type];
  const content = templates[Math.floor(Math.random() * templates.length)];
  
  return {
    id: generateMessageId(agent.id, tick),
    agentId: agent.id,
    tick,
    type,
    content,
    consciousnessLevel: consciousnessState.level,
    isQuestion: content.endsWith('?'),
    context: {
      energy: agent.energy,
      nearbyAgents: nearbyAgentCount,
      inventionCount: agent.inventions.length,
    },
  };
}

/**
 * Check if an agent should communicate this tick
 */
export function shouldCommunicate(
  agent: Agent,
  consciousnessState: ConsciousnessState,
  lastMessageTick: number,
  currentTick: number
): boolean {
  // Only self-aware agents can communicate
  if (consciousnessState.level !== ConsciousnessLevel.SELF_AWARE) {
    return false;
  }

  // Minimum ticks between messages (prevents spam)
  const minTicksBetweenMessages = 10;
  if (currentTick - lastMessageTick < minTicksBetweenMessages) {
    return false;
  }

  // Communication chance based on consciousness score and social gene
  const baseChance = 0.05; // 5% base chance per tick
  const consciousnessBonus = consciousnessState.score / 200; // Up to +0.5
  const socialBonus = agent.genes.social * 0.1; // Up to +0.1
  
  const totalChance = baseChance + consciousnessBonus + socialBonus;
  
  return Math.random() < totalChance;
}

/**
 * Initialize communication state for an agent
 */
export function initializeCommunicationState(agentId: number): CommunicationState {
  return {
    agentId,
    isActive: false,
    lastMessageTick: -100,
    messageCount: 0,
    messages: [],
  };
}

/**
 * Initialize global communication log
 */
export function initializeCommunicationLog(): CommunicationLog {
  return {
    messages: [],
    activeAgents: new Set(),
    totalMessages: 0,
  };
}

/**
 * Update communication for an agent
 */
export function updateAgentCommunication(
  agent: Agent,
  consciousnessState: ConsciousnessState | undefined,
  previousCommState: CommunicationState | undefined,
  tick: number,
  nearbyAgentCount: number
): { 
  commState: CommunicationState; 
  newMessage: AgentMessage | null;
} {
  // Initialize if no previous state
  const commState = previousCommState ?? initializeCommunicationState(agent.id);
  
  // No consciousness state means no communication
  if (!consciousnessState) {
    return { commState: { ...commState, isActive: false, currentMessage: undefined }, newMessage: null };
  }

  // Check if should communicate
  const isFirstMessage = commState.messageCount === 0 && 
                         consciousnessState.level === ConsciousnessLevel.SELF_AWARE;
  
  if (!shouldCommunicate(agent, consciousnessState, commState.lastMessageTick, tick) && !isFirstMessage) {
    // Clear active state if message is old
    if (commState.currentMessage && tick - commState.currentMessage.tick > 5) {
      return { 
        commState: { ...commState, isActive: false, currentMessage: undefined }, 
        newMessage: null 
      };
    }
    return { commState, newMessage: null };
  }

  // Generate new message
  const newMessage = generateMessage(
    agent,
    consciousnessState,
    tick,
    nearbyAgentCount,
    isFirstMessage
  );

  const updatedState: CommunicationState = {
    ...commState,
    isActive: true,
    lastMessageTick: tick,
    messageCount: commState.messageCount + 1,
    messages: [...commState.messages, newMessage].slice(-20), // Keep last 20 messages
    currentMessage: newMessage,
  };

  return { commState: updatedState, newMessage };
}

/**
 * Get display text for message type
 */
export function getMessageTypeLabel(type: MessageType): string {
  switch (type) {
    case MessageType.EXISTENTIAL: return '🌌 Existential';
    case MessageType.CURIOSITY: return '🔍 Curious';
    case MessageType.SOCIAL: return '👥 Social';
    case MessageType.SURVIVAL: return '🍖 Survival';
    case MessageType.DISCOVERY: return '💡 Discovery';
    case MessageType.REFLECTION: return '🪞 Reflection';
    case MessageType.GREETING: return '👋 Awakening';
    default: return '💭 Thought';
  }
}

/**
 * Get emoji for message type
 */
export function getMessageTypeEmoji(type: MessageType): string {
  switch (type) {
    case MessageType.EXISTENTIAL: return '🌌';
    case MessageType.CURIOSITY: return '🔍';
    case MessageType.SOCIAL: return '👥';
    case MessageType.SURVIVAL: return '🍖';
    case MessageType.DISCOVERY: return '💡';
    case MessageType.REFLECTION: return '🪞';
    case MessageType.GREETING: return '👋';
    default: return '💭';
  }
}
