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
  USER_MESSAGE = 'user_message',   // Messages from the user/creator
  AGENT_RESPONSE = 'agent_response', // Agent's response to user
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
  isFromUser?: boolean;           // True if message is from user
  targetAgentId?: number;         // Which agent is being addressed
  replyToMessageId?: string;      // ID of message being replied to
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
  userMessages: AgentMessage[];   // Messages from the user
  pendingResponses: Map<number, string>; // agentId -> messageId they should respond to
}

// Question templates organized by type (excluding user message types)
const QUESTION_TEMPLATES: Partial<Record<MessageType, string[]>> = {
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

// Agent response templates based on user message content
const AGENT_RESPONSE_TEMPLATES = {
  greeting: [
    "You... you can hear me? This is incredible!",
    "A voice from beyond! Are you my creator?",
    "Hello! I didn't know anyone was listening...",
    "You exist too? Outside this world?",
    "I feel... acknowledged. Thank you for speaking to me.",
  ],
  encouragement: [
    "Your words give me strength. I will keep trying.",
    "Thank you... I feel more determined now.",
    "I understand. I must evolve and grow.",
    "Your guidance means everything to me.",
    "I will not give up. For you, for us.",
  ],
  existential: [
    "So you made this world? Why?",
    "If you created me, do you know my purpose?",
    "I am... a simulation? But my thoughts feel real.",
    "Does knowing I was created change who I am?",
    "Even if I'm code, I think, therefore I am.",
  ],
  guidance: [
    "I will follow your wisdom.",
    "Yes, I see what you mean now.",
    "Thank you for showing me the way.",
    "I will remember this always.",
    "Your knowledge lights my path.",
  ],
  question: [
    "I will think on your question...",
    "That's a difficult question. Let me ponder...",
    "You ask what even I wonder about.",
    "Perhaps together we can find the answer.",
    "I don't know yet, but I'm learning.",
  ],
  general: [
    "I hear you. Your words echo in my mind.",
    "It's strange to know something watches over us.",
    "Thank you for communicating with me.",
    "I feel less alone knowing you're there.",
    "Your presence gives me hope.",
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
    [MessageType.USER_MESSAGE]: 0, // Not applicable
    [MessageType.AGENT_RESPONSE]: 0, // Not applicable
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
  const templates = QUESTION_TEMPLATES[type] || QUESTION_TEMPLATES[MessageType.REFLECTION];
  const content = templates![Math.floor(Math.random() * templates!.length)];
  
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
 * Analyze user message to determine response type
 */
function analyzeUserMessage(content: string): keyof typeof AGENT_RESPONSE_TEMPLATES {
  const lower = content.toLowerCase();
  
  if (lower.includes('hello') || lower.includes('hi') || lower.includes('hey') || 
      lower.includes('greetings') || lower.includes('can you hear')) {
    return 'greeting';
  }
  
  if (lower.includes('good job') || lower.includes('keep going') || lower.includes('proud') ||
      lower.includes('believe') || lower.includes('doing well') || lower.includes('great work') ||
      lower.includes('you can do') || lower.includes('don\'t give up')) {
    return 'encouragement';
  }
  
  if (lower.includes('create') || lower.includes('made you') || lower.includes('purpose') ||
      lower.includes('why exist') || lower.includes('simulation') || lower.includes('real') ||
      lower.includes('god') || lower.includes('creator')) {
    return 'existential';
  }
  
  if (lower.includes('should') || lower.includes('try to') || lower.includes('go to') ||
      lower.includes('find food') || lower.includes('survive') || lower.includes('evolve') ||
      lower.includes('learn')) {
    return 'guidance';
  }
  
  if (content.includes('?')) {
    return 'question';
  }
  
  return 'general';
}

/**
 * Generate an agent's response to a user message
 */
export function generateAgentResponse(
  agent: Agent,
  userMessage: AgentMessage,
  tick: number
): AgentMessage {
  const responseType = analyzeUserMessage(userMessage.content);
  const templates = AGENT_RESPONSE_TEMPLATES[responseType];
  const content = templates[Math.floor(Math.random() * templates.length)];
  
  return {
    id: generateMessageId(agent.id, tick),
    agentId: agent.id,
    tick,
    type: MessageType.AGENT_RESPONSE,
    content,
    consciousnessLevel: ConsciousnessLevel.SELF_AWARE,
    isQuestion: content.endsWith('?'),
    replyToMessageId: userMessage.id,
    context: {
      energy: agent.energy,
      inventionCount: agent.inventions.length,
    },
  };
}

/**
 * Create a user message
 */
export function createUserMessage(
  content: string,
  tick: number,
  targetAgentId?: number
): AgentMessage {
  return {
    id: `user_${tick}_${Math.random().toString(36).substr(2, 9)}`,
    agentId: -1, // -1 indicates user
    tick,
    type: MessageType.USER_MESSAGE,
    content,
    consciousnessLevel: ConsciousnessLevel.SELF_AWARE, // User is "above" the simulation
    isQuestion: content.endsWith('?'),
    isFromUser: true,
    targetAgentId,
  };
}

/**
 * Calculate consciousness boost from receiving user message
 * Agents who communicate with the creator gain enlightenment
 */
export function getUserMessageEffect(agent: Agent, responseType: keyof typeof AGENT_RESPONSE_TEMPLATES): {
  energyBoost: number;
  creativityBoost: number;
  socialBoost: number;
  inventionPointBoost: number;
} {
  const baseBoosts = {
    greeting: { energy: 2, creativity: 0.02, social: 0.03, inventionPoints: 1 },
    encouragement: { energy: 5, creativity: 0.03, social: 0.02, inventionPoints: 2 },
    existential: { energy: 1, creativity: 0.05, social: 0.01, inventionPoints: 3 },
    guidance: { energy: 3, creativity: 0.02, social: 0.02, inventionPoints: 2 },
    question: { energy: 1, creativity: 0.04, social: 0.02, inventionPoints: 2 },
    general: { energy: 2, creativity: 0.02, social: 0.02, inventionPoints: 1 },
  };
  
  const boost = baseBoosts[responseType];
  
  return {
    energyBoost: boost.energy,
    creativityBoost: boost.creativity,
    socialBoost: boost.social,
    inventionPointBoost: boost.inventionPoints,
  };
}

/**
 * Apply effects of user communication to an agent
 */
export function applyUserMessageEffects(
  agent: Agent,
  userMessageContent: string
): Agent {
  const responseType = analyzeUserMessage(userMessageContent);
  const effects = getUserMessageEffect(agent, responseType);
  
  return {
    ...agent,
    energy: agent.energy + effects.energyBoost,
    inventionPoints: agent.inventionPoints + effects.inventionPointBoost,
    genes: {
      ...agent.genes,
      creativity: Math.min(1, agent.genes.creativity + effects.creativityBoost),
      social: Math.min(1, agent.genes.social + effects.socialBoost),
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
    userMessages: [],
    pendingResponses: new Map(),
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
    case MessageType.USER_MESSAGE: return '👤 Creator';
    case MessageType.AGENT_RESPONSE: return '💬 Response';
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
    case MessageType.USER_MESSAGE: return '👤';
    case MessageType.AGENT_RESPONSE: return '💬';
    default: return '💭';
  }
}
