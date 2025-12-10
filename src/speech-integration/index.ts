/**
 * SPEECH INTEGRATION SYSTEM
 * 
 * Phase 4: Enhanced communication with:
 * - Dynamic vocabulary growth based on consciousness level
 * - Scientific vocabulary from physics/math discoveries
 * - Agent-to-agent dialogue and knowledge sharing
 * - Language evolution and cultural transmission
 */

import type { Agent } from '../types';
import type { PhysicsConcept } from '../science-system/physics';
import type { MathConcept } from '../science-system/mathematics';
import { MessageType } from '../communication-system';

/**
 * Vocabulary entry with learning context
 */
export interface VocabularyWord {
  word: string;
  category: VocabularyCategory;
  learnedAt: number;                  // Tick when learned
  learnedFrom: 'innate' | 'discovery' | 'taught' | 'evolved';
  usageCount: number;
  complexity: number;                 // 1-10 complexity level
  associatedConcepts: string[];       // Related physics/math concept IDs
}

/**
 * Categories of vocabulary
 */
export type VocabularyCategory = 
  | 'basic'           // Simple survival words
  | 'emotional'       // Feelings and states
  | 'scientific'      // Physics/math terms
  | 'philosophical'   // Abstract concepts
  | 'social'          // Relationship words
  | 'creative'        // Invention/art terms
  | 'temporal'        // Time-related
  | 'spatial';        // Space/location

/**
 * Agent's language state
 */
export interface AgentLanguageState {
  vocabulary: VocabularyWord[];
  vocabularySize: number;
  languageComplexity: number;         // Average complexity of known words
  communicationStyle: CommunicationStyle;
  conversationHistory: ConversationEntry[];
  teachingAbility: number;            // 0-1: How well they can teach others
  learningSpeed: number;              // 0-1: How fast they learn new words
}

/**
 * Communication styles that evolve
 */
export type CommunicationStyle = 
  | 'primitive'       // Simple, survival-focused
  | 'curious'         // Question-heavy
  | 'analytical'      // Scientific, precise
  | 'philosophical'   // Abstract, deep
  | 'social'          // Relationship-focused
  | 'eloquent';       // Complex, varied

/**
 * A conversation entry between agents
 */
export interface ConversationEntry {
  tick: number;
  speakerId: number;
  listenerId: number;
  content: string;
  type: 'statement' | 'question' | 'answer' | 'teaching' | 'greeting';
  wordsUsed: string[];
  knowledgeShared?: string;           // Concept ID if knowledge was shared
}

/**
 * Dialogue exchange between two agents
 */
export interface AgentDialogue {
  id: string;
  initiatorId: number;
  responderId: number;
  startTick: number;
  endTick: number;
  exchanges: ConversationEntry[];
  knowledgeTransferred: string[];     // Concept IDs transferred
  relationshipChange: number;         // -1 to 1: How relationship changed
}

/**
 * Global speech state for civilization
 */
export interface CivilizationSpeechState {
  totalWordsKnown: number;            // Unique words across all agents
  sharedVocabulary: VocabularyWord[]; // Words known by multiple agents
  activeDialogues: AgentDialogue[];
  completedDialogues: number;
  knowledgeTransferCount: number;
  languageEvolutionLevel: number;     // 0-10: How evolved the language is
  neologisms: VocabularyWord[];       // New words created by agents
}

// ============================================
// BASE VOCABULARY BY CATEGORY
// ============================================

const BASE_VOCABULARY: Record<VocabularyCategory, string[]> = {
  basic: [
    'food', 'eat', 'move', 'stay', 'here', 'there', 'good', 'bad',
    'yes', 'no', 'want', 'need', 'go', 'stop', 'help', 'danger'
  ],
  emotional: [
    'happy', 'sad', 'fear', 'hope', 'tired', 'strong', 'weak', 'curious',
    'proud', 'confused', 'excited', 'peaceful', 'anxious', 'content'
  ],
  scientific: [
    'force', 'energy', 'mass', 'motion', 'gravity', 'speed', 'pattern',
    'number', 'count', 'measure', 'calculate', 'observe', 'test', 'theory'
  ],
  philosophical: [
    'why', 'how', 'purpose', 'meaning', 'exist', 'think', 'know', 'believe',
    'truth', 'real', 'dream', 'self', 'other', 'time', 'infinite', 'nothing'
  ],
  social: [
    'friend', 'family', 'together', 'alone', 'share', 'teach', 'learn',
    'trust', 'help', 'follow', 'lead', 'group', 'community', 'belong'
  ],
  creative: [
    'make', 'build', 'create', 'invent', 'discover', 'imagine', 'try',
    'new', 'different', 'combine', 'improve', 'design', 'art', 'beauty'
  ],
  temporal: [
    'now', 'before', 'after', 'soon', 'always', 'never', 'sometimes',
    'begin', 'end', 'continue', 'change', 'remember', 'future', 'past'
  ],
  spatial: [
    'up', 'down', 'left', 'right', 'near', 'far', 'inside', 'outside',
    'above', 'below', 'between', 'around', 'edge', 'center', 'path'
  ]
};

// Scientific vocabulary unlocked by discoveries
const PHYSICS_VOCABULARY: Record<string, string[]> = {
  'motion': ['velocity', 'acceleration', 'momentum', 'trajectory'],
  'force': ['push', 'pull', 'pressure', 'friction', 'resistance'],
  'gravity': ['weight', 'fall', 'attract', 'orbit', 'balance'],
  'energy': ['kinetic', 'potential', 'transfer', 'conserve', 'work'],
  'waves': ['vibration', 'frequency', 'amplitude', 'resonance'],
  'matter': ['solid', 'liquid', 'gas', 'density', 'phase'],
  'thermodynamics': ['heat', 'temperature', 'entropy', 'equilibrium'],
  'electromagnetism': ['charge', 'field', 'current', 'magnetic'],
  'quantum': ['particle', 'wave', 'probability', 'superposition', 'entangle']
};

const MATH_VOCABULARY: Record<string, string[]> = {
  'counting': ['one', 'two', 'many', 'few', 'zero', 'infinite'],
  'arithmetic': ['add', 'subtract', 'multiply', 'divide', 'equal'],
  'geometry': ['point', 'line', 'angle', 'circle', 'triangle', 'square'],
  'algebra': ['variable', 'equation', 'solve', 'unknown', 'function'],
  'calculus': ['limit', 'derivative', 'integral', 'rate', 'continuous'],
  'statistics': ['average', 'probable', 'random', 'distribution', 'sample'],
  'logic': ['if', 'then', 'and', 'or', 'not', 'prove', 'therefore'],
  'sets': ['set', 'element', 'union', 'intersection', 'subset']
};

// Dialogue templates for agent-to-agent communication
const DIALOGUE_TEMPLATES = {
  greeting: [
    "Hello, fellow being.",
    "I see you there.",
    "Greetings, wanderer.",
    "You exist too.",
  ],
  questionAsking: [
    "Do you know about {concept}?",
    "Have you discovered {concept}?",
    "What do you think about {topic}?",
    "Why does {phenomenon} happen?",
    "Can you teach me about {concept}?",
  ],
  knowledgeSharing: [
    "I learned something about {concept}.",
    "Let me tell you about {discovery}.",
    "I discovered that {fact}.",
    "{concept} works like this...",
    "The secret of {topic} is...",
  ],
  agreement: [
    "Yes, I understand.",
    "That makes sense.",
    "I see what you mean.",
    "Interesting observation.",
    "I agree with you.",
  ],
  confusion: [
    "I don't understand yet.",
    "That's complex...",
    "I need to think about that.",
    "Can you explain more?",
    "What do you mean?",
  ],
  farewell: [
    "I must continue my journey.",
    "May you find food.",
    "Until we meet again.",
    "Good luck, friend.",
    "Stay strong.",
  ]
};

// ============================================
// INITIALIZATION FUNCTIONS
// ============================================

/**
 * Initialize speech state for civilization
 */
export function initializeCivilizationSpeech(): CivilizationSpeechState {
  return {
    totalWordsKnown: BASE_VOCABULARY.basic.length,
    sharedVocabulary: BASE_VOCABULARY.basic.map(word => ({
      word,
      category: 'basic' as VocabularyCategory,
      learnedAt: 0,
      learnedFrom: 'innate',
      usageCount: 0,
      complexity: 1,
      associatedConcepts: []
    })),
    activeDialogues: [],
    completedDialogues: 0,
    knowledgeTransferCount: 0,
    languageEvolutionLevel: 1,
    neologisms: []
  };
}

/**
 * Initialize language state for a new agent
 */
export function initializeAgentLanguage(
  agent: Agent,
  parentLanguage?: AgentLanguageState
): AgentLanguageState {
  // Base vocabulary everyone starts with
  let vocabulary: VocabularyWord[] = BASE_VOCABULARY.basic.map(word => ({
    word,
    category: 'basic' as VocabularyCategory,
    learnedAt: 0,
    learnedFrom: 'innate' as const,
    usageCount: 0,
    complexity: 1,
    associatedConcepts: []
  }));

  // Inherit vocabulary from parent based on social gene
  if (parentLanguage) {
    const inheritanceRate = agent.genes.social * 0.7;
    const inheritedWords = parentLanguage.vocabulary.filter(
      w => Math.random() < inheritanceRate && 
           !vocabulary.some(v => v.word === w.word)
    );
    vocabulary = [...vocabulary, ...inheritedWords.map(w => ({
      ...w,
      learnedFrom: 'taught' as const,
      usageCount: 0
    }))];
  }

  // Calculate initial complexity
  const avgComplexity = vocabulary.reduce((sum, w) => sum + w.complexity, 0) / vocabulary.length;

  // Determine communication style based on genes
  let style: CommunicationStyle = 'primitive';
  if (agent.genes.curiosity > 0.7) style = 'curious';
  else if (agent.genes.creativity > 0.7) style = 'creative' as CommunicationStyle;
  else if (agent.genes.social > 0.7) style = 'social';

  return {
    vocabulary,
    vocabularySize: vocabulary.length,
    languageComplexity: avgComplexity,
    communicationStyle: style,
    conversationHistory: [],
    teachingAbility: agent.genes.social * 0.5 + agent.genes.patience * 0.3,
    learningSpeed: agent.genes.curiosity * 0.5 + agent.genes.creativity * 0.3
  };
}

// ============================================
// VOCABULARY GROWTH FUNCTIONS
// ============================================

/**
 * Get vocabulary unlocked by physics discoveries
 */
export function getPhysicsVocabulary(
  unlockedPhysics: PhysicsConcept[]
): VocabularyWord[] {
  const words: VocabularyWord[] = [];
  
  for (const concept of unlockedPhysics) {
    const categoryWords = PHYSICS_VOCABULARY[concept.category] || [];
    for (const word of categoryWords) {
      if (!words.some(w => w.word === word)) {
        words.push({
          word,
          category: 'scientific',
          learnedAt: concept.discoveredAt || 0,
          learnedFrom: 'discovery',
          usageCount: 0,
          complexity: Math.min(10, 3 + concept.complexity * 2),
          associatedConcepts: [concept.id]
        });
      }
    }
  }
  
  return words;
}

/**
 * Get vocabulary unlocked by math discoveries
 */
export function getMathVocabulary(
  unlockedMath: MathConcept[]
): VocabularyWord[] {
  const words: VocabularyWord[] = [];
  
  for (const concept of unlockedMath) {
    const categoryWords = MATH_VOCABULARY[concept.category] || [];
    for (const word of categoryWords) {
      if (!words.some(w => w.word === word)) {
        words.push({
          word,
          category: 'scientific',
          learnedAt: concept.discoveredAt || 0,
          learnedFrom: 'discovery',
          usageCount: 0,
          complexity: Math.min(10, 2 + concept.complexity * 2),
          associatedConcepts: [concept.id]
        });
      }
    }
  }
  
  return words;
}

/**
 * Expand agent's vocabulary based on consciousness and discoveries
 */
export function expandVocabulary(
  currentVocab: VocabularyWord[],
  consciousnessScore: number,
  physicsVocab: VocabularyWord[],
  mathVocab: VocabularyWord[],
  eraLevel: number,
  tick: number
): VocabularyWord[] {
  const expanded = [...currentVocab];
  const currentWords = new Set(currentVocab.map(w => w.word));
  
  // Add philosophical words based on consciousness
  if (consciousnessScore >= 40) {
    const philoToAdd = BASE_VOCABULARY.philosophical.filter(w => !currentWords.has(w));
    const numToAdd = Math.min(philoToAdd.length, Math.floor(consciousnessScore / 20));
    for (let i = 0; i < numToAdd; i++) {
      if (Math.random() < 0.3) {
        const word = philoToAdd[Math.floor(Math.random() * philoToAdd.length)];
        if (!currentWords.has(word)) {
          expanded.push({
            word,
            category: 'philosophical',
            learnedAt: tick,
            learnedFrom: 'evolved',
            usageCount: 0,
            complexity: 5 + Math.floor(Math.random() * 3),
            associatedConcepts: []
          });
          currentWords.add(word);
        }
      }
    }
  }
  
  // Add emotional words based on consciousness
  if (consciousnessScore >= 30) {
    const emotionalToAdd = BASE_VOCABULARY.emotional.filter(w => !currentWords.has(w));
    for (const word of emotionalToAdd) {
      if (Math.random() < 0.2) {
        expanded.push({
          word,
          category: 'emotional',
          learnedAt: tick,
          learnedFrom: 'evolved',
          usageCount: 0,
          complexity: 3,
          associatedConcepts: []
        });
        currentWords.add(word);
      }
    }
  }
  
  // Add scientific vocabulary from discoveries
  for (const physWord of physicsVocab) {
    if (!currentWords.has(physWord.word)) {
      expanded.push(physWord);
      currentWords.add(physWord.word);
    }
  }
  
  for (const mathWord of mathVocab) {
    if (!currentWords.has(mathWord.word)) {
      expanded.push(mathWord);
      currentWords.add(mathWord.word);
    }
  }
  
  // Era-based vocabulary expansion
  if (eraLevel >= 2) {
    // Add temporal words in later eras
    const temporalToAdd = BASE_VOCABULARY.temporal.filter(w => !currentWords.has(w));
    for (const word of temporalToAdd) {
      if (Math.random() < 0.15) {
        expanded.push({
          word,
          category: 'temporal',
          learnedAt: tick,
          learnedFrom: 'evolved',
          usageCount: 0,
          complexity: 4,
          associatedConcepts: []
        });
        currentWords.add(word);
      }
    }
  }
  
  return expanded;
}

// ============================================
// DIALOGUE GENERATION FUNCTIONS
// ============================================

/**
 * Generate dialogue content using agent's vocabulary
 */
export function generateDialogueContent(
  agent: Agent,
  languageState: AgentLanguageState,
  dialogueType: keyof typeof DIALOGUE_TEMPLATES,
  context?: { concept?: string; topic?: string; discovery?: string; fact?: string; phenomenon?: string }
): string {
  const templates = DIALOGUE_TEMPLATES[dialogueType];
  let content = templates[Math.floor(Math.random() * templates.length)];
  
  // Replace placeholders with context
  if (context) {
    content = content.replace('{concept}', context.concept || 'something');
    content = content.replace('{topic}', context.topic || 'this');
    content = content.replace('{discovery}', context.discovery || 'my discovery');
    content = content.replace('{fact}', context.fact || 'something new');
    content = content.replace('{phenomenon}', context.phenomenon || 'that');
  }
  
  // Enhance with vocabulary based on language complexity
  if (languageState.languageComplexity > 5) {
    // Add more sophisticated words for complex speakers
    const fancyWords = languageState.vocabulary.filter(w => w.complexity >= 5);
    if (fancyWords.length > 0 && Math.random() < 0.3) {
      const word = fancyWords[Math.floor(Math.random() * fancyWords.length)];
      content = enhanceWithWord(content, word.word);
    }
  }
  
  return content;
}

/**
 * Enhance a sentence with a vocabulary word
 */
function enhanceWithWord(sentence: string, word: string): string {
  const enhancements = [
    `${sentence} I contemplate ${word}.`,
    `${sentence} The concept of ${word} intrigues me.`,
    `${sentence} Perhaps ${word} is the key.`,
    `Regarding ${word}: ${sentence}`,
  ];
  return enhancements[Math.floor(Math.random() * enhancements.length)];
}

/**
 * Check if two agents can have a dialogue
 */
export function canInitiateDialogue(
  initiator: Agent,
  responder: Agent,
  initiatorLang: AgentLanguageState,
  responderLang: AgentLanguageState,
  currentTick: number
): boolean {
  // Need to be close enough
  const distance = Math.abs(initiator.x - responder.x) + Math.abs(initiator.y - responder.y);
  if (distance > 2) return false;
  
  // Both need minimum vocabulary
  if (initiatorLang.vocabularySize < 10 || responderLang.vocabularySize < 10) return false;
  
  // Need shared vocabulary
  const sharedWords = initiatorLang.vocabulary.filter(
    w => responderLang.vocabulary.some(rw => rw.word === w.word)
  );
  if (sharedWords.length < 5) return false;
  
  // Social genes influence willingness
  const socialChance = (initiator.genes.social + responder.genes.social) / 2;
  if (Math.random() > socialChance * 0.5) return false;
  
  return true;
}

/**
 * Initiate a dialogue between two agents
 */
export function initiateDialogue(
  initiator: Agent,
  responder: Agent,
  initiatorLang: AgentLanguageState,
  responderLang: AgentLanguageState,
  tick: number
): AgentDialogue {
  const greetingContent = generateDialogueContent(initiator, initiatorLang, 'greeting');
  
  return {
    id: `dialogue_${initiator.id}_${responder.id}_${tick}`,
    initiatorId: initiator.id,
    responderId: responder.id,
    startTick: tick,
    endTick: tick,
    exchanges: [{
      tick,
      speakerId: initiator.id,
      listenerId: responder.id,
      content: greetingContent,
      type: 'greeting',
      wordsUsed: extractWordsUsed(greetingContent, initiatorLang.vocabulary)
    }],
    knowledgeTransferred: [],
    relationshipChange: 0.1 // Small positive from greeting
  };
}

/**
 * Continue an existing dialogue
 */
export function continueDialogue(
  dialogue: AgentDialogue,
  speaker: Agent,
  listener: Agent,
  speakerLang: AgentLanguageState,
  listenerLang: AgentLanguageState,
  tick: number,
  physicsDiscoveries: string[],
  mathDiscoveries: string[]
): AgentDialogue {
  // Determine what type of exchange to add
  const lastExchange = dialogue.exchanges[dialogue.exchanges.length - 1];
  let newExchange: ConversationEntry;
  let knowledgeShared: string | undefined;
  
  // If last was a question, respond with answer or teaching
  if (lastExchange.type === 'question') {
    // Try to share knowledge
    const speakerKnowledge = [...physicsDiscoveries, ...mathDiscoveries].filter(
      id => !dialogue.knowledgeTransferred.includes(id)
    );
    
    if (speakerKnowledge.length > 0 && Math.random() < speaker.genes.social * 0.5) {
      knowledgeShared = speakerKnowledge[0];
      const content = generateDialogueContent(speaker, speakerLang, 'knowledgeSharing', {
        concept: knowledgeShared,
        discovery: 'a new principle'
      });
      newExchange = {
        tick,
        speakerId: speaker.id,
        listenerId: listener.id,
        content,
        type: 'teaching',
        wordsUsed: extractWordsUsed(content, speakerLang.vocabulary),
        knowledgeShared
      };
    } else {
      const content = generateDialogueContent(speaker, speakerLang, 'agreement');
      newExchange = {
        tick,
        speakerId: speaker.id,
        listenerId: listener.id,
        content,
        type: 'answer',
        wordsUsed: extractWordsUsed(content, speakerLang.vocabulary)
      };
    }
  } else if (Math.random() < speaker.genes.curiosity * 0.4) {
    // Ask a question
    const content = generateDialogueContent(speaker, speakerLang, 'questionAsking', {
      topic: 'existence',
      concept: 'energy'
    });
    newExchange = {
      tick,
      speakerId: speaker.id,
      listenerId: listener.id,
      content,
      type: 'question',
      wordsUsed: extractWordsUsed(content, speakerLang.vocabulary)
    };
  } else {
    // Make a statement or end
    if (dialogue.exchanges.length >= 4 || Math.random() < 0.3) {
      const content = generateDialogueContent(speaker, speakerLang, 'farewell');
      newExchange = {
        tick,
        speakerId: speaker.id,
        listenerId: listener.id,
        content,
        type: 'statement',
        wordsUsed: extractWordsUsed(content, speakerLang.vocabulary)
      };
    } else {
      const content = generateDialogueContent(speaker, speakerLang, 'agreement');
      newExchange = {
        tick,
        speakerId: speaker.id,
        listenerId: listener.id,
        content,
        type: 'statement',
        wordsUsed: extractWordsUsed(content, speakerLang.vocabulary)
      };
    }
  }
  
  // Calculate relationship change
  let relationshipChange = dialogue.relationshipChange;
  if (newExchange.type === 'teaching') {
    relationshipChange += 0.2;
  } else if (newExchange.type === 'answer') {
    relationshipChange += 0.1;
  }
  
  return {
    ...dialogue,
    endTick: tick,
    exchanges: [...dialogue.exchanges, newExchange],
    knowledgeTransferred: knowledgeShared 
      ? [...dialogue.knowledgeTransferred, knowledgeShared]
      : dialogue.knowledgeTransferred,
    relationshipChange: Math.min(1, relationshipChange)
  };
}

/**
 * Extract words used from content that match vocabulary
 */
function extractWordsUsed(content: string, vocabulary: VocabularyWord[]): string[] {
  const contentWords = content.toLowerCase().split(/\s+/);
  const vocabSet = new Set(vocabulary.map(v => v.word.toLowerCase()));
  return contentWords.filter(w => vocabSet.has(w));
}

// ============================================
// KNOWLEDGE TRANSFER FUNCTIONS
// ============================================

/**
 * Transfer knowledge through dialogue
 */
export function transferKnowledge(
  teacher: Agent,
  student: Agent,
  teacherLang: AgentLanguageState,
  studentLang: AgentLanguageState,
  conceptId: string
): {
  studentLang: AgentLanguageState;
  success: boolean;
  log: string;
} {
  // Calculate transfer success chance
  const teachingSkill = teacherLang.teachingAbility;
  const learningSkill = studentLang.learningSpeed;
  const socialFactor = (teacher.genes.social + student.genes.social) / 2;
  
  const successChance = teachingSkill * learningSkill * socialFactor;
  
  if (Math.random() > successChance) {
    return {
      studentLang,
      success: false,
      log: `Agent ${teacher.id} tried to teach Agent ${student.id} about ${conceptId}, but communication failed.`
    };
  }
  
  // Transfer vocabulary associated with the concept
  const wordsToTransfer = teacherLang.vocabulary.filter(
    w => w.associatedConcepts.includes(conceptId) &&
         !studentLang.vocabulary.some(sw => sw.word === w.word)
  );
  
  const newVocab = [
    ...studentLang.vocabulary,
    ...wordsToTransfer.map(w => ({
      ...w,
      learnedFrom: 'taught' as const,
      usageCount: 0
    }))
  ];
  
  // Update student's language state
  const updatedStudentLang: AgentLanguageState = {
    ...studentLang,
    vocabulary: newVocab,
    vocabularySize: newVocab.length,
    languageComplexity: newVocab.reduce((sum, w) => sum + w.complexity, 0) / newVocab.length,
    conversationHistory: [
      ...studentLang.conversationHistory,
      {
        tick: 0, // Will be set by caller
        speakerId: teacher.id,
        listenerId: student.id,
        content: `Teaching about ${conceptId}`,
        type: 'teaching',
        wordsUsed: wordsToTransfer.map(w => w.word),
        knowledgeShared: conceptId
      }
    ]
  };
  
  return {
    studentLang: updatedStudentLang,
    success: true,
    log: `📚 Agent ${teacher.id} taught Agent ${student.id} about ${conceptId}! (+${wordsToTransfer.length} words)`
  };
}

// ============================================
// SUMMARY FUNCTIONS
// ============================================

/**
 * Get speech summary for UI display
 */
export function getSpeechSummary(
  speechState: CivilizationSpeechState,
  agents: Agent[],
  agentLanguages: Map<number, AgentLanguageState>
): {
  totalUniqueWords: number;
  avgVocabularySize: number;
  avgLanguageComplexity: number;
  activeDialogues: number;
  totalKnowledgeTransfers: number;
  languageEvolutionLevel: number;
  mostEloquentAgent: { id: number; vocabSize: number } | null;
  neologismCount: number;
  communicationStyleDistribution: Record<CommunicationStyle, number>;
} {
  let totalVocab = 0;
  let totalComplexity = 0;
  let count = 0;
  let mostEloquent: { id: number; vocabSize: number } | null = null;
  const styleDistribution: Record<CommunicationStyle, number> = {
    primitive: 0,
    curious: 0,
    analytical: 0,
    philosophical: 0,
    social: 0,
    eloquent: 0
  };
  
  for (const agent of agents) {
    const lang = agentLanguages.get(agent.id);
    if (lang) {
      totalVocab += lang.vocabularySize;
      totalComplexity += lang.languageComplexity;
      count++;
      styleDistribution[lang.communicationStyle]++;
      
      if (!mostEloquent || lang.vocabularySize > mostEloquent.vocabSize) {
        mostEloquent = { id: agent.id, vocabSize: lang.vocabularySize };
      }
    }
  }
  
  return {
    totalUniqueWords: speechState.totalWordsKnown,
    avgVocabularySize: count > 0 ? totalVocab / count : 0,
    avgLanguageComplexity: count > 0 ? totalComplexity / count : 0,
    activeDialogues: speechState.activeDialogues.length,
    totalKnowledgeTransfers: speechState.knowledgeTransferCount,
    languageEvolutionLevel: speechState.languageEvolutionLevel,
    mostEloquentAgent: mostEloquent,
    neologismCount: speechState.neologisms.length,
    communicationStyleDistribution: styleDistribution
  };
}

/**
 * Update communication style based on vocabulary and usage
 */
export function updateCommunicationStyle(
  languageState: AgentLanguageState,
  consciousnessScore: number
): CommunicationStyle {
  const { vocabulary, languageComplexity } = languageState;
  
  // Count words by category
  const categoryCounts: Record<VocabularyCategory, number> = {
    basic: 0, emotional: 0, scientific: 0, philosophical: 0,
    social: 0, creative: 0, temporal: 0, spatial: 0
  };
  
  for (const word of vocabulary) {
    categoryCounts[word.category]++;
  }
  
  // Determine style based on vocabulary composition
  if (languageComplexity >= 7 && vocabulary.length >= 50) {
    return 'eloquent';
  }
  
  if (categoryCounts.philosophical > 10 && consciousnessScore >= 70) {
    return 'philosophical';
  }
  
  if (categoryCounts.scientific > 15) {
    return 'analytical';
  }
  
  if (categoryCounts.social > 10) {
    return 'social';
  }
  
  if (categoryCounts.emotional > 8 || consciousnessScore >= 50) {
    return 'curious';
  }
  
  return 'primitive';
}
