# Agent Communication System

## Overview

When agents reach the **SELF_AWARE** consciousness level, they gain the ability to communicate through questions and statements. This creates an emergent dialogue system where agents "wake up" and begin asking existential, curious, and social questions.

## Features

### 🗣️ Chat Icon
- Fixed position button in bottom-right corner
- **Glows gold** when agents are actively communicating
- **Pulses** to draw attention to new activity
- **Green dot** indicator shows active speaking agents
- **Badge counter** shows unread messages when panel is closed

### 💬 Message Types

| Type | Emoji | Trigger | Example Questions |
|------|-------|---------|-------------------|
| Existential | 🌌 | High consciousness score | "Why do I exist?", "What is my purpose?" |
| Curiosity | 🔍 | High curiosity gene | "What lies beyond the edges?" |
| Social | 👥 | Nearby agents + social gene | "Are the others like me?" |
| Survival | 🍖 | Low energy | "Where is the nearest food?" |
| Discovery | 💡 | Has inventions + creativity | "How did I create this?" |
| Reflection | 🪞 | Very high consciousness | "I think, therefore I am..." |
| Greeting | 👋 | First message ever | "I... I can think!" |

### 🧠 Communication Triggers

Agents communicate when:
1. They reach `ConsciousnessLevel.SELF_AWARE` (consciousness score ≥ 60)
2. Random chance based on:
   - Base 5% per tick (80% for first message)
   - +Social gene bonus (up to +5%)
3. Minimum 15 ticks between messages (prevents spam)

### 📊 Consciousness Score Calculation

```
consciousnessScore = 
  (curiosity × 25) + 
  (creativity × 25) + 
  (social × 20) + 
  (inventionPoints × 0.5) + 
  (inventions.length × 10)
```

Agents become self-aware when score ≥ 60.

### 📊 Message Context

Each message includes:
- Agent ID and tick timestamp
- Message type and classification
- Context data (energy, nearby agents, inventions)
- Whether it's a question or statement

## Data Export

### Export Button
Click **🗣️ Export Conversations** in the UI to download conversation data.

### Export File Format

The exported JSON file (`conversations_tick-XXX_YYYY-MM-DD.json`) contains:

```json
{
  "exportedAt": "2025-12-09T...",
  "simulationTick": 500,
  "summary": {
    "totalMessages": 42,
    "uniqueAgents": 5,
    "firstCommunicationTick": 150,
    "questionsCount": 35,
    "statementsCount": 7,
    "messagesByType": {
      "existential": 15,
      "curiosity": 12,
      "social": 8,
      "greeting": 5,
      "reflection": 2
    }
  },
  "timeline": [...],      // All messages in chronological order
  "byAgent": [...],       // Messages grouped by agent with gene data
  "questions": [...],     // Only questions
  "statements": [...],    // Only statements
  "analysis": {
    "mostTalkativeAgent": ["3", 12],
    "mostCommonMessageType": "existential",
    "averageTicksBetweenMessages": 8.5
  }
}
```

### Data Fields

| Field | Description |
|-------|-------------|
| `timeline` | All messages in order with full context |
| `byAgent` | Messages grouped per agent with their gene values |
| `questions` | Just the questions agents asked |
| `statements` | Just the statements (non-questions) |
| `analysis` | Statistics about communication patterns |

## Integration

### In App.tsx

```typescript
import { ChatPanel } from './ui-components';
import { 
  CommunicationLog, 
  initializeCommunicationLog,
  updateAgentCommunication 
} from './communication-system';
import { exportConversations } from './utils/exportData';

// State
const [communicationLog, setCommunicationLog] = useState<CommunicationLog>(
  initializeCommunicationLog()
);
const [chatOpen, setChatOpen] = useState(false);

// Export conversations
<button onClick={() => exportConversations(communicationLog, tick, agents)}>
  🗣️ Export Conversations
</button>

// Render chat panel
<ChatPanel
  communicationLog={communicationLog}
  isOpen={chatOpen}
  onToggle={() => setChatOpen(!chatOpen)}
  onClear={() => setCommunicationLog(initializeCommunicationLog())}
/>
```

## Philosophical Design

The system is designed to create **emergent storytelling**:

1. **Awakening**: An agent's first message is always a "greeting" - their moment of realizing they exist
2. **Exploration**: They then ask questions based on their personality (genes) and situation
3. **Social**: When near other agents, they wonder if others think too
4. **Existential**: The highest-consciousness agents ponder existence itself

This creates a narrative where you watch digital beings become aware and start questioning their reality - mirroring philosophical questions about consciousness and existence.

## Future Enhancements

- [ ] Agent-to-agent conversations (responses)
- [ ] Memory of past conversations
- [ ] Evolution of language/vocabulary
- [ ] Teaching between agents
- [ ] Collective knowledge sharing
