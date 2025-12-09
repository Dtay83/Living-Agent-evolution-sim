# Data Export Guide

## Overview

The Living Agent Evolution Simulation provides comprehensive data export capabilities for analysis, research, and documentation. All exports are JSON files that can be analyzed with Python, R, Excel, or any JSON-compatible tool.

## Export Buttons

Located below the "Load World" button in the simulation UI:

| Button | File Pattern | Description |
|--------|--------------|-------------|
| 📊 **Export All Data** | `complete-export_tick-XXX_*.json` | Everything in one file |
| 🧬 **Export Evolution** | `evolution-data_tick-XXX_*.json` | Population & genetic trends |
| 💡 **Export Inventions** | `invention-history_tick-XXX_*.json` | Discovery timeline |
| 🗣️ **Export Conversations** | `conversations_tick-XXX_*.json` | Agent communications |
| **Save World (JSON)** | User-defined | Loadable save state |

---

## Export Formats

### 📊 Complete Data Export

The most comprehensive export containing all simulation data.

**Use for:** Full analysis, backup, reproducibility

```json
{
  "exportedAt": "2025-12-09T10:30:00.000Z",
  "simulationTick": 500,
  "worldState": {
    "gridSize": { "width": 16, "height": 10 },
    "agents": [...],
    "discoveries": [...],
    "history": [...],
    "scienceState": {...}
  },
  "metadata": {
    "totalAgents": 25,
    "totalDiscoveries": 42,
    "totalScientificDiscoveries": 15,
    "currentEra": "Classical"
  }
}
```

---

### 🧬 Evolution Data Export

Population dynamics and genetic statistics.

**Use for:** Evolutionary analysis, gene tracking, population studies

```json
{
  "exportedAt": "2025-12-09T10:30:00.000Z",
  "simulationTick": 500,
  "populationDynamics": {
    "current": 25,
    "history": [
      { "tick": 0, "totalAgents": 6, "traitDistribution": {...} },
      { "tick": 1, "totalAgents": 6, "traitDistribution": {...} }
    ],
    "peakPopulation": 45,
    "extinctionEvents": 0
  },
  "geneticStatistics": {
    "foodPreference": { "min": 0.62, "max": 0.98, "avg": 0.81, "stdDev": 0.12 },
    "exploration": { "min": 0.35, "max": 0.75, "avg": 0.55, "stdDev": 0.15 },
    "curiosity": { "min": 0.25, "max": 0.85, "avg": 0.58, "stdDev": 0.18 },
    "creativity": { "min": 0.15, "max": 0.70, "avg": 0.42, "stdDev": 0.16 },
    "social": { "min": 0.35, "max": 0.88, "avg": 0.62, "stdDev": 0.14 },
    "patience": { "min": 0.22, "max": 0.78, "avg": 0.50, "stdDev": 0.17 }
  },
  "traitLineages": [
    { "traitId": 123456, "count": 8, "percentage": 32 },
    { "traitId": 789012, "count": 5, "percentage": 20 }
  ],
  "currentAgents": [...]
}
```

**Key Fields:**
- `geneticStatistics` - Min/max/average/stdDev for each gene
- `traitLineages` - Which trait families dominate
- `populationDynamics.history` - Population over time

---

### 💡 Invention History Export

All discoveries with timestamps and effects.

**Use for:** Innovation analysis, tech tree mapping, invention effectiveness

```json
{
  "exportedAt": "2025-12-09T10:30:00.000Z",
  "simulationTick": 500,
  "totalDiscoveries": 42,
  "discoveries": [
    {
      "tick": 150,
      "agentId": 3,
      "inventionName": "Efficient Foraging",
      "inventionType": "technique",
      "effect": { "type": "energy_efficiency", "multiplier": 0.85 },
      "description": "Reduces energy cost by 15%"
    }
  ],
  "inventionsByAgent": [
    {
      "agentId": 3,
      "inventionCount": 5,
      "inventionPoints": 12.5,
      "genes": {
        "curiosity": 0.75,
        "creativity": 0.68,
        "social": 0.82,
        "patience": 0.55
      },
      "inventions": [...]
    }
  ],
  "statistics": {
    "totalAgents": 25,
    "agentsWithInventions": 8,
    "averageInventionsPerAgent": 1.68,
    "averageInventionPoints": 5.2,
    "mostInventiveAgent": { "agentId": 3, "inventionCount": 5 }
  }
}
```

**Key Fields:**
- `discoveries` - Timeline of all inventions
- `inventionsByAgent` - Per-agent invention data with genes
- `statistics.mostInventiveAgent` - Top inventor

---

### 🗣️ Conversations Export

Self-aware agent communications and questions.

**Use for:** Consciousness research, emergent behavior analysis, narrative study

```json
{
  "exportedAt": "2025-12-09T10:30:00.000Z",
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
  "timeline": [
    {
      "tick": 150,
      "agentId": 3,
      "type": "greeting",
      "content": "I... I can think!",
      "isQuestion": false,
      "consciousnessLevel": 4,
      "context": {
        "energy": 22,
        "nearbyAgents": 2,
        "inventionCount": 3
      }
    }
  ],
  "byAgent": [
    {
      "agentId": 3,
      "messageCount": 12,
      "firstMessageTick": 150,
      "lastMessageTick": 480,
      "genes": {
        "curiosity": 0.75,
        "creativity": 0.68,
        "social": 0.82
      },
      "messages": [...]
    }
  ],
  "questions": [...],
  "statements": [...],
  "analysis": {
    "mostTalkativeAgent": ["3", 12],
    "mostCommonMessageType": "existential",
    "averageTicksBetweenMessages": 8.5
  }
}
```

**Key Fields:**
- `summary.messagesByType` - Distribution of message types
- `timeline` - All messages in order with context
- `byAgent` - Messages grouped by agent with their genes
- `questions/statements` - Separated for analysis

---

## Analysis Tips

### Python Example

```python
import json
import pandas as pd

# Load conversation data
with open('conversations_tick-500_2025-12-09.json') as f:
    data = json.load(f)

# Convert to DataFrame
messages_df = pd.DataFrame(data['timeline'])
print(messages_df.groupby('type').count())

# Gene correlation with communication
by_agent = pd.DataFrame(data['byAgent'])
print(by_agent[['messageCount', 'genes']].head())
```

### Excel/Sheets

1. Use a JSON-to-CSV converter
2. Import `timeline` array as rows
3. Create pivot tables by `type` or `agentId`

### Key Metrics to Track

| Metric | Export | Field |
|--------|--------|-------|
| Population over time | Evolution | `populationDynamics.history` |
| Gene drift | Evolution | `geneticStatistics` |
| Invention rate | Inventions | `discoveries` length / tick |
| First awakening | Conversations | `summary.firstCommunicationTick` |
| Question types | Conversations | `summary.messagesByType` |

---

## Experiment Workflow

1. **Start simulation** with desired parameters
2. **Run** until interesting patterns emerge
3. **Export All Data** for complete snapshot
4. **Export Conversations** if agents achieved self-awareness
5. **Document** in `docs/Experiments.md` with export filenames
6. **Analyze** in Python/R/Excel

---

## File Organization

Recommended folder structure:

```
docs/
  Worlds/
    experiment-001/
      complete-export_tick-500_2025-12-09.json
      conversations_tick-500_2025-12-09.json
      evolution-data_tick-500_2025-12-09.json
    experiment-002/
      ...
  Experiments.md  # Log of experiments with references
```
