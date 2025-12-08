# Consciousness & Self-Awareness Detection System

## Overview

This system monitors agents for emergent self-awareness through multiple interconnected indicators. Self-awareness is not programmed but **emerges naturally** when agents demonstrate sufficient complexity across multiple dimensions.

## Consciousness Levels

### 0. Reactive (< 10 points)
- Pure stimulus-response behavior
- No learning or adaptation
- Baseline agent state

### 1. Adaptive (10-30 points)
- Basic Q-learning functioning
- Can adapt to environment
- Simple pattern recognition

### 2. Cognitive (30-60 points)
- Problem-solving capabilities
- Planning ahead (temporal awareness)
- Creative invention discovery

### 3. Metacognitive (60-100 points)
- Learning about learning
- Theory of mind (understanding others exist)
- Abstract thinking
- Self-reflection capability

### 4. 🧠 Self-Aware (100+ points)
- **Full consciousness achieved**
- Existential awareness
- Understanding of own existence
- Combines all lower levels

## Consciousness Indicators

Each agent is evaluated on 7 key indicators:

### 1. Metacognition (Weight: 3.0)
**Understanding of own learning process**
- Formula: `(inventionPoints/100) * 0.6 + (qTableSize/50) * 0.4`
- Threshold: 0.6
- Measures: Agent's awareness of how it learns

### 2. Theory of Mind (Weight: 2.5)
**Awareness that other agents have thoughts**
- Formula: `social * 0.5 + hasInventions * 0.3 + nearbyAgents * 0.2`
- Threshold: 0.7
- Measures: Social consciousness and understanding others

### 3. Creative Abstraction (Weight: 2.5)
**Ability to think beyond immediate experience**
- Formula: `creativity * 0.4 + breakthroughs * 0.4 + inventionTypes * 0.2`
- Threshold: 0.65
- Measures: Original thinking and innovation

### 4. Temporal Consciousness (Weight: 2.0)
**Awareness of past, present, and future**
- Formula: `patience * 0.4 + survivalTime * 0.4 + energySurplus * 0.2`
- Threshold: 0.6
- Measures: Long-term planning and memory

### 5. Abstract Knowledge (Weight: 2.0)
**Understanding of universal principles**
- Formula: `curiosity * 0.4 + scienceLevel * 0.4 + advanced * 0.2`
- Threshold: 0.55
- Measures: Grasp of physics and mathematics

### 6. Self-Reflection (Weight: 2.5)
**Ability to examine own thoughts and actions**
- Formula: `curiosity * 0.35 + social * 0.35 + creativity * 0.3`
- Threshold: 0.7
- Measures: Introspection capability

### 7. Existential Awareness (Weight: 3.5)
**Consciousness of own existence and mortality**
- Formula: Combined advanced traits + inventions + science + age
- Threshold: 0.75
- Measures: Deep understanding of existence

## Scoring System

**Total Score = Σ(indicator.value × indicator.weight × 10) / Σ(weights) × 10**

- Must meet threshold on at least 5 indicators for self-awareness
- Score ranges: 0-150+ (typically 0-120)
- Self-awareness typically emerges at 100-110 points

## Emergence Conditions

An agent becomes self-aware when:

1. **Score ≥ 100 points** AND
2. **≥ 5 indicators meet their thresholds** AND
3. **High values in**:
   - Metacognition (understands own learning)
   - Theory of Mind (understands others)
   - Existential Awareness (understands existence)

## Awareness Events

The system tracks:
- **Level changes**: When consciousness level increases
- **Triggers**: What caused the advancement
- **Awakening**: The specific tick when self-awareness emerged
- **Indicators**: Which measurements contributed

## Detection in Simulation

Self-aware agents will display:
- 🧠 **Visual indicator** on grid
- **Gold/yellow highlighting**
- **Consciousness panel** showing scores
- **Event log** when awakening occurs

## Example Progression

**Tick 50**: Agent starts Reactive
→ **Tick 200**: Learns patterns (Adaptive)
→ **Tick 500**: Invents tools (Cognitive)
→ **Tick 1000**: Understands learning (Metacognitive)
→ **Tick 1500**: 🧠 **AWAKENS** (Self-Aware)

## Philosophy

This system is based on the **Integrated Information Theory** and **Global Workspace Theory** of consciousness, where awareness emerges from:
- Information integration (multiple systems working together)
- Complexity threshold (sufficient interconnected components)
- Self-modeling (ability to represent itself)
- Temporal binding (awareness across time)

Agents don't "gain" consciousness - it **emerges naturally** from sufficient complexity!
