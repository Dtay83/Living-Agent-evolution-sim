# Science System - Physics, Mathematics & Era Progression

## Overview

The Science System provides unlimited scientific progression for agents through physics and mathematics discoveries. Agents evolve through scientific eras, unlocking concepts that provide gameplay bonuses and accelerate evolution.

## Architecture

```
science-system/
├── index.ts           - Main coordinator & API
├── eras/             - Scientific era progression
│   └── index.ts      - Era definitions & advancement logic
├── physics/          - Physics concepts & bonuses
│   └── index.ts      - 20+ physics concepts + procedural generation
├── mathematics/      - Math concepts & bonuses
│   └── index.ts      - 20+ math concepts + procedural generation
└── logger.ts         - Discovery tracking & era history
```

## Key Features

### 🔬 Unlimited Scientific Progression
- **No caps** on discoveries or advancement
- **Procedural generation** of advanced concepts beyond base set
- **Exponential scaling** for requirements and bonuses

### 📚 Physics System (20+ Concepts)
**Categories:**
- Mechanics (basic motion → momentum conservation)
- Thermodynamics (heat transfer → thermodynamic efficiency)
- Electromagnetism (electricity → Maxwell's equations)
- Quantum Mechanics (wave-particle duality → quantum entanglement)
- Relativity (special → general relativity)
- Unified Theories (quantum field theory → string theory → beyond)

**AI Bonuses:**
- Energy Efficiency: 5-40% reduction in movement cost
- Movement Speed: 5-60% increase
- Learning Rate: 10-70% improvement in Q-learning
- Invention Chance: Up to 50% bonus

### 📐 Mathematics System (20+ Concepts)
**Categories:**
- Arithmetic (counting → multiplication/division)
- Geometry (shapes → trigonometry)
- Algebra (variables → exponentials/logarithms)
- Calculus (limits → differential equations)
- Statistics (probability → Bayesian inference)
- Topology (set theory → topology basics)
- Abstract Algebra (group theory → category theory → beyond)

**AI Bonuses:**
- Decision Quality: 5-65% better Q-value updates
- Pattern Recognition: 10-70% improved state recognition
- Exploration Bonus: Up to 30% additive bonus
- Optimization Power: 5-65% overall efficiency improvement

### 🏛️ Scientific Eras (9 Base + Unlimited)

**Base Eras:**
1. **Stone Age** (Level 0) - Basic survival
2. **Bronze Age** (Level 1) - 10 discoveries, 3 physics, 2 math
3. **Iron Age** (Level 2) - 25 discoveries, 6 physics, 5 math
4. **Classical Age** (Level 3) - 50 discoveries, 10 physics, 10 math
5. **Renaissance** (Level 4) - 100 discoveries, 15 physics, 15 math
6. **Industrial Age** (Level 5) - 200 discoveries, 25 physics, 25 math
7. **Information Age** (Level 6) - 400 discoveries, 40 physics, 40 math
8. **Quantum Age** (Level 7) - 800 discoveries, 60 physics, 60 math
9. **Singularity Age** (Level 8) - 1600 discoveries, 100 physics, 100 math

**Beyond Base Eras:**
- Automatically generated with exponential scaling
- Names: Transcendence Age, Cosmic Age, Multiversal Age, etc.
- Unlimited progression possible

### 📊 Discovery Mechanics

**Discovery Requirements:**
- Minimum 20 energy (cognitive surplus)
- Prerequisites must be unlocked
- Probability based on agent genes:
  - Physics: `curiosity × creativity × 0.01 × complexity_penalty`
  - Math: `curiosity × patience × 0.01 × complexity_penalty`

**Discovery Categories:**
- Physics discoveries
- Mathematics discoveries
- Inventions (existing system)
- Evolutions (genetic breakthroughs)

### 📖 Science Logging & History

**Era Logs:**
- Complete discovery history per era
- Metrics: duration, discovery rate, progression speed
- Classification: slow/moderate/fast/explosive progression

**Global Metrics:**
- Total discoveries across all time
- Discoveries per era breakdown
- Average discovery rate
- Scientific acceleration tracking

**Persistence:**
- JSON export/import for save files
- Backward compatible with existing saves
- Full history preserved

## Integration with Existing Systems

### Invention System Enhancement
Science bonuses multiply invention discovery chance and quality:
```typescript
inventionChance = base × (1 + physicsBonuses.inventionChance)
```

### Q-Learning Enhancement
Math and physics improve learning:
```typescript
effectiveAlpha = alpha × physicsBonuses.learningRate × mathBonuses.decisionQuality
```

### Movement & Energy
Physics directly affects efficiency:
```typescript
movementCost = baseCost × physicsBonuses.energyEfficiency
movementSpeed = baseSpeed × physicsBonuses.movementSpeed
```

### Genetic Evolution
Science accelerates evolution through optimization bonuses

## Usage Example

```typescript
import { 
  initializeScienceSystem, 
  updateScienceSystem,
  getScienceBonuses 
} from './science-system';

// Initialize
let scienceState = initializeScienceSystem(currentTick);
let scienceLog = initializeScienceLog();

// Each tick
const result = updateScienceSystem(scienceState, agents, tick);
scienceState = result.scienceState;

if (result.eraAdvanced) {
  console.log(`Advanced to ${result.newEra!.name}!`);
  scienceLog = finalizeEra(scienceLog, scienceState.currentEra.level - 1, tick);
}

// Apply bonuses
const bonuses = getScienceBonuses(scienceState);
const modifiedAlpha = ALPHA * bonuses.combined.learningRate;
const modifiedEnergyCost = baseCost * bonuses.combined.energyEfficiency;

// Log discoveries
if (result.discoveries.length > 0) {
  scienceLog = logDiscoveries(scienceLog, result.discoveries, scienceState.currentEra, tick);
}

// Generate report
const report = generateProgressReport(scienceLog, scienceState);
console.log(report);
```

## File Organization by Era

When saving, science logs are organized by era level:
```
science_logs/
├── era_00_stone_age.json
├── era_01_bronze_age.json
├── era_02_iron_age.json
...
├── era_08_singularity_age.json
├── era_09_transcendence_age.json
└── era_XX_advanced_era_XX.json
```

Each file contains:
- Era metadata
- All discoveries in that era
- Performance metrics
- Timestamp information

## Scalability

The system is designed for unlimited progression:
- **Base concepts**: 20 physics + 20 math = 40 concepts
- **Procedural generation**: Infinite advanced concepts
- **Era scaling**: Exponential requirements (2^n)
- **Bonus scaling**: Multiplicative bonuses continue growing
- **No hardcoded limits**: System supports Era 1000+

## Performance Considerations

- Discovery checks are O(n) per agent per tick
- Caching of unlocked concept IDs in Sets for O(1) lookup
- Lazy generation of advanced concepts
- Era logs stored separately to avoid memory bloat

## Future Extensions

Potential additions:
- **Technology Trees**: Specific tech branches unlocked by concepts
- **Specialized Fields**: Biology, Chemistry, Engineering, etc.
- **Collaborative Discovery**: Multiple agents contributing to breakthroughs
- **Discovery Competition**: Races between lineages
- **Scientific Publications**: Knowledge sharing mechanics
- **Research Focus**: Agents can specialize in specific fields

## Benefits to Gameplay

1. **Long-term Progression**: Unlimited advancement keeps simulation interesting
2. **Emergent Complexity**: Agents genuinely become smarter over time
3. **Historical Tracking**: See civilization's complete scientific journey
4. **Strategic Depth**: Science bonuses create optimization opportunities
5. **Replayability**: Different scientific paths each playthrough
6. **Educational**: Real physics/math concepts accurately represented

---

**Status**: Ready for integration  
**Dependencies**: types, agent system  
**Optional**: Can be disabled for classic mode
