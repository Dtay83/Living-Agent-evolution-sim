/**
 * Genetics module for the Living Agent Evolution Simulation
 * Handles gene creation, mutation, and evolution
 */

import type { Genes, Agent, Cell } from '../types';
import { CONFIG, GRID_WIDTH, GRID_HEIGHT, INITIAL_AGENTS } from '../core/config';
import { randomInt, randomTraitId, mutateValue } from '../utils';

/**
 * Generates random genes for a new agent
 * @returns Randomly initialized genetic traits
 */
export function createRandomGenes(): Genes {
  return {
    foodPreference: CONFIG.genes.foodPreference.min + Math.random() * (CONFIG.genes.foodPreference.max - CONFIG.genes.foodPreference.min),
    exploration: CONFIG.genes.exploration.min + Math.random() * (CONFIG.genes.exploration.max - CONFIG.genes.exploration.min),
    reproductionThreshold: CONFIG.genes.reproductionThreshold.min + Math.random() * (CONFIG.genes.reproductionThreshold.max - CONFIG.genes.reproductionThreshold.min),
    mutationRate: CONFIG.genes.mutationRate.min + Math.random() * (CONFIG.genes.mutationRate.max - CONFIG.genes.mutationRate.min),
    traitId: randomTraitId(),
    // Learning & invention genes
    curiosity: CONFIG.genes.curiosity.min + Math.random() * (CONFIG.genes.curiosity.max - CONFIG.genes.curiosity.min),
    social: CONFIG.genes.social.min + Math.random() * (CONFIG.genes.social.max - CONFIG.genes.social.min),
    creativity: CONFIG.genes.creativity.min + Math.random() * (CONFIG.genes.creativity.max - CONFIG.genes.creativity.min),
    patience: CONFIG.genes.patience.min + Math.random() * (CONFIG.genes.patience.max - CONFIG.genes.patience.min),
  };
}

/**
 * Mutates parent genes to create offspring genes
 * Each gene has a chance to mutate based on the parent's mutation rate
 * @param parent - Parent genes to mutate
 * @returns Mutated genes for offspring
 */
export function mutateGenes(parent: Genes): Genes {
  const mutationRate = parent.mutationRate;

  const foodPreference = mutateValue(
    parent.foodPreference,
    mutationRate,
    CONFIG.genes.mutation.foodPreferenceMagnitude,
    0.0,
    1.0
  );

  const exploration = mutateValue(
    parent.exploration,
    mutationRate,
    CONFIG.genes.mutation.explorationMagnitude,
    0.0,
    1.0
  );

  const reproductionThreshold = mutateValue(
    parent.reproductionThreshold,
    mutationRate,
    CONFIG.genes.mutation.reproductionThresholdMagnitude,
    CONFIG.genes.mutation.reproductionThresholdMin,
    CONFIG.genes.mutation.reproductionThresholdMax
  );

  // mutationRate itself can evolve
  const newMutationRate = mutateValue(
    parent.mutationRate,
    mutationRate,
    CONFIG.genes.mutation.mutationRateMagnitude,
    CONFIG.genes.mutation.mutationRateEvolveMin,
    CONFIG.genes.mutation.mutationRateEvolveMax
  );
  // Mutate learning & invention genes - UNCAPPED for unlimited evolution
  const curiosity = mutateValue(
    parent.curiosity,
    mutationRate,
    CONFIG.genes.mutation.curiosityMagnitude,
    0.0,
    Infinity  // NO CAP - curiosity can grow without limit
  );

  const social = mutateValue(
    parent.social,
    mutationRate,
    CONFIG.genes.mutation.socialMagnitude,
    0.0,
    Infinity  // NO CAP - social can grow without limit
  );

  const creativity = mutateValue(
    parent.creativity,
    mutationRate,
    CONFIG.genes.mutation.creativityMagnitude,
    0.0,
    Infinity  // NO CAP - creativity can grow without limit
  );

  const patience = mutateValue(
    parent.patience,
    mutationRate,
    CONFIG.genes.mutation.patienceMagnitude,
    0.0,
    Infinity  // NO CAP - patience can grow without limit
  );

  // Sometimes spawn a totally new traitId => random trait generation
  const traitId =
    Math.random() < CONFIG.genes.mutation.newTraitChance ? randomTraitId() : parent.traitId;

  return {
    foodPreference,
    exploration,
    reproductionThreshold,
    mutationRate: newMutationRate,
    traitId,
    curiosity,
    social,
    creativity,
    patience,
  };
}

/**
 * Creates the initial population of agents
 * @param grid - Current grid state
 * @returns Array of initial agents
 */
export function createInitialAgents(grid: Cell[][]): Agent[] {
  const agents: Agent[] = [];
  let idCounter = 1;
  const taken: Set<string> = new Set();

  while (agents.length < INITIAL_AGENTS) {
    const y = randomInt(GRID_HEIGHT);
    const x = randomInt(GRID_WIDTH);
    const key = `${x},${y}`;
    if (taken.has(key) || grid[y][x].food) continue;
    taken.add(key);
    agents.push({
      id: idCounter++,
      x,
      y,
      energy: CONFIG.energy.initialMin + randomInt(CONFIG.energy.initialMax - CONFIG.energy.initialMin),
      genes: createRandomGenes(),
      memory: { qTable: {} },
      lastRule: "none",
      inventions: [],
      inventionPoints: 0
    });
  }
  return agents;
}
