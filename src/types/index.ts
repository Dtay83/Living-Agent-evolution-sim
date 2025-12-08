/**
 * Core type definitions for the Living Agent Evolution Simulation
 * Centralizes all interfaces and types used throughout the application
 */

export type Direction = "up" | "down" | "left" | "right" | "stay";

/**
 * Genetic traits that define an agent's behavior and capabilities
 */
export interface Genes {
  foodPreference: number;        // 0–1: prioritize food when hungry
  exploration: number;           // 0–1: how often they wander
  reproductionThreshold: number; // energia needed to reproduce
  mutationRate: number;          // 0–1: chance each gene mutates
  traitId: number;               // lineage / random trait marker
  
  // Learning & invention genes
  curiosity: number;             // 0–1: likelihood of discovering new things
  social: number;                // 0–1: ability to teach/learn from others
  creativity: number;            // 0–1: scales the quality and power of discovered inventions
  patience: number;              // 0–1: influences storage capacity and long-term benefits
}

/**
 * Q-Learning memory table for reinforcement learning
 */
export interface Memory {
  qTable: Record<string, number>; // RL value table: (state|action) -> Q
}

/**
 * Effect types that inventions can provide to agents
 */
export type InventionEffect = 
  | { type: 'energy_efficiency'; multiplier: number }
  | { type: 'food_detection_range'; range: number }
  | { type: 'reproduction_boost'; bonus: number }
  | { type: 'defense'; protection: number }
  | { type: 'storage'; capacity: number };

/**
 * An invention discovered by an agent
 */
export interface Invention {
  id: string;
  name: string;
  type: 'tool' | 'technique' | 'structure';
  effect: InventionEffect;
  discoveredAt: number;      // tick when invented
  discoveredBy: number;      // agent ID
  requirements: string[];    // prerequisites to use
  description?: string;      // human-readable description
}

/**
 * Event tracking when an invention is discovered
 */
export interface DiscoveryEvent {
  tick: number;
  agentId: number;
  invention: Invention;
}

/**
 * An autonomous agent in the simulation
 */
export interface Agent {
  id: number;
  x: number;
  y: number;
  energy: number; // "energia"
  genes: Genes;
  memory: Memory;
  lastRule?: string;
  inventions: Invention[];      // Things this agent has discovered
  inventionPoints: number;      // Accumulated creativity points for discovering new inventions
}

/**
 * A cell in the simulation grid
 */
export interface Cell {
  food: boolean;
  agentId?: number;
}

/**
 * Historical data point for tracking population over time
 */
export interface HistoryPoint {
  tick: number;
  totalAgents: number;
  byTrait: Record<number, number>;
}

/**
 * Complete state of the simulation world
 */
export interface WorldState {
  grid: Cell[][];
  agents: Agent[];
  tick: number;
  history: HistoryPoint[];
  discoveries: DiscoveryEvent[];
  gridWidth: number;   // Current grid width (can expand)
  gridHeight: number;  // Current grid height (can expand)
}

/**
 * Move decision from agent AI
 */
export interface MoveDecision {
  dir: Direction;
  rule: string;
  action: Direction;
}
