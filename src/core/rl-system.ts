/**
 * Reinforcement Learning (Q-Learning) System
 * Implements value-based learning for agent decision making
 */

import type { Agent, Cell, Direction, MoveDecision } from '../types';
import { CONFIG, GRID_WIDTH, GRID_HEIGHT, EPSILON } from '../core/config';
import { randomInt } from '../utils';

/**
 * All possible actions an agent can take
 */
export const ALL_ACTIONS: Direction[] = ["up", "down", "left", "right", "stay"];

/**
 * Generate a state key representing the agent's current situation
 * Improved representation with directional food information
 * @param agent - Current agent
 * @param grid - Current grid state
 * @returns State key string
 */
export function getStateKey(agent: Agent, grid: Cell[][]): string {
  const { x, y, energy } = agent;

  let level: "low" | "mid" | "high";
  if (energy <= 6) level = "low";
  else if (energy <= 14) level = "mid";
  else level = "high";

  // Directional food detection for more informative state
  const foodUp = y > 0 && grid[y - 1][x].food ? 1 : 0;
  const foodDown = y < GRID_HEIGHT - 1 && grid[y + 1][x].food ? 1 : 0;
  const foodLeft = x > 0 && grid[y][x - 1].food ? 1 : 0;
  const foodRight = x < GRID_WIDTH - 1 && grid[y][x + 1].food ? 1 : 0;

  return `${level}_${foodUp}${foodDown}${foodLeft}${foodRight}`;
}

/**
 * Create Q-table key from state and action
 * @param stateKey - State identifier
 * @param action - Action to take
 * @returns Combined key for Q-table lookup
 */
export function qKey(stateKey: string, action: Direction): string {
  return `${stateKey}|${action}`;
}

/**
 * Get Q-value for a state-action pair
 * @param qTable - Q-learning value table
 * @param stateKey - Current state
 * @param action - Action to evaluate
 * @returns Q-value (defaults to 0 if not found)
 */
export function getQ(qTable: Record<string, number>, stateKey: string, action: Direction): number {
  return qTable[qKey(stateKey, action)] ?? 0;
}

/**
 * Set Q-value for a state-action pair (immutable)
 * @param qTable - Current Q-table
 * @param stateKey - State identifier
 * @param action - Action taken
 * @param value - New Q-value
 * @returns Updated Q-table
 */
export function setQ(
  qTable: Record<string, number>,
  stateKey: string,
  action: Direction,
  value: number
): Record<string, number> {
  return { ...qTable, [qKey(stateKey, action)]: value };
}

/**
 * Find the best action and its value for a given state
 * @param qTable - Q-learning value table
 * @param stateKey - Current state
 * @returns Best action and its Q-value
 */
export function bestActionAndValue(
  qTable: Record<string, number>,
  stateKey: string
): { action: Direction; value: number } {
  let bestAction: Direction = "stay";
  let bestValue = Number.NEGATIVE_INFINITY;
  for (const a of ALL_ACTIONS) {
    const v = getQ(qTable, stateKey, a);
    if (v > bestValue) {
      bestValue = v;
      bestAction = a;
    }
  }
  if (bestValue === Number.NEGATIVE_INFINITY) {
    return { action: "stay", value: 0 };
  }
  return { action: bestAction, value: bestValue };
}

/**
 * Choose an action using epsilon-greedy strategy
 * Balances exploration (random) and exploitation (best known)
 * @param qTable - Q-learning value table
 * @param stateKey - Current state
 * @returns Chosen action
 */
export function chooseAction(
  qTable: Record<string, number>,
  stateKey: string
): Direction {
  if (Math.random() < EPSILON) {
    return ALL_ACTIONS[randomInt(ALL_ACTIONS.length)];
  }
  return bestActionAndValue(qTable, stateKey).action;
}

/**
 * Decide agent's move combining hard-coded survival rules with RL
 * Hard rule: If hungry and food is adjacent, go for food
 * Otherwise: Use Q-learning policy
 * @param agent - Agent making decision
 * @param grid - Current grid state
 * @param stateKey - Pre-computed state key
 * @returns Move decision with direction and reasoning
 */
export function decideMove(
  agent: Agent,
  grid: Cell[][],
  stateKey: string
): MoveDecision {
  const { x, y, energy, genes } = agent;

  const neighbors: { x: number; y: number; dir: Direction }[] = [
    { x, y: y - 1, dir: "up" },
    { x, y: y + 1, dir: "down" },
    { x: x - 1, y, dir: "left" },
    { x: x + 1, y, dir: "right" }
  ].filter(
    p => p.x >= 0 && p.x < GRID_WIDTH && p.y >= 0 && p.y < GRID_HEIGHT
  );

  const foodNeighbors = neighbors.filter(n => grid[n.y][n.x].food);

  const hungryThreshold = CONFIG.energy.hungryCutoffRatio * genes.foodPreference;

  // HARD RULE: if hungry + food adjacent, go for food
  if (energy <= hungryThreshold && foodNeighbors.length > 0) {
    const choice = foodNeighbors[randomInt(foodNeighbors.length)];
    return {
      dir: choice.dir,
      rule: "Rule 1: seek food (hard survival)",
      action: choice.dir
    };
  }

  // RL-DRIVEN CHOICE
  const action = chooseAction(agent.memory.qTable, stateKey);
  let ruleDesc = "RL: learned policy";
  if (action === "stay") {
    ruleDesc = "RL: choose stay";
  }

  return { dir: action, rule: ruleDesc, action };
}
