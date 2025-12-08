/**
 * Utility functions for the Living Agent Evolution Simulation
 * Provides common helper functions used throughout the application
 */

import type { Direction, Cell } from '../types';
import { CONFIG, GRID_WIDTH, GRID_HEIGHT } from '../core/config';

/**
 * Direction movement deltas for grid navigation
 */
export const DIRECTION_DELTAS: Record<Direction, { dx: number; dy: number }> = {
  up: { dx: 0, dy: -1 },
  down: { dx: 0, dy: 1 },
  left: { dx: -1, dy: 0 },
  right: { dx: 1, dy: 0 },
  stay: { dx: 0, dy: 0 },
};

/**
 * Generates a random integer from 0 to max (exclusive)
 * @param max - Upper bound (exclusive)
 */
export function randomInt(max: number): number {
  return Math.floor(Math.random() * max);
}

/**
 * Generates a random trait ID for genetic lineage tracking
 */
export function randomTraitId(): number {
  return randomInt(999999);
}

/**
 * Apply direction movement with bounds checking
 * @param x - Current x position
 * @param y - Current y position
 * @param dir - Direction to move
 * @returns New position clamped to grid boundaries
 */
export function applyDirection(
  x: number,
  y: number,
  dir: Direction
): { x: number; y: number } {
  const delta = DIRECTION_DELTAS[dir];
  return {
    x: Math.max(0, Math.min(GRID_WIDTH - 1, x + delta.dx)),
    y: Math.max(0, Math.min(GRID_HEIGHT - 1, y + delta.dy)),
  };
}

/**
 * Creates an empty grid filled with empty cells
 * @param width - Grid width
 * @param height - Grid height
 */
export function createEmptyGrid(width?: number, height?: number): Cell[][] {
  const w = width || GRID_WIDTH;
  const h = height || GRID_HEIGHT;
  return Array.from({ length: h }, () =>
    Array.from({ length: w }, () => ({ food: false } as Cell))
  );
}

/**
 * Expands the grid by adding rows/columns when needed for reproduction
 * @param grid - Current grid
 * @param direction - Direction to expand ('right', 'bottom', or 'both')
 * @param expandBy - Number of cells to add
 * @returns Expanded grid
 */
export function expandGrid(
  grid: Cell[][], 
  direction: 'right' | 'bottom' | 'both',
  expandBy: number = 2
): Cell[][] {
  const currentHeight = grid.length;
  const currentWidth = grid[0]?.length || 0;
  
  if (direction === 'right' || direction === 'both') {
    // Add columns to the right
    const newGrid = grid.map(row => [
      ...row,
      ...Array.from({ length: expandBy }, () => ({ food: false } as Cell))
    ]);
    grid = newGrid;
  }
  
  if (direction === 'bottom' || direction === 'both') {
    // Add rows to the bottom
    const newWidth = grid[0]?.length || currentWidth;
    const newRows = Array.from({ length: expandBy }, () =>
      Array.from({ length: newWidth }, () => ({ food: false } as Cell))
    );
    grid = [...grid, ...newRows];
  }
  
  return grid;
}

/**
 * Places random food on the grid
 * @param grid - Current grid state
 * @param count - Number of food items to place
 * @returns New grid with food placed
 */
export function placeRandomFood(grid: Cell[][], count: number): Cell[][] {
  const copy = grid.map(row => row.map(cell => ({ ...cell })));
  const height = copy.length;
  const width = copy[0]?.length || 0;
  
  let placed = 0;
  let safety = 0;
  while (placed < count && safety < 2000) {
    safety++;
    const y = randomInt(height);
    const x = randomInt(width);
    if (!copy[y][x].food && copy[y][x].agentId === undefined) {
      copy[y][x].food = true;
      placed++;
    }
  }
  return copy;
}

/**
 * Generates a color based on trait ID for visualization
 * @param traitId - Genetic trait identifier
 * @returns RGB color string
 */
export function colorForTrait(traitId: number): string {
  const hue = (traitId * 137.508) % 360;
  return `hsl(${hue}, 70%, 50%)`;
}

/**
 * Mutates a value with given probability and magnitude
 * @param base - The base value to potentially mutate
 * @param mutationRate - Probability (0-1) that mutation occurs
 * @param magnitude - Maximum amount the value can change
 * @param min - Minimum allowed value after mutation
 * @param max - Maximum allowed value after mutation
 * @returns The potentially mutated value, clamped to [min, max]
 */
export function mutateValue(
  base: number,
  mutationRate: number,
  magnitude: number,
  min: number,
  max: number
): number {
  let value = base;
  if (Math.random() < mutationRate) {
    const delta = (Math.random() * 2 - 1) * magnitude;
    value = Math.min(max, Math.max(min, value + delta));
  }
  return value;
}

// Export data export utilities
export {
  exportInventionHistory,
  exportEvolutionData,
  exportScienceLogs,
  exportCompleteData,
} from './exportData';
