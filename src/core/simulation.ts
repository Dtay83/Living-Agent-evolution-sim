/**
 * Core Simulation Logic for the Living Agent Evolution Simulation
 * Handles the main simulation step: movement, reproduction, Q-learning updates
 */

import type { Agent, Cell, DiscoveryEvent } from '../types';
import { CONFIG, GRID_WIDTH, GRID_HEIGHT, ALPHA, GAMMA, setGridDimensions } from './config';
import { applyDirection, placeRandomFood, randomInt, expandGrid } from '../utils';
import { mutateGenes, createInitialAgents } from './genetics';
import { getStateKey, decideMove, getQ, setQ, bestActionAndValue } from './rl-system';
import { 
  checkForDiscovery, 
  getMovementCost, 
  getReproductionBonus 
} from '../invention-system';

/**
 * Execute one simulation step
 * Processes agent movement, reproduction, learning, and invention discovery
 * 
 * @param agents - Current agents in the simulation
 * @param grid - Current grid state
 * @param tick - Current simulation tick
 * @returns Updated agents, grid, action log, and discoveries
 */
export function stepWorld(
  agents: Agent[],
  grid: Cell[][],
  tick: number
): { agents: Agent[]; grid: Cell[][]; log: string[]; discoveries: DiscoveryEvent[]; gridExpanded: boolean } {
  let currentGrid = grid;
  let gridExpanded = false;
  const currentHeight = currentGrid.length;
  const currentWidth = currentGrid[0]?.length || 0;
  
  // Update global dimensions in case grid has changed
  setGridDimensions(currentWidth, currentHeight);
  
  const newGrid: Cell[][] = currentGrid.map(row =>
    row.map(cell => ({ ...cell, agentId: undefined }))
  );

  const logs: string[] = [];
  const updatedAgents: Agent[] = [];
  const discoveries: DiscoveryEvent[] = [];

  let nextId = agents.reduce((max, a) => Math.max(max, a.id), 0) + 1;

  /**
   * Collision handling: Track intended destinations
   * Prevents multiple agents from moving to the same cell
   */
  const destinationMap = new Map<string, number>(); // "x,y" -> agentId

  // Phase 1: Decide moves for all agents
  interface AgentMove {
    agent: Agent;
    newPos: { x: number; y: number };
    decision: { dir: string; rule: string; action: string };
    stateKey: string;
  }
  const agentMoves: AgentMove[] = [];

  for (const agent of agents) {
    if (agent.energy <= 0) continue;

    // RL state before move
    const stateKey = getStateKey(agent, grid);
    const decision = decideMove(agent, grid, stateKey);

    // Apply direction with bounds checking
    const newPos = applyDirection(agent.x, agent.y, decision.dir as any);

    agentMoves.push({ agent, newPos, decision, stateKey });
  }

  // Phase 2: Process moves with collision detection
  for (const { agent, newPos, decision, stateKey } of agentMoves) {
    const destKey = `${newPos.x},${newPos.y}`;
    
    // Track the final position (may change due to collision)
    let finalX = newPos.x;
    let finalY = newPos.y;
    
    // Check if another agent already claimed this destination
    if (destinationMap.has(destKey)) {
      // Collision detected - agent stays in place
      logs.push(
        `Agent ${agent.id} collision at (${newPos.x},${newPos.y}), stayed at (${agent.x},${agent.y})`
      );
      // Use original position
      finalX = agent.x;
      finalY = agent.y;
    } else {
      destinationMap.set(destKey, agent.id);
    }

    // Apply energy cost with invention effects
    const movementCost = getMovementCost(agent);
    let newEnergy = agent.energy - movementCost;
    const cell = newGrid[finalY][finalX];
    let ateFood = false;

    if (cell.food) {
      ateFood = true;
      cell.food = false;
      newEnergy += CONFIG.simulation.foodEnergyBonus;
    }

    let reward = -1;
    if (ateFood) reward += CONFIG.simulation.foodEnergyBonus;

    let parentAgent: Agent = {
      ...agent,
      x: finalX,
      y: finalY,
      energy: newEnergy,
      lastRule: decision.rule
    };

    // Check for invention discovery
    const discoveryResult = checkForDiscovery(parentAgent, tick);
    parentAgent = discoveryResult.updatedAgent; // Update with new invention points
    
    if (discoveryResult.invention) {
      parentAgent = {
        ...parentAgent,
        inventions: [...parentAgent.inventions, discoveryResult.invention]
      };
      discoveries.push({
        tick,
        agentId: parentAgent.id,
        invention: discoveryResult.invention
      });
      logs.push(
        `Agent ${parentAgent.id} discovered ${discoveryResult.invention.name}! (${discoveryResult.invention.description})`
      );
    }

    // REPRODUCTION
    const reproThreshold = parentAgent.genes.reproductionThreshold;
    let reproduced = false;

    // Apply reproduction bonus from inventions
    const reproBonus = getReproductionBonus(parentAgent);
    const effectiveEnergy = parentAgent.energy + reproBonus;

    if (effectiveEnergy > reproThreshold) {
      let neighborSpots = [
        { x: finalX, y: finalY - 1 },
        { x: finalX, y: finalY + 1 },
        { x: finalX - 1, y: finalY },
        { x: finalX + 1, y: finalY }
      ].filter(
        p =>
          p.x >= 0 &&
          p.x < currentWidth &&
          p.y >= 0 &&
          p.y < currentHeight &&
          newGrid[p.y][p.x].agentId === undefined &&
          !destinationMap.has(`${p.x},${p.y}`) // Also check collision map
      );

      // If no space available, try to expand the grid
      if (neighborSpots.length === 0 && 
          (currentWidth < CONFIG.grid.maxWidth || currentHeight < CONFIG.grid.maxHeight)) {
        
        // Determine expansion direction based on parent's position
        let expandDirection: 'right' | 'bottom' | 'both' = 'both';
        
        if (finalX >= currentWidth - 2 && currentWidth < CONFIG.grid.maxWidth) {
          expandDirection = 'right';
        } else if (finalY >= currentHeight - 2 && currentHeight < CONFIG.grid.maxHeight) {
          expandDirection = 'bottom';
        } else if (currentWidth < CONFIG.grid.maxWidth && currentHeight < CONFIG.grid.maxHeight) {
          expandDirection = 'both';
        }
        
        // Expand the grid
        const expandedGrid = expandGrid(newGrid, expandDirection, CONFIG.grid.expandBy);
        
        // Replace grid contents properly
        newGrid.splice(0, newGrid.length, ...expandedGrid);
        
        const newWidth = expandedGrid[0]?.length || currentWidth;
        const newHeight = expandedGrid.length;
        
        setGridDimensions(newWidth, newHeight);
        gridExpanded = true;
        
        logs.push(
          `Grid expanded to ${newWidth}×${newHeight} to accommodate population growth!`
        );
        
        // Recalculate neighbor spots with new grid size
        neighborSpots = [
          { x: finalX, y: finalY - 1 },
          { x: finalX, y: finalY + 1 },
          { x: finalX - 1, y: finalY },
          { x: finalX + 1, y: finalY }
        ].filter(
          p =>
            p.x >= 0 &&
            p.x < newWidth &&
            p.y >= 0 &&
            p.y < newHeight &&
            newGrid[p.y]?.[p.x]?.agentId === undefined &&
            !destinationMap.has(`${p.x},${p.y}`)
        );
      }

      if (neighborSpots.length > 0) {
        const spot = neighborSpots[randomInt(neighborSpots.length)];

        const childEnergy = Math.floor(parentAgent.energy / 2);
        parentAgent = { ...parentAgent, energy: parentAgent.energy - childEnergy };

        const childGenes = mutateGenes(parentAgent.genes);
        
        // Inheritance: children inherit parent's inventions based on social gene
        const inheritedInventions = parentAgent.inventions.filter(
          inv => Math.random() < parentAgent.genes.social * CONFIG.invention.inheritanceRate
        );
        
        // Children inherit a portion of parent's invention points
        const inheritedPoints = parentAgent.inventionPoints * parentAgent.genes.social * CONFIG.invention.pointInheritanceRate;
        
        const child: Agent = {
          id: nextId++,
          x: spot.x,
          y: spot.y,
          energy: childEnergy,
          genes: childGenes,
          memory: { qTable: {} },
          lastRule: 'Born (memória genética + "Mutation")',
          inventions: inheritedInventions.map(inv => ({
            ...inv,
            // Mark as inherited, not discovered by this agent
            discoveredBy: parentAgent.id,
          })),
          inventionPoints: inheritedPoints,
        };

        newGrid[spot.y][spot.x].agentId = child.id;
        destinationMap.set(`${spot.x},${spot.y}`, child.id); // Register child position
        updatedAgents.push(child);

        reward += CONFIG.simulation.reproductionReward;
        reproduced = true;

        const inheritMsg = inheritedInventions.length > 0 
          ? ` inherited ${inheritedInventions.length} inventions`
          : '';
        logs.push(
          `Agent ${parentAgent.id} reproduced: child ${child.id} at (${spot.x},${spot.y}) with traitId ${child.genes.traitId}, energia ${childEnergy}${inheritMsg}`
        );
      }
    }

    // RL UPDATE
    const newStateKey = getStateKey(
      { ...parentAgent, x: finalX, y: finalY },
      newGrid
    );
    const oldQ = getQ(parentAgent.memory.qTable, stateKey, decision.action as any);
    const bestNext = bestActionAndValue(parentAgent.memory.qTable, newStateKey).value;
    const updatedQ =
      (1 - ALPHA) * oldQ + ALPHA * (reward + GAMMA * bestNext);
    const newQTable = setQ(
      parentAgent.memory.qTable,
      stateKey,
      decision.action as any,
      updatedQ
    );

    parentAgent = { ...parentAgent, memory: { qTable: newQTable } };

    if (parentAgent.energy > 0) {
      updatedAgents.push(parentAgent);
      newGrid[parentAgent.y][parentAgent.x].agentId = parentAgent.id;

      logs.push(
        `Agent ${parentAgent.id} used ${decision.rule}, moved to (${parentAgent.x},${parentAgent.y})` +
          (ateFood ? " and ate food (+5 energia)" : "") +
          (reproduced ? " and reproduced (+2 reward)" : "") +
          `, energia now ${parentAgent.energy}, traitId=${parentAgent.genes.traitId}`
      );
    } else {
      reward -= CONFIG.simulation.deathPenalty;
      logs.push(
        `Agent ${agent.id} ran out of energia at (${finalX},${finalY}) and was removed.`
      );
    }
  }
  // Food spawning: scale with population so the ecosystem can sustain growth.
  const gridArea = (newGrid[0]?.length || 1) * newGrid.length;
  const foodOnGrid = newGrid.reduce(
    (sum, row) => sum + row.filter(c => c.food).length,
    0
  );
  const foodCap = Math.floor(gridArea * CONFIG.simulation.maxFoodOnGrid);
  const desiredSpawn = Math.ceil(
    CONFIG.simulation.foodSpawnCount +
      updatedAgents.length * CONFIG.simulation.foodPerAgent
  );
  const spawnCount = Math.max(0, Math.min(desiredSpawn, foodCap - foodOnGrid));

  let finalGrid = newGrid;
  if (spawnCount > 0) {
    finalGrid = placeRandomFood(newGrid, spawnCount);
  }

  // Extinction floor: reseed founders so the simulation keeps running.
  let finalAgents = updatedAgents;
  if (finalAgents.length < CONFIG.simulation.minPopulation) {
    const needed = CONFIG.simulation.minPopulation - finalAgents.length;
    const reseeded = createInitialAgents(finalGrid).slice(0, needed);
    let seedId = finalAgents.reduce((max, a) => Math.max(max, a.id), nextId);
    for (const seed of reseeded) {
      const colonist = { ...seed, id: ++seedId };
      if (finalGrid[colonist.y]?.[colonist.x]?.agentId === undefined) {
        finalGrid[colonist.y][colonist.x].agentId = colonist.id;
        finalAgents.push(colonist);
      }
    }
    logs.push(
      `Population collapsed - reseeded ${needed} founder agents to continue evolution.`
    );
  }

  return { agents: finalAgents, grid: finalGrid, log: logs, discoveries, gridExpanded };
}

/**
 * Initialize a new world state
 * @returns Initial grid, agents, and dimensions
 */
export function initializeWorld(): { grid: Cell[][]; agents: Agent[]; gridWidth: number; gridHeight: number } {
  const width = CONFIG.grid.initialWidth;
  const height = CONFIG.grid.initialHeight;
  
  setGridDimensions(width, height);
  
  const grid = placeRandomFood(
    Array.from({ length: height }, () =>
      Array.from({ length: width }, () => ({ food: false }))
    ),
    CONFIG.simulation.initialFood
  );
  
  const agents = createInitialAgents(grid);
  
  return { grid, agents, gridWidth: width, gridHeight: height };
}
