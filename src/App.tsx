// Contact: Name: dtay83 <dartey.banahene@gmail.com>
import React, { useState, useMemo, useEffect, useRef, useCallback } from "react";

type Direction = "up" | "down" | "left" | "right" | "stay";

interface Genes {
  foodPreference: number;        // 0–1: prioritize food when hungry
  exploration: number;           // 0–1: how often they wander
  reproductionThreshold: number; // energia needed to reproduce
  mutationRate: number;          // 0–1: chance each gene mutates
  traitId: number;               // lineage / random trait marker
}

interface Memory {
  qTable: Record<string, number>; // RL value table: (state|action) -> Q
}

interface Agent {
  id: number;
  x: number;
  y: number;
  energy: number; // "energia"
  genes: Genes;
  memory: Memory;
  lastRule?: string;
}

interface Cell {
  food: boolean;
  agentId?: number;
}

interface HistoryPoint {
  tick: number;
  totalAgents: number;
  byTrait: Record<number, number>;
}

interface WorldState {
  grid: Cell[][];
  agents: Agent[];
  tick: number;
  history: HistoryPoint[];
}

// Grid + sim settings
const GRID_WIDTH = 16;
const GRID_HEIGHT = 10;
const INITIAL_AGENTS = 6;
const INITIAL_FOOD = 18;

// RL hyperparameters
const ALPHA = 0.3;   // learning rate
const GAMMA = 0.9;   // discount factor
const EPSILON = 0.2; // exploration probability

function createEmptyGrid(): Cell[][] {
  return Array.from({ length: GRID_HEIGHT }, () =>
    Array.from({ length: GRID_WIDTH }, () => ({ food: false } as Cell))
  );
}

function randomInt(max: number) {
  return Math.floor(Math.random() * max);
}

function placeRandomFood(grid: Cell[][], count: number): Cell[][] {
  const copy = grid.map(row => row.map(cell => ({ ...cell })));
  let placed = 0;
  let safety = 0;
  while (placed < count && safety < 2000) {
    safety++;
    const y = randomInt(GRID_HEIGHT);
    const x = randomInt(GRID_WIDTH);
    if (!copy[y][x].food && copy[y][x].agentId === undefined) {
      copy[y][x].food = true;
      placed++;
    }
  }
  return copy;
}

function randomTraitId(): number {
  return randomInt(999999);
}

// Initial genes generator
function createRandomGenes(): Genes {
  return {
    foodPreference: 0.6 + Math.random() * 0.4,        // 0.6–1.0
    exploration: 0.3 + Math.random() * 0.5,           // 0.3–0.8
    reproductionThreshold: 15 + Math.random() * 8,    // 15–23 energia
    mutationRate: 0.1 + Math.random() * 0.2,          // 0.1–0.3
    traitId: randomTraitId()
  };
}

function createInitialAgents(grid: Cell[][]): Agent[] {
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
      energy: 10 + randomInt(6), // 10–15
      genes: createRandomGenes(),
      memory: { qTable: {} },
      lastRule: "none"
    });
  }
  return agents;
}

// Mutation helper: slightly nudge a value with probability
function mutateValue(
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

// Genetic memória + random trait generation
function mutateGenes(parent: Genes): Genes {
  const mutationRate = parent.mutationRate;

  const foodPreference = mutateValue(
    parent.foodPreference,
    mutationRate,
    0.15,
    0.0,
    1.0
  );

  const exploration = mutateValue(
    parent.exploration,
    mutationRate,
    0.2,
    0.0,
    1.0
  );

  const reproductionThreshold = mutateValue(
    parent.reproductionThreshold,
    mutationRate,
    3,
    8,
    30
  );

  // mutationRate itself can evolve
  const newMutationRate = mutateValue(
    parent.mutationRate,
    mutationRate,
    0.05,
    0.01,
    0.6
  );

  // sometimes spawn a totally new traitId => random trait generation
  const traitId =
    Math.random() < 0.08 ? randomTraitId() : parent.traitId; // 8% chance of "new family"

  return {
    foodPreference,
    exploration,
    reproductionThreshold,
    mutationRate: newMutationRate,
    traitId
  };
}

// --- RL helpers ---

function getStateKey(agent: Agent, grid: Cell[][]): string {
  const { x, y, energy } = agent;

  let level: "low" | "mid" | "high";
  if (energy <= 6) level = "low";
  else if (energy <= 14) level = "mid";
  else level = "high";

  const neighbors = [
    { x, y: y - 1 },
    { x, y: y + 1 },
    { x: x - 1, y },
    { x: x + 1, y }
  ].filter(
    p => p.x >= 0 && p.x < GRID_WIDTH && p.y >= 0 && p.y < GRID_HEIGHT
  );

  const foodNearby = neighbors.some(p => grid[p.y][p.x].food);

  return `${level}_${foodNearby ? 1 : 0}`;
}

function qKey(stateKey: string, action: Direction): string {
  return `${stateKey}|${action}`;
}

function getQ(qTable: Record<string, number>, stateKey: string, action: Direction): number {
  return qTable[qKey(stateKey, action)] ?? 0;
}

function setQ(
  qTable: Record<string, number>,
  stateKey: string,
  action: Direction,
  value: number
): Record<string, number> {
  return { ...qTable, [qKey(stateKey, action)]: value };
}

const ALL_ACTIONS: Direction[] = ["up", "down", "left", "right", "stay"];

function bestActionAndValue(
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

function chooseAction(
  qTable: Record<string, number>,
  stateKey: string
): Direction {
  if (Math.random() < EPSILON) {
    return ALL_ACTIONS[randomInt(ALL_ACTIONS.length)];
  }
  return bestActionAndValue(qTable, stateKey).action;
}

// Decision logic: mix hard survival rule with RL exploration
function decideMove(
  agent: Agent,
  grid: Cell[][],
  stateKey: string
): { dir: Direction; rule: string; action: Direction } {
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

  const hungryThreshold = 6 * genes.foodPreference;

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

// Apply one simulation step, including reproduction + mutation + RL updates
function stepWorld(
  agents: Agent[],
  grid: Cell[][]
): { agents: Agent[]; grid: Cell[][]; log: string[] } {
  const newGrid: Cell[][] = grid.map(row =>
    row.map(cell => ({ ...cell, agentId: undefined }))
  );

  const logs: string[] = [];
  const updatedAgents: Agent[] = [];

  let nextId = agents.reduce((max, a) => Math.max(max, a.id), 0) + 1;

  for (const agent of agents) {
    if (agent.energy <= 0) continue;

    // RL state before move
    const stateKey = getStateKey(agent, grid);
    const decision = decideMove(agent, grid, stateKey);

    let newX = agent.x;
    let newY = agent.y;

    if (decision.dir === "up") newY--;
    if (decision.dir === "down") newY++;
    if (decision.dir === "left") newX--;
    if (decision.dir === "right") newX++;

    newX = Math.max(0, Math.min(GRID_WIDTH - 1, newX));
    newY = Math.max(0, Math.min(GRID_HEIGHT - 1, newY));

    let newEnergy = agent.energy - 1; // base time cost
    const cell = newGrid[newY][newX];
    let ateFood = false;

    if (cell.food) {
      ateFood = true;
      cell.food = false;
      newEnergy += 5;
    }

    let reward = -1;
    if (ateFood) reward += 5;

    let parentAgent: Agent = {
      ...agent,
      x: newX,
      y: newY,
      energy: newEnergy,
      lastRule: decision.rule
    };

    // REPRODUCTION
    const reproThreshold = parentAgent.genes.reproductionThreshold;
    let reproduced = false;

    if (parentAgent.energy > reproThreshold) {
      const neighborSpots = [
        { x: newX, y: newY - 1 },
        { x: newX, y: newY + 1 },
        { x: newX - 1, y: newY },
        { x: newX + 1, y: newY }
      ].filter(
        p =>
          p.x >= 0 &&
          p.x < GRID_WIDTH &&
          p.y >= 0 &&
          p.y < GRID_HEIGHT &&
          newGrid[p.y][p.x].agentId === undefined
      );

      if (neighborSpots.length > 0) {
        const spot = neighborSpots[randomInt(neighborSpots.length)];

        const childEnergy = Math.floor(parentAgent.energy / 2);
        parentAgent = { ...parentAgent, energy: parentAgent.energy - childEnergy };

        const childGenes = mutateGenes(parentAgent.genes);
        const child: Agent = {
          id: nextId++,
          x: spot.x,
          y: spot.y,
          energy: childEnergy,
          genes: childGenes,
          memory: { qTable: {} },
          lastRule: 'Born (memória genética + "Mutation")'
        };

        newGrid[spot.y][spot.x].agentId = child.id;
        updatedAgents.push(child);

        reward += 2;
        reproduced = true;

        logs.push(
          `Agent ${parentAgent.id} reproduced: child ${child.id} at (${spot.x},${spot.y}) with traitId ${child.genes.traitId}, energia ${childEnergy}`
        );
      }
    }

    // RL UPDATE
    const newStateKey = getStateKey(
      { ...parentAgent, x: newX, y: newY },
      newGrid
    );
    const oldQ = getQ(parentAgent.memory.qTable, stateKey, decision.action);
    const bestNext = bestActionAndValue(parentAgent.memory.qTable, newStateKey).value;
    const updatedQ =
      (1 - ALPHA) * oldQ + ALPHA * (reward + GAMMA * bestNext);
    const newQTable = setQ(
      parentAgent.memory.qTable,
      stateKey,
      decision.action,
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
      reward -= 5;
      logs.push(
        `Agent ${agent.id} ran out of energia at (${newX},${newY}) and was removed.`
      );
    }
  }

  // Spawn new food
  if (Math.random() < 0.6) {
    placeRandomFood(newGrid, 1);
  }

  return { agents: updatedAgents, grid: newGrid, log: logs };
}

// Color by traitId (lineage): simple hash → hue
function colorForTrait(traitId: number): string {
  const hue = traitId % 360;
  return `hsl(${hue}, 70%, 55%)`;
}

// Simple line chart for population over time
const PopulationChart: React.FC<{ history: HistoryPoint[] }> = ({ history }) => {
  if (history.length < 2) return <p>Not enough data for chart yet.</p>;

  const width = 360;
  const height = 120;
  const maxPop = Math.max(...history.map(h => h.totalAgents), 1);
  const points = history.map((h, idx) => {
    const x = (idx / Math.max(history.length - 1, 1)) * (width - 20) + 10;
    const y = height - 10 - (h.totalAgents / maxPop) * (height - 20);
    return { x, y };
  });

  return (
    <svg width={width} height={height} style={{ background: "#0e1528", borderRadius: 8 }}>
      <polyline
        fill="none"
        stroke="#4caf50"
        strokeWidth={2}
        points={points.map(p => `${p.x},${p.y}`).join(" ")}
      />
      <line
        x1={10}
        y1={height - 10}
        x2={width - 10}
        y2={height - 10}
        stroke="#555"
        strokeWidth={1}
      />
      <line x1={10} y1={10} x2={10} y2={height - 10} stroke="#555" strokeWidth={1} />
      <text x={width - 60} y={height - 16} fontSize={10} fill="#ccc">
        ticks
      </text>
      <text x={14} y={18} fontSize={10} fill="#ccc">
        pop
      </text>
    </svg>
  );
};

// Bar chart for trait distribution
const TraitChart: React.FC<{
  last: HistoryPoint | null;
  watchedTraitId: number | null;
  onSelectTrait: (traitId: number) => void;
}> = ({ last, watchedTraitId, onSelectTrait }) => {
  if (!last) return <p>No trait data yet.</p>;
  const entries = Object.entries(last.byTrait);
  if (entries.length === 0) return <p>No agentes alive.</p>;

  const sorted = entries.sort((a, b) => b[1] - a[1]).slice(0, 5);
  const maxCount = Math.max(...sorted.map(([, c]) => c), 1);

  return (
    <div style={{ display: "flex", alignItems: "flex-end", gap: 8, height: 120 }}>
      {sorted.map(([traitStr, count]) => {
        const traitId = Number(traitStr);
        const heightRatio = count / maxCount;
        const isWatched = watchedTraitId === traitId;
        return (
          <div
            key={traitStr}
            style={{ textAlign: "center", cursor: "pointer" }}
            onClick={() => onSelectTrait(traitId)}
            title={`Click to watch traitId ${traitId}`}
          >
            <div
              style={{
                width: 30,
                height: heightRatio * 90,
                background: colorForTrait(traitId),
                borderRadius: 4,
                marginBottom: 4,
                outline: isWatched ? "2px solid #ffeb3b" : "none"
              }}
            />
            <div style={{ fontSize: 9 }}>id {traitStr.slice(0, 4)}</div>
            <div style={{ fontSize: 9 }}>{count}</div>
          </div>
        );
      })}
    </div>
  );
};

// Download world state as JSON ("guardar"/"speichere")
function downloadWorld(state: WorldState) {
  const blob = new Blob([JSON.stringify(state)], {
    type: "application/json"
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `world_tick_${state.tick}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

const App: React.FC = () => {
  const [grid, setGrid] = useState<Cell[][]>(() => {
    const empty = createEmptyGrid();
    const withFood = placeRandomFood(empty, INITIAL_FOOD);
    return withFood;
  });

  const [agents, setAgents] = useState<Agent[]>(() => {
    const baseGrid = createEmptyGrid();
    return createInitialAgents(baseGrid);
  });

  const [log, setLog] = useState<string[]>([]);
  const [selectedAgentId, setSelectedAgentId] = useState<number | null>(null);
  const [tick, setTick] = useState(0);
  const [history, setHistory] = useState<HistoryPoint[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [speedMs, setSpeedMs] = useState(400);
  const [watchedTraitId, setWatchedTraitId] = useState<number | null>(null);
  const [showStartupModal, setShowStartupModal] = useState(true);

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const tickRef = useRef(0);
  tickRef.current = tick;

  const renderedGrid = useMemo(() => {
    const copy = grid.map(row => row.map(cell => ({ ...cell, agentId: undefined })));
    for (const agent of agents) {
      if (
        agent.x >= 0 &&
        agent.x < GRID_WIDTH &&
        agent.y >= 0 &&
        agent.y < GRID_HEIGHT
      ) {
        copy[agent.y][agent.x].agentId = agent.id;
      }
    }
    return copy;
  }, [grid, agents]);

  const selectedAgent = useMemo(
    () => agents.find(a => a.id === selectedAgentId) || null,
    [agents, selectedAgentId]
  );

  const pushHistory = useCallback((newAgents: Agent[], newTick: number) => {
    const byTrait: Record<number, number> = {};
    for (const a of newAgents) {
      byTrait[a.genes.traitId] = (byTrait[a.genes.traitId] || 0) + 1;
    }
    const point: HistoryPoint = {
      tick: newTick,
      totalAgents: newAgents.length,
      byTrait
    };
    setHistory(prev => [...prev, point].slice(-60));
  }, []);

  const handleStep = useCallback(() => {
    const { agents: newAgents, grid: newGrid, log: newLog } = stepWorld(
      agents,
      renderedGrid
    );
    const newTick = tickRef.current + 1;
    setTick(newTick);
    setAgents(newAgents);
    setGrid(newGrid);
    setLog(prev => [...newLog, ...prev].slice(0, 80));
    pushHistory(newAgents, newTick);
  }, [agents, renderedGrid, pushHistory]);

  const handleReset = () => {
    const empty = createEmptyGrid();
    const withFood = placeRandomFood(empty, INITIAL_FOOD);
    const newAgents = createInitialAgents(withFood);
    setGrid(withFood);
    setAgents(newAgents);
    setLog([]);
    setSelectedAgentId(null);
    setTick(0);
    setHistory([]);
    setIsRunning(false);
    setWatchedTraitId(null);
  };

  // Auto-run interval
  useEffect(() => {
    if (!isRunning) return;
    const id = setInterval(() => {
      handleStep();
    }, speedMs);
    return () => clearInterval(id);
  }, [isRunning, speedMs, handleStep]);

  const lastHistory = history.length > 0 ? history[history.length - 1] : null;

  // Startup modal: ask to start new or load
  const handleStartupNew = () => {
    setShowStartupModal(false);
  };

  const handleStartupLoad = () => {
    setShowStartupModal(false);
    fileInputRef.current?.click();
  };

  const handleFileLoad = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const text = await file.text();
    const parsed = JSON.parse(text) as WorldState;
    setGrid(parsed.grid);
    setAgents(parsed.agents);
    setTick(parsed.tick);
    setHistory(parsed.history);
    setLog([]);
    setSelectedAgentId(null);
    setIsRunning(false);
    setWatchedTraitId(null);
  };

  return (
    <div
      style={{
        display: "flex",
        height: "100vh",
        fontFamily: "system-ui, sans-serif",
        background: "#050814",
        color: "#f4f4f4",
        padding: "12px",
        boxSizing: "border-box",
        position: "relative"
      }}
    >
      {/* Startup modal */}
      {showStartupModal && (
        <div
          style={{
            position: "absolute",
            inset: 0,
            background: "rgba(0,0,0,0.7)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 10
          }}
        >
          <div
            style={{
              background: "#151a30",
              padding: "20px",
              borderRadius: 10,
              border: "1px solid #333",
              maxWidth: 360
            }}
          >
            <h3>Welcome to the world</h3>
            <p style={{ fontSize: 14, opacity: 0.9 }}>
              Start a new "simulação"/world, or load a previously "guardada"/saved world
              (JSON file).
            </p>
            <div style={{ marginTop: 12, display: "flex", gap: 8 }}>
              <button onClick={handleStartupNew}>New world</button>
              <button onClick={handleStartupLoad}>Load from file</button>
            </div>
          </div>
        </div>
      )}

      {/* Hidden file input for load */}
      <input
        type="file"
        accept="application/json"
        style={{ display: "none" }}
        ref={fileInputRef}
        onChange={handleFileLoad}
      />

      {/* LEFT: World grid */}
      <div style={{ flex: "0 0 auto", marginRight: "16px" }}>
        <h2>NPC World – "memória genética" + RL</h2>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: `repeat(${GRID_WIDTH}, 22px)`,
            gridTemplateRows: `repeat(${GRID_HEIGHT}, 22px)`,
            gap: "2px",
            border: "1px solid #444",
            padding: "4px",
            background: "#151a30"
          }}
        >
          {renderedGrid.map((row, y) =>
            row.map((cell, x) => {
              const agent = agents.find(a => a.id === cell.agentId);
              const isSelected = selectedAgentId === agent?.id;
              const isWatched =
                agent && watchedTraitId !== null && agent.genes.traitId === watchedTraitId;

              let bg = "#1f2640";
              if (cell.food) bg = "#2c9c3f";
              if (agent) bg = colorForTrait(agent.genes.traitId);
              if (cell.food && agent) bg = "#ffb300";

              return (
                <div
                  key={`${x}-${y}`}
                  onClick={() => agent && setSelectedAgentId(agent.id)}
                  style={{
                    width: "22px",
                    height: "22px",
                    borderRadius: "4px",
                    background: bg,
                    border: isSelected
                      ? "2px solid #ffeb3b"
                      : isWatched
                      ? "2px solid #ff5252"
                      : "1px solid #333",
                    boxSizing: "border-box",
                    cursor: agent ? "pointer" : "default"
                  }}
                  title={
                    agent
                      ? `Agent ${agent.id} – energia ${agent.energy}, traitId ${agent.genes.traitId}`
                      : cell.food
                      ? "Food"
                      : ""
                  }
                />
              );
            })
          )}
        </div>

        <div style={{ marginTop: "12px" }}>
          <button onClick={handleStep} style={{ marginRight: "8px" }}>
            Step
          </button>
          <button
            onClick={() => setIsRunning(r => !r)}
            style={{ marginRight: "8px" }}
          >
            {isRunning ? "Pause" : "Play"}
          </button>
          <button onClick={handleReset}>Reset World</button>
          <button
            onClick={() => downloadWorld({ grid, agents, tick, history })}
            style={{ marginLeft: 8 }}
          >
            Save World (JSON)
          </button>
          <button
            onClick={() => fileInputRef.current?.click()}
            style={{ marginLeft: 8 }}
          >
            Load World
          </button>
          <div style={{ marginTop: 8, fontSize: 12 }}>
            Speed:{" "}
            <input
              type="range"
              min={100}
              max={1200}
              step={100}
              value={speedMs}
              onChange={e => setSpeedMs(Number(e.target.value))}
            />{" "}
            {speedMs} ms/tick
          </div>
          <span style={{ marginLeft: 12, fontSize: 12, opacity: 0.8 }}>
            Tick: {tick} | Agentes: {agents.length}
          </span>
        </div>
      </div>

      {/* RIGHT: Side panel */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column" }}>
        {/* Agent inspector */}
        <div
          style={{
            marginBottom: "12px",
            padding: "10px",
            background: "#151a30",
            borderRadius: "8px",
            border: "1px solid #333"
          }}
        >
          <h3>Agent Inspector – "agente" / "Figur"</h3>
          {selectedAgent ? (
            <>
              <p>
                <strong>ID:</strong> {selectedAgent.id}
              </p>
              <p>
                <strong>Position:</strong> ({selectedAgent.x},{selectedAgent.y})
              </p>
              <p>
                <strong>Energia:</strong> {selectedAgent.energy}
              </p>
              <p>
                <strong>Last Rule ("Regra" / "Regle"):</strong>{" "}
                {selectedAgent.lastRule}
              </p>
              <p>
                <strong>Genes:</strong>
                <br />
                foodPref={selectedAgent.genes.foodPreference.toFixed(2)},{" "}
                explore={selectedAgent.genes.exploration.toFixed(2)},{" "}
                reproThresh={selectedAgent.genes.reproductionThreshold.toFixed(1)},{" "}
                mutationRate={selectedAgent.genes.mutationRate.toFixed(2)}
                <br />
                traitId={selectedAgent.genes.traitId}
              </p>
            </>
          ) : (
            <p>Click an agente in the grid to inspect its "memória genética".</p>
          )}
        </div>

        {/* Rules + RL + evolution description */}
        <div
          style={{
            marginBottom: "12px",
            padding: "10px",
            background: "#151a30",
            borderRadius: "8px",
            border: "1px solid #333"
          }}
        >
          <h3>Behavior, RL & Evolution</h3>
          <ul style={{ fontSize: "0.9em" }}>
            <li>
              <strong>Hard Rule:</strong> if hungry and food is adjacent, move
              toward food (survival).
            </li>
            <li>
              <strong>RL Policy:</strong> otherwise, a learned policy chooses
              up/down/left/right/stay to maximize future reward.
            </li>
            <li>
              <strong>Reproduction:</strong> if energia &gt; reproductionThreshold,
              agente may split energia with a child, whose genes are mutated.
            </li>
          </ul>
        </div>

        {/* Charts */}
        <div
          style={{
            marginBottom: "12px",
            padding: "10px",
            background: "#151a30",
            borderRadius: "8px",
            border: "1px solid #333"
          }}
        >
          <h3>Population Over Time – "população"</h3>
          <PopulationChart history={history} />
        </div>

        <div
          style={{
            marginBottom: "12px",
            padding: "10px",
            background: "#151a30",
            borderRadius: "8px",
            border: "1px solid #333"
          }}
        >
          <h3>
            Trait Distribution – "famílias" / "Stämme"
            {watchedTraitId !== null && (
              <span style={{ fontSize: 12, marginLeft: 8 }}>
                Watching traitId {watchedTraitId}
              </span>
            )}
          </h3>
          <TraitChart
            last={lastHistory}
            watchedTraitId={watchedTraitId}
            onSelectTrait={tid => setWatchedTraitId(tid)}
          />
        </div>

        {/* Log */}
        <div
          style={{
            flex: 1,
            padding: "10px",
            background: "#151a30",
            borderRadius: "8px",
            border: "1px solid #333",
            overflowY: "auto"
          }}
        >
          <h3>Action Log – "histórico" / "Protokoll"</h3>
          {log.length === 0 ? (
            <p>No steps yet. Press "Step" or "Play" to advance the world.</p>
          ) : (
            <ul style={{ paddingLeft: "18px" }}>
              {log.map((entry, i) => (
                <li key={i} style={{ fontSize: "0.8em" }}>
                  {entry}
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
};

export default App;