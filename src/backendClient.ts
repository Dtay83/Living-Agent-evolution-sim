export type BackendHealth = {
  ok: boolean;
  service: string;
  version?: string;
};

export type BackendGenome = {
  curiosity: number;
  compression_bias: number;
  symbolic_precision: number;
  sociality: number;
  mutation_rate: number;
  energy_efficiency: number;
};

export type BackendAgent = {
  agent_id: string;
  energy: number;
  age: number;
  genome: BackendGenome;
};

export type BackendMetrics = {
  entropy: number;
  compression_delta: number;
  novelty: number;
  information_pressure: number;
};

export type BackendDiscovery = {
  discovery_id: string;
  world_id: string;
  tick: number;
  hypothesis: string;
  valid: boolean;
  feedback: string;
  residual: number | null;
  created_at: string;
};

export type BackendWorld = {
  world_id: string;
  seed: number;
  tick: number;
  agents: BackendAgent[];
  metrics: BackendMetrics;
  accepted_discoveries: BackendDiscovery[];
  created_at: string;
  updated_at: string;
};

// Vite proxies "/api" to the FastAPI server in dev (see vite.config.ts).
const BACKEND_URL: string =
  (import.meta as any).env?.VITE_BACKEND_URL ?? "/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BACKEND_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });

  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body?.detail) detail = String(body.detail);
    } catch {
      /* non-JSON error body */
    }
    throw new Error(`Backend request ${path} failed: ${detail}`);
  }

  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export async function getBackendHealth(): Promise<BackendHealth> {
  return request<BackendHealth>("/health");
}

export async function createWorld(options: {
  worldId?: string;
  seed?: number;
  agentCount?: number;
}): Promise<BackendWorld> {
  return request<BackendWorld>("/worlds", {
    method: "POST",
    body: JSON.stringify({
      world_id: options.worldId ?? null,
      seed: options.seed ?? 1,
      agent_count: options.agentCount ?? 12,
    }),
  });
}

export async function getWorld(worldId: string): Promise<BackendWorld> {
  return request<BackendWorld>(`/worlds/${encodeURIComponent(worldId)}`);
}

export async function tickWorld(worldId: string, steps = 1): Promise<BackendWorld> {
  return request<BackendWorld>(`/worlds/${encodeURIComponent(worldId)}/tick`, {
    method: "POST",
    body: JSON.stringify({ steps }),
  });
}

export async function listDiscoveries(worldId: string): Promise<BackendDiscovery[]> {
  return request<BackendDiscovery[]>(
    `/worlds/${encodeURIComponent(worldId)}/discoveries`
  );
}

const WORLD_STORAGE_KEY = "living-agent-world-id";

/** Reuse the persisted world for this browser, or create a new one. */
export async function resolveSessionWorld(
  seed: number,
  agentCount: number
): Promise<BackendWorld> {
  const existing = localStorage.getItem(WORLD_STORAGE_KEY);
  if (existing) {
    try {
      return await getWorld(existing);
    } catch {
      localStorage.removeItem(WORLD_STORAGE_KEY);
    }
  }
  const world = await createWorld({ seed, agentCount });
  localStorage.setItem(WORLD_STORAGE_KEY, world.world_id);
  return world;
}

export function forgetSessionWorld(): void {
  localStorage.removeItem(WORLD_STORAGE_KEY);
}
