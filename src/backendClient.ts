export type BackendHealth = {
  ok: boolean;
  service: string;
};

const BACKEND_URL = "http://127.0.0.1:8000";

export async function getBackendHealth(): Promise<BackendHealth> {
  const response = await fetch(`${BACKEND_URL}/health`);

  if (!response.ok) {
    throw new Error(`Backend health check failed: ${response.status}`);
  }

  return await response.json();
}
