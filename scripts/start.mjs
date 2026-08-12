// Boots the FastAPI backend, waits until /health reports ok, then starts Vite.
import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const backendDir = path.join(root, "backend");
const port = Number(process.env.BACKEND_PORT ?? 8000);
const healthUrl = `http://127.0.0.1:${port}/health`;
const isWindows = process.platform === "win32";

const children = [];

function resolvePython() {
  const venv = isWindows
    ? path.join(backendDir, ".venv", "Scripts", "python.exe")
    : path.join(backendDir, ".venv", "bin", "python");
  if (existsSync(venv)) return venv;
  return process.env.PYTHON ?? (isWindows ? "py" : "python3");
}

function run(command, args, options) {
  const child = spawn(command, args, { stdio: "inherit", shell: isWindows, ...options });
  children.push(child);
  return child;
}

function shutdown(code) {
  for (const child of children) {
    if (!child.killed) child.kill();
  }
  process.exit(code);
}

async function waitForHealth(timeoutMs = 60_000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(healthUrl);
      if (response.ok) {
        const body = await response.json();
        if (body?.ok) return body;
      }
    } catch {
      /* backend not listening yet */
    }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  throw new Error(`Backend did not become healthy at ${healthUrl}`);
}

async function main() {
  const python = resolvePython();
  console.log(`[start] launching backend with ${python} on port ${port}`);

  const backend = run(
    python,
    ["-m", "uvicorn", "living_agent_v2.api:app", "--host", "127.0.0.1", "--port", String(port)],
    { cwd: backendDir }
  );

  backend.on("exit", (code) => {
    if (code !== 0) {
      console.error(
        `[start] backend exited with code ${code}. Create the venv first:\n` +
          `  cd backend && py -3.12 -m venv .venv && .venv\\Scripts\\activate && pip install -e ".[test]"`
      );
      shutdown(code ?? 1);
    }
  });

  const health = await waitForHealth();
  console.log(`[start] backend healthy: ${health.service} v${health.version ?? "?"}`);

  const frontend = run("npm", ["run", "dev"], {
    cwd: root,
    env: { ...process.env, VITE_BACKEND_TARGET: `http://127.0.0.1:${port}` },
  });
  frontend.on("exit", (code) => shutdown(code ?? 0));
}

for (const signal of ["SIGINT", "SIGTERM"]) {
  process.on(signal, () => shutdown(0));
}

main().catch((error) => {
  console.error(`[start] ${error.message}`);
  shutdown(1);
});
