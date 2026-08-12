// Runs only the FastAPI backend (useful for API testing).
import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const backendDir = path.join(root, "backend");
const port = Number(process.env.BACKEND_PORT ?? 8000);
const isWindows = process.platform === "win32";

const venv = isWindows
  ? path.join(backendDir, ".venv", "Scripts", "python.exe")
  : path.join(backendDir, ".venv", "bin", "python");
const python = existsSync(venv)
  ? venv
  : process.env.PYTHON ?? (isWindows ? "py" : "python3");

const child = spawn(
  python,
  [
    "-m",
    "uvicorn",
    "living_agent_v2.api:app",
    "--reload",
    "--host",
    "127.0.0.1",
    "--port",
    String(port),
  ],
  { cwd: backendDir, stdio: "inherit", shell: isWindows }
);

child.on("exit", (code) => process.exit(code ?? 0));
