import type {
  CellSceneSpec,
  LoginResponse,
  CellLatest,
  CellHistory,
  RULResult,
  FleetSummary,
  FleetAlerts,
  ActionTicket,
  DynamicLCA,
  SecondLifeBid,
  CyclerDetection,
} from "./types";

// Same-origin by default. The deployment path is the API serving this build
// from /app (see src/api.py), so every request goes to the page's own host and
// no CORS allow-list or second host is involved. Only `npm run dev` is
// cross-origin (Vite on :5173, API on :8000) — that case sets
// VITE_API_BASE_URL, see .env.example.
const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "";

const TOKEN_KEY = "battery_api_token";

export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}

export function setToken(token: string): void {
  localStorage.setItem(TOKEN_KEY, token);
}

export function clearToken(): void {
  localStorage.removeItem(TOKEN_KEY);
}

class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const token = getToken();
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(options.headers as Record<string, string> | undefined),
  };
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }
  const resp = await fetch(`${BASE_URL}${path}`, { ...options, headers });
  if (!resp.ok) {
    const body = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new ApiError(resp.status, body.detail ?? resp.statusText);
  }
  return resp.json() as Promise<T>;
}

export async function login(username: string, password: string): Promise<LoginResponse> {
  return request<LoginResponse>("/auth/login", {
    method: "POST",
    body: JSON.stringify({ username, password }),
  });
}

export async function listCells(): Promise<{ cells: string[]; count: number }> {
  return request("/cells");
}

export async function getCellLatest(cellId: string): Promise<CellLatest> {
  return request(`/cells/${encodeURIComponent(cellId)}`);
}

export async function getCellHistory(cellId: string, limit = 200): Promise<CellHistory> {
  return request(`/cells/${encodeURIComponent(cellId)}/history?limit=${limit}`);
}

export async function getCellRul(cellId: string): Promise<RULResult> {
  return request(`/cells/${encodeURIComponent(cellId)}/rul`);
}

// The 3D cell scene. A raw `CellSceneSpec` rather than a modelled type: the
// contract is the published JSON Schema (docs/cell_scene.schema.json) and the
// versioned `schemaVersion` field inside the document, so a server that adds a
// field cannot silently drift from a renderer that was built before it — see
// frontend/src/scene/index.ts's `checkSchemaVersion`.
export async function getCellScene(cellId: string, horizonCycles = 300): Promise<CellSceneSpec> {
  return request(`/cells/${encodeURIComponent(cellId)}/scene?horizon_cycles=${horizonCycles}`);
}

export async function getFleetSummary(): Promise<FleetSummary> {
  return request("/fleet/summary");
}

export async function getFleetAlerts(): Promise<FleetAlerts> {
  return request("/fleet/alerts");
}

// ── Action Center ───────────────────────────────────────────────────────────

export async function listActions(): Promise<ActionTicket[]> {
  return request("/actions");
}

export async function triageAction(actionId: string, status: string, assignedTo?: string): Promise<ActionTicket> {
  return request(`/actions/${encodeURIComponent(actionId)}/triage`, {
    method: "POST",
    body: JSON.stringify({ status, assigned_to: assignedTo }),
  });
}

export async function dispatchAction(actionId: string, targetSystem: string): Promise<{ status: string; dispatch_reference: string; summary: string }> {
  return request(`/actions/${encodeURIComponent(actionId)}/dispatch`, {
    method: "POST",
    body: JSON.stringify({ target_system: targetSystem }),
  });
}

// ── Dynamic LCA & Passport ──────────────────────────────────────────────────

export async function getDynamicLCA(cellId: string, region = "EU_AVG"): Promise<DynamicLCA> {
  return request(`/cells/${encodeURIComponent(cellId)}/dynamic-lca?region=${encodeURIComponent(region)}`);
}

export async function getVerifiablePassport(cellId: string): Promise<any> {
  return request(`/cells/${encodeURIComponent(cellId)}/verifiable-passport`);
}

export async function getSecondLifeBids(cellId: string): Promise<SecondLifeBid[]> {
  return request(`/cells/${encodeURIComponent(cellId)}/second-life-bids`);
}

// ── Ingestion & Telemetry ───────────────────────────────────────────────────

export async function detectCyclerFormat(columns: string[]): Promise<CyclerDetection> {
  return request("/ingest/cycler-detect", {
    method: "POST",
    body: JSON.stringify(columns),
  });
}

export async function processStreamingSample(sample: {
  cell_id: string;
  voltage_v: number;
  current_a: number;
  temperature_c: number;
}): Promise<any> {
  return request("/telemetry/process", {
    method: "POST",
    body: JSON.stringify(sample),
  });
}

export { ApiError };
