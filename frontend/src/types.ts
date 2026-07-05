// Mirrors the Pydantic response models in src/api.py.

export interface LoginResponse {
  access_token: string;
  token_type: string;
  org_id: number;
  org_name: string;
  role: string;
  display_name: string;
}

export interface CellLatest {
  cell_id: string;
  cycle_number: number;
  soh_pct: number;
  status: string;
  rul_pred: number | null;
  rul_reliable: boolean;
  fade_rate_30cy: number | null;
  capacity_ah: number | null;
  resistance_normalized: number | null;
  coulombic_efficiency: number | null;
}

export interface HistoryPoint {
  cycle_number: number;
  soh_pct: number;
  rul_pred: number | null;
  capacity_ah: number | null;
}

export interface CellHistory {
  cell_id: string;
  total_cycles: number;
  returned: number;
  rul_reliable: boolean;
  history: HistoryPoint[];
}

export interface RULResult {
  cell_id: string;
  cycle_number: number;
  soh_pct: number;
  rul_pred: number | null;
  rul_q10: number | null;
  rul_q90: number | null;
  rul_reliable: boolean;
  eol_threshold_pct: number;
}

export interface FleetSummary {
  total_cells: number;
  n_healthy: number;
  n_degrading: number;
  n_eol: number;
  fleet_soh_mean: number;
  fleet_soh_min: number;
  fleet_soh_max: number;
  cells_near_eol_3m: number;
  cells_near_eol_6m: number;
  capex_estimate_12m_usd: number;
}

export interface AlertItem {
  severity: "critical" | "high" | "medium";
  cell_id: string;
  title: string;
  body: string;
}

export interface FleetAlerts {
  total_alerts: number;
  critical: number;
  high: number;
  medium: number;
  alerts: AlertItem[];
}
