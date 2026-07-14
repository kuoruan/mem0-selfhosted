export interface Memory {
  id: string;
  memory: string;
  user_id?: string;
  agent_id?: string;
  created_at?: string;
  updated_at?: string;
}

export interface ApiKey {
  id: string;
  label: string;
  key_prefix: string;
  created_at: string;
  last_used_at: string | null;
}

export interface ApiKeyCreateResponse {
  id: string;
  label: string;
  key: string;
  key_prefix: string;
  created_at: string;
}

export interface ApiRequestLog {
  id: string;
  created_at: string;
  method: string;
  path: string;
  status_code: number;
  latency_ms: number;
  auth_type: string;
}

export type EntityType = "user" | "agent" | "app" | "run";

export interface ParentEntityInfo {
  id: string;
  type: EntityType;
  name: string | null;
}

export interface Entity {
  id: string;
  type: EntityType;
  name: string | null;
  created_at: string | null;
  updated_at: string | null;
  owner: UserInfo | null;
  parent: ParentEntityInfo | null;
  is_owner: boolean;
}

export type EntityPermissionLevel = "read" | "write" | "admin";

export interface UserInfo {
  id: string;
  name: string;
  email: string;
}

export interface EntityPermission {
  id: string;
  user: UserInfo;
  permission: EntityPermissionLevel;
  granted_by: UserInfo | null;
  created_at: string;
}

export interface User {
  id: string;
  name: string;
  email: string;
  role: string;
  auth_provider: string;
  created_at: string;
}
