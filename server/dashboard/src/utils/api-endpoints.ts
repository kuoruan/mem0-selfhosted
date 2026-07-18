export const AUTH_ENDPOINTS = {
  SETUP_STATUS: "/auth/setup-status",
  REGISTER: "/auth/register",
  LOGIN: "/auth/login",
  REFRESH: "/auth/refresh",
  ME: "/auth/me",
  CHANGE_PASSWORD: "/auth/change-password",
  ONBOARDING_COMPLETE: "/auth/onboarding-complete",
  OIDC_PROVIDERS: "/auth/oidc/providers",
  OIDC_LOGIN: (provider: string) => `/auth/oidc/${provider}/login`,
  OIDC_EXCHANGE: "/auth/oidc/exchange",
} as const;

export const MEMORY_ENDPOINTS = {
  BASE: "/memories",
  BY_ID: (memoryId: string) => `/memories/${memoryId}`,
  HISTORY: (memoryId: string) => `/memories/${memoryId}/history`,
  CONFIGURE: "/configure",
  CONFIGURE_PROVIDERS: "/configure/providers",
  RESET: "/reset",
  GENERATE_INSTRUCTIONS: "/generate-instructions",
} as const;

export const API_KEY_ENDPOINTS = {
  BASE: "/api-keys",
  BY_ID: (keyId: string) => `/api-keys/${keyId}`,
} as const;

export const REQUEST_ENDPOINTS = {
  BASE: "/requests",
} as const;

export const ENTITY_ENDPOINTS = {
  BASE: "/entities",
  BY_ID: (type: string, id: string) =>
    `/entities/${type}/${encodeURIComponent(id)}`,
  PERMISSIONS: (type: string, id: string) =>
    `/entities/${type}/${encodeURIComponent(id)}/permissions`,
  PERMISSION_BY_USER: (type: string, id: string, granteeId: string) =>
    `/entities/${type}/${encodeURIComponent(id)}/permissions/${granteeId}`,
  TRANSFER_OWNER: (type: string, id: string) =>
    `/entities/${type}/${encodeURIComponent(id)}/transfer-owner`,
  COUNT: (type: string, id: string) =>
    `/entities/${type}/${encodeURIComponent(id)}/count`,
  UPDATE: (type: string, id: string) =>
    `/entities/${type}/${encodeURIComponent(id)}`,
} as const;

export const USER_ENDPOINTS = {
  BASE: "/users",
} as const;
