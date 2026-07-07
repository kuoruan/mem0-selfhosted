/**
 * Shared refresh-token cookie configuration.
 *
 * Used by both the login refresh flow (`/api/auth/refresh`) and the OIDC
 * exchange flow (`/api/auth/oidc-exchange`) so the httpOnly cookie that holds
 * the refresh_token is configured identically in both places.
 */

export const REFRESH_TOKEN_COOKIE_NAME = "mem0_refresh_token";

function shouldUseSecureCookie(): boolean {
  const dashboardUrl = process.env.DASHBOARD_URL;
  if (!dashboardUrl) {
    return process.env.NODE_ENV === "production";
  }
  try {
    return new URL(dashboardUrl).protocol === "https:";
  } catch {
    return process.env.NODE_ENV === "production";
  }
}

export function getRefreshTokenCookieOptions() {
  return {
    httpOnly: true,
    secure: shouldUseSecureCookie(),
    sameSite: "lax" as const,
    path: "/",
    maxAge: 30 * 24 * 60 * 60, // 30 days
  };
}
