"use client";

import { useEffect, useRef } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { setAccessToken } from "@/utils/api";
import { safeRedirectPath } from "@/utils/helpers";

export default function OidcCallbackInner() {
  const router = useRouter();
  const searchParams = useSearchParams();
  // React Strict Mode (dev) mounts effects twice. The exchange code is
  // single-use, so a second concurrent fetch would hit 401 and its catch
  // branch (redirect to /login) would clobber the successful navigation.
  // Guard so the flow runs exactly once.
  const ranRef = useRef(false);

  useEffect(() => {
    if (ranRef.current) return;
    ranRef.current = true;

    const run = async () => {
      // The OIDC backend redirects here with tokens in the URL fragment:
      // /auth/callback#access_token=...&code=...&token_type=bearer
      // or with an error: /auth/callback#error=...
      // Fragments are not available server-side, so parse them client-side.
      //
      // The refresh_token is never placed in the fragment — the backend stores
      // it behind a short-lived (60s) single-use exchange code. The frontend
      // sends the code to /api/auth/oidc-exchange, which stores the refresh_token
      // as an httpOnly cookie.

      const hash = window.location.hash.substring(1);
      const params = new URLSearchParams(hash);

      const accessToken = params.get("access_token");
      const exchangeCode = params.get("code");
      const errorParam = params.get("error");
      const rawNext = searchParams.get("next");
      const next = safeRedirectPath(rawNext, "/dashboard/requests");

      if (errorParam) {
        // Redirect to login with error message
        window.location.href = `/login?error=${encodeURIComponent(errorParam)}`;
        return;
      }

      if (!accessToken || !exchangeCode) {
        window.location.href = `/login?error=${encodeURIComponent("OIDC authentication failed: missing tokens")}`;
        return;
      }

      try {
        // Store the access token in memory
        setAccessToken(accessToken);

        // Exchange the short-lived code for the refresh token (stored as httpOnly cookie)
        const res = await fetch("/api/auth/oidc-exchange", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ exchange_code: exchangeCode }),
        });

        if (!res.ok) {
          throw new Error("Failed to exchange OIDC code");
        }

        // Navigate to the dashboard (client-side to preserve the in-memory token).
        // Use replace so the token-laden callback URL is not kept in browser history.
        router.replace(next);
      } catch {
        setAccessToken(null);
        window.location.href = `/login?error=${encodeURIComponent("Failed to complete OIDC login")}`;
      }
    };

    run();
  }, [router, searchParams]);

  return (
    <div className="flex min-h-screen items-center justify-center">
      <div className="text-center space-y-4">
        <div className="size-8 border-2 border-primary border-t-transparent rounded-full animate-spin mx-auto" />
        <p className="text-onSurface-default-secondary text-sm">
          Completing sign in…
        </p>
      </div>
    </div>
  );
}
