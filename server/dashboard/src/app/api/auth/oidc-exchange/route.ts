import { cookies } from "next/headers";
import { NextRequest, NextResponse } from "next/server";
import { AUTH_ENDPOINTS } from "@/utils/api-endpoints";
import {
  REFRESH_TOKEN_COOKIE_NAME,
  getRefreshTokenCookieOptions,
} from "@/lib/auth-cookie";
import { getServerApiUrl } from "@/lib/server-api-url";

export async function POST(request: NextRequest) {
  let body: { exchange_code?: unknown };
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }
  const cookieStore = await cookies();

  if (typeof body.exchange_code !== "string" || !body.exchange_code) {
    return NextResponse.json(
      { error: "Missing exchange_code" },
      { status: 400 },
    );
  }
  const exchangeCode = body.exchange_code;

  const res = await fetch(
    `${getServerApiUrl()}${AUTH_ENDPOINTS.OIDC_EXCHANGE}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ exchange_code: exchangeCode }),
    },
  );

  if (!res.ok) {
    // Surface the upstream status so the client can distinguish an
    // invalid/expired code (4xx) from a limiter/provider issue (5xx).
    return NextResponse.json(
      { error: "Exchange failed" },
      { status: res.status },
    );
  }

  const data = await res.json();

  cookieStore.set(
    REFRESH_TOKEN_COOKIE_NAME,
    data.refresh_token,
    getRefreshTokenCookieOptions(),
  );

  return NextResponse.json({ ok: true });
}
