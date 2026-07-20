export const toTitleCase = (str: string) => {
  if (!str) return "";
  str = str.toLowerCase();
  return str.replace(/\b\w/g, (char) => char.toUpperCase());
};

export const WILDCARD = "*";

/** Return true if *value* is the wildcard sentinel ``"*"``. */
export const isWildcard = (value: unknown): boolean => value === WILDCARD;

/** Return ``"--"`` for ``undefined``/``null``, otherwise the value itself. */
export const orDash = (v: string | undefined | null): string => v ?? "--";

/** Empty paginated response fallback (``results: [], count: 0, next: null, previous: null``). */
export const EMPTY_PAGE_RESULTS = {
  results: [],
  count: 0,
  next: null,
  previous: null,
};

/**
 * Return a safe relative redirect path, or the fallback if the input is unsafe.
 *
 * Blocks: absolute URLs, protocol-relative URLs (//), backslashes,
 * and whitespace/control characters (which browsers normalize, enabling open-redirect attacks).
 */
export const safeRedirectPath = (
  raw: string | null,
  fallback = "/dashboard/requests",
): string => {
  if (!raw) return fallback;
  if (!raw.startsWith("/")) return fallback;
  if (raw.startsWith("//")) return fallback;
  if (raw.includes("\\")) return fallback;
  if (/\s/.test(raw)) return fallback;
  return raw;
};
