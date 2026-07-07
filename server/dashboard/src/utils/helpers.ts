export const toTitleCase = (str: string) => {
  if (!str) return "";
  str = str.toLowerCase();
  return str.replace(/\b\w/g, (char) => char.toUpperCase());
};

/**
 * Return a safe relative redirect path, or the fallback if the input is unsafe.
 *
 * Blocks: absolute URLs, protocol-relative URLs (//), backslashes,
 * and whitespace/control characters (which browsers normalize, enabling open-redirect attacks).
 */
export const safeRedirectPath = (raw: string | null, fallback = "/dashboard/requests"): string => {
  if (!raw) return fallback;
  if (!raw.startsWith("/")) return fallback;
  if (raw.startsWith("//")) return fallback;
  if (raw.includes("\\")) return fallback;
  if (/\s/.test(raw)) return fallback;
  return raw;
};
