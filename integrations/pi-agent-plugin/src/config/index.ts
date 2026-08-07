import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import type { Mem0Config, DreamConfig } from "../types.ts";

const AGENT_ROOT = path.join(os.homedir(), ".pi", "agent");
export const CONFIG_DIR = AGENT_ROOT;
const CONFIG_PATH = path.join(AGENT_ROOT, "mem0-config.json");

const DEFAULT_DREAM: DreamConfig = {
  enabled: true,
  auto: true,
  minHours: 24,
  minSessions: 5,
  minMemories: 20,
};

const DEFAULT_CONFIG: Mem0Config = {
  apiKey: "",
  userId: "",
  apiUrl: "",
  autoCapture: true,
  defaultScope: "project",
  contextInjection: true,
  searchThreshold: 0.3,
  dream: DEFAULT_DREAM,
};

function isValidHttpUrl(url: string): boolean {
  return /^https?:\/\//i.test(url);
}

/**
 * Resolve the REST API base URL. Precedence:
 * 1. ``MEM0_API_URL`` env var
 * 2. ``apiUrl`` from config file
 * 3. Empty (platform default ``https://api.mem0.ai`` handled by MemoryClient)
 *
 * Throws when a URL was explicitly configured but is invalid — prevents
 * silently falling back to the platform default.
 */
function resolveApiUrl(fileApiUrl: string): string {
  const url = (process.env.MEM0_API_URL ?? "").trim();
  if (url) {
    if (!isValidHttpUrl(url)) {
      throw new Error(`MEM0_API_URL does not start with http:// or https:// (got ${JSON.stringify(url)}).`);
    }
    return url.replace(/\/+$/, "");
  }

  const fileUrl = (fileApiUrl ?? "").trim();
  if (fileUrl) {
    if (!isValidHttpUrl(fileUrl)) {
      throw new Error(`apiUrl in config does not start with http:// or https:// (got ${JSON.stringify(fileUrl)}).`);
    }
    return fileUrl.replace(/\/+$/, "");
  }
  return "";
}

export function loadConfig(): Mem0Config {
  let fileConfig: Partial<Mem0Config> = {};

  if (fs.existsSync(CONFIG_PATH)) {
    try {
      const raw = fs.readFileSync(CONFIG_PATH, "utf-8");
      fileConfig = JSON.parse(raw);
    } catch {
      // Corrupted config — use defaults
    }
  }

  const dream: DreamConfig = {
    ...DEFAULT_DREAM,
    ...(fileConfig.dream ?? {}),
  };

  const config: Mem0Config = {
    ...DEFAULT_CONFIG,
    ...fileConfig,
    dream,
  };

  if (process.env.MEM0_API_KEY) {
    config.apiKey = process.env.MEM0_API_KEY;
  }
  if (process.env.MEM0_USER_ID) {
    config.userId = process.env.MEM0_USER_ID;
  }

  config.apiUrl = resolveApiUrl(config.apiUrl);

  return config;
}
