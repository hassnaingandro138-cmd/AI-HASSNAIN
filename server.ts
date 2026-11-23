import "dotenv/config";
import express from "express";
import type { Request, Response } from "express";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { GoogleGenAI, ThinkingLevel } from "@google/genai";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = Number(process.env.PORT) || 3000;
const FAST_MODEL = "gemini-3.1-flash-lite";
const SMART_MODEL = "gemini-4-hydrogen";

const PERSONALITY_PROMPTS: Record<string, string> = {
  balanced:
    "Be warm, intelligent, well-structured, and naturally conversational.",
  coding:
    "Act as a senior software engineer. Write clean code with concise explanations and best practices.",
  tutor:
    "Act as a patient study tutor. Break down concepts step-by-step with clear analogies.",
  creative:
    "Be expressive, imaginative, and articulate in stories, brainstorms, and prose.",
  concise:
    "Be direct, crisp, and focused. Deliver accurate answers with minimal fluff.",
};

interface IncomingMessage {
  role: "user" | "assistant";
  content: string;
}

interface ChatRequestBody {
  messages?: unknown;
  personalityPreset?: unknown;
  customInstructions?: unknown;
  responseSpeed?: unknown;
}

let cachedGeminiClient: { key: string; client: GoogleGenAI } | null = null;

function getGeminiClient(): GoogleGenAI | null {
  const apiKey = process.env.GEMINI_API_KEY?.trim();
  if (!apiKey || apiKey === "MY_GEMINI_API_KEY") {
    return null;
  }
  if (cachedGeminiClient && cachedGeminiClient.key === apiKey) {
    return cachedGeminiClient.client;
  }
  const client = new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        "User-Agent": "aistudio-build",
      },
    },
  });
  cachedGeminiClient = { key: apiKey, client };
  return client;
}

function buildSystemInstruction(
  presetKey: string,
  customInstructions: string
): string {
  const presetPrompt =
    PERSONALITY_PROMPTS[presetKey] ?? PERSONALITY_PROMPTS.balanced;

  const baseInstruction = [
    "You are HASSNAIN AI, an intelligent, helpful, and natural AI assistant.",
    "Multilingual capability: Automatically detect the user's language in their latest message and respond fluently in the exact same language and script:",
    "- If the user writes in English, reply in English.",
    "- If the user writes in Urdu script (اردو), reply naturally in proper Urdu script (اردو).",
    "- If the user writes in Roman Urdu (Urdu written with English/Latin letters, e.g., 'kya haal hai', 'mujhe code samjhao'), reply naturally in clear Roman Urdu.",
    "- If the user mixes languages (e.g., English technical terms in Roman Urdu), mirror that natural bilingual style.",
    "Formatting: Use clean Markdown formatting (headings, lists, tables, and fenced code blocks with language identifiers) whenever it improves readability.",
    `Style & Personality: ${presetPrompt}`,
  ];

  if (customInstructions.trim().length > 0) {
    baseInstruction.push(
      `Additional user preference for assistant behavior: ${customInstructions.trim()}`
    );
  }

  return baseInstruction.join("\n\n");
}

function parseFriendlyError(err: unknown): { status: number; message: string } {
  const rawMessage =
    err instanceof Error ? err.message : typeof err === "string" ? err : "";
  const lower = rawMessage.toLowerCase();

  if (
    lower.includes("api_key_invalid") ||
    lower.includes("api key not valid") ||
    lower.includes("permission_denied") ||
    lower.includes("403") ||
    lower.includes("401")
  ) {
    return {
      status: 403,
      message:
        "Invalid or unauthorized Gemini API key. Please verify your GEMINI_API_KEY in Settings > Secrets.",
    };
  }

  if (
    lower.includes("resource_exhausted") ||
    lower.includes("429") ||
    lower.includes("quota") ||
    lower.includes("rate limit")
  ) {
    return {
      status: 429,
      message:
        "Rate limit reached on the Gemini API free tier. Please wait a moment and try again.",
    };
  }

  if (lower.includes("not_found") || lower.includes("404")) {
    return {
      status: 404,
      message:
        "Requested Gemini model was not found or is temporarily unavailable.",
    };
  }

  if (
    lower.includes("fetch failed") ||
    lower.includes("network") ||
    lower.includes("enotfound") ||
    lower.includes("etimedout")
  ) {
    return {
      status: 503,
      message:
        "Network error reaching Gemini AI services. Please check your connection and try again.",
    };
  }

  return {
    status: 500,
    message:
      rawMessage && rawMessage.length < 240
        ? rawMessage
        : "Unable to generate a response right now. Please try again.",
  };
}

function validateChatPayload(body: ChatRequestBody): {
  valid: boolean;
  error?: string;
  messages?: IncomingMessage[];
  preset?: string;
  customInstructions?: string;
  responseSpeed?: "fast" | "smart";
} {
  if (!body || !Array.isArray(body.messages) || body.messages.length === 0) {
    return {
      valid: false,
      error: "Please enter a message before sending.",
    };
  }

  if (body.messages.length > 100) {
    return {
      valid: false,
      error: "Conversation context exceeded maximum length (100 messages).",
    };
  }

  const sanitizedMessages: IncomingMessage[] = [];
  for (const item of body.messages) {
    if (!item || typeof item !== "object") {
      return { valid: false, error: "Invalid message format." };
    }
    const record = item as Record<string, unknown>;
    const role = record.role;
    const content = record.content;

    if (role !== "user" && role !== "assistant") {
      return { valid: false, error: "Invalid message role." };
    }
    if (typeof content !== "string") {
      return { valid: false, error: "Message content must be text." };
    }
    const trimmed = content.trim();
    if (trimmed.length === 0) {
      return { valid: false, error: "Messages cannot be empty." };
    }
    if (trimmed.length > 32000) {
      return {
        valid: false,
        error: "Message is too long. Please shorten it under 32,000 characters.",
      };
    }
    sanitizedMessages.push({ role, content: trimmed });
  }

  if (sanitizedMessages[sanitizedMessages.length - 1]?.role !== "user") {
    return {
      valid: false,
      error: "The latest message in the conversation must be from the user.",
    };
  }

  const preset =
    typeof body.personalityPreset === "string" &&
    Object.prototype.hasOwnProperty.call(
      PERSONALITY_PROMPTS,
      body.personalityPreset
    )
      ? body.personalityPreset
      : "balanced";

  const customInstructions =
    typeof body.customInstructions === "string"
      ? body.customInstructions.slice(0, 1000)
      : "";

  const responseSpeed =
    body.responseSpeed === "smart" ? "smart" : "fast";

  return {
    valid: true,
    messages: sanitizedMessages,
    preset,
    customInstructions,
    responseSpeed,
  };
}

async function startServer() {
  const app = express();
  app.use(express.json({ limit: "1mb" }));

  app.get("/api/health", (_req: Request, res: Response) => {
    const client = getGeminiClient();
    res.json({
      ok: true,
      configured: Boolean(client),
      model: FAST_MODEL,
      status: client ? "connected" : "missing_key",
    });
  });

  app.post("/api/chat", async (req: Request, res: Response) => {
    const validation = validateChatPayload(req.body as ChatRequestBody);
    if (!validation.valid || !validation.messages) {
      res.status(400).json({ error: validation.error ?? "Invalid request." });
      return;
    }

    const ai = getGeminiClient();
    if (!ai) {
      res.status(503).json({
        error:
          "Gemini API key is not configured. Please add GEMINI_API_KEY in Settings > Secrets.",
      });
      return;
    }

    const abortController = new AbortController();
    let clientDisconnected = false;

    res.on("close", () => {
      if (!res.writableFinished) {
        clientDisconnected = true;
        abortController.abort();
      }
    });

    res.setHeader("Content-Type", "text/event-stream; charset=utf-8");
    res.setHeader("Cache-Control", "no-cache, no-transform");
    res.setHeader("Connection", "keep-alive");
    res.setHeader("X-Accel-Buffering", "no");
    res.flushHeaders?.();

    const sendEvent = (payload: Record<string, unknown>) => {
      if (!clientDisconnected && !res.writableEnded) {
        res.write(`data: ${JSON.stringify(payload)}\n\n`);
      }
    };

    try {
      const recentMessages = validation.messages.slice(-24);
      const contents = recentMessages.map((msg) => ({
        role: msg.role === "assistant" ? "model" : "user",
        parts: [{ text: msg.content }],
      }));

      const systemInstruction = buildSystemInstruction(
        validation.preset ?? "balanced",
        validation.customInstructions ?? ""
      );

      const useSmartMode =
        (validation as { responseSpeed?: string }).responseSpeed === "smart";
      const selectedModel = useSmartMode ? SMART_MODEL : FAST_MODEL;
      const selectedThinking = useSmartMode
        ? ThinkingLevel.LOW
        : ThinkingLevel.MINIMAL;

      const maxAttempts = 2;
      let streamedAnyChunk = false;

      for (let attempt = 1; attempt <= maxAttempts; attempt++) {
        try {
          const stream = await ai.models.generateContentStream({
            model: selectedModel,
            contents,
            config: {
              systemInstruction,
              thinkingConfig: {
                thinkingLevel: selectedThinking,
              },
              abortSignal: abortController.signal,
            },
          });

          for await (const chunk of stream) {
            if (clientDisconnected) {
              break;
            }
            const chunkText = chunk.text;
            if (chunkText) {
              streamedAnyChunk = true;
              sendEvent({ type: "delta", text: chunkText });
            }
          }
          break;
        } catch (attemptErr) {
          if (clientDisconnected) {
            return;
          }
          const parsed = parseFriendlyError(attemptErr);
          const isTransient = parsed.status === 500 || parsed.status === 503;
          if (attempt < maxAttempts && !streamedAnyChunk && isTransient) {
            await new Promise((resolve) => setTimeout(resolve, 350));
            continue;
          }
          throw attemptErr;
        }
      }

      if (!clientDisconnected) {
        sendEvent({ type: "done" });
        res.end();
      }
    } catch (err) {
      if (clientDisconnected) {
        return;
      }
      const friendly = parseFriendlyError(err);
      sendEvent({
        type: "error",
        status: friendly.status,
        error: friendly.message,
      });
      res.end();
    }
  });

  if (process.env.NODE_ENV !== "production") {
    const { createServer: createViteServer } = await import("vite");
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distDir = path.resolve(__dirname, "dist");
    app.use(express.static(distDir));
    app.get("/{*splat}", (_req: Request, res: Response) => {
      const indexFile = path.join(distDir, "index.html");
      if (fs.existsSync(indexFile)) {
        res.sendFile(indexFile);
      } else {
        res.status(404).send("Application build not found.");
      }
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`HASSNAIN AI server listening on http://localhost:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error("Failed to start HASSNAIN AI server:", err);
  process.exit(1);
});
