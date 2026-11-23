import type { NodeApiRequest } from "../lib/geminiApi.ts";
import { handleChatRequest } from "../lib/geminiApi.ts";
import type { ServerResponse } from "node:http";

export const config = {
  maxDuration: 60,
};

export default async function handler(
  req: NodeApiRequest,
  res: ServerResponse
): Promise<void> {
  await handleChatRequest(req, res);
}
