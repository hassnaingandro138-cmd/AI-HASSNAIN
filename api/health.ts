import type { NodeApiRequest } from "../lib/geminiApi.ts";
import { handleHealthRequest } from "../lib/geminiApi.ts";
import type { ServerResponse } from "node:http";

export default function handler(
  req: NodeApiRequest,
  res: ServerResponse
): void {
  handleHealthRequest(req, res);
}
