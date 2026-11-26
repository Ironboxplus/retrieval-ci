// Line protocol over stdin/stdout: {"id":1,"tex":"x^2","display":false} in,
// {"id":1,"ok":true,"chars":"x2"} or {"id":1,"ok":false,"error":"..."} out.
// "chars" is the text of the MathML leaves (mi/mo/mn/mtext) separated by spaces: what a reader sees.
import katex from "katex";
import readline from "node:readline";

const LEAF = /<(mi|mo|mn|mtext)\b[^>]*>(.*?)<\/\1>/gs;
const ENTITIES = { "&lt;": "<", "&gt;": ">", "&amp;": "&", "&quot;": '"', "&#x27;": "'", "&#39;": "'" };

function leafText(mathml) {
  const leaves = [];
  for (const match of mathml.matchAll(LEAF)) {
    leaves.push(match[2].replace(/<[^>]*>/g, "").replace(/&(?:lt|gt|amp|quot|#x27|#39);/g, (e) => ENTITIES[e]));
  }
  return leaves.join(" ");
}

const lines = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
lines.on("line", (line) => {
  let request;
  try {
    request = JSON.parse(line);
  } catch (error) {
    process.stdout.write(JSON.stringify({ id: null, ok: false, error: `bad request: ${error.message}` }) + "\n");
    return;
  }
  try {
    const mathml = katex.renderToString(request.tex, {
      throwOnError: true,
      output: "mathml",
      displayMode: Boolean(request.display),
      strict: "ignore",
    });
    process.stdout.write(JSON.stringify({ id: request.id, ok: true, chars: leafText(mathml) }) + "\n");
  } catch (error) {
    process.stdout.write(JSON.stringify({ id: request.id, ok: false, error: String(error.message ?? error) }) + "\n");
  }
});
