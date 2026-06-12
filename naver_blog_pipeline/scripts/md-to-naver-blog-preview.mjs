import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { convert } from "@jjlabsio/md-to-naver-blog";

function parseArgs(argv) {
  const args = new Map();
  for (let index = 2; index < argv.length; index += 2) {
    const key = argv[index];
    const value = argv[index + 1];
    if (!key?.startsWith("--") || value === undefined) {
      throw new Error(`Invalid argument near ${key ?? "(empty)"}`);
    }
    args.set(key.slice(2), value);
  }
  return args;
}

function required(args, key) {
  const value = args.get(key);
  if (!value) {
    throw new Error(`Missing --${key}`);
  }
  return value;
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function normalizeUrl({ type, raw }) {
  if (type === "image") {
    return raw.replaceAll("\\", "/");
  }
  return raw;
}

const args = parseArgs(process.argv);
const inputPath = resolve(required(args, "input"));
const htmlPath = resolve(required(args, "html"));
const jsonPath = resolve(required(args, "json"));

const markdown = readFileSync(inputPath, "utf8");
const result = convert(markdown, { transformUrl: normalizeUrl });
const documentHtml = `<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>${escapeHtml(result.title || "Naver Blog Preview")}</title>
</head>
<body>
${result.html}
</body>
</html>
`;

writeFileSync(htmlPath, documentHtml, "utf8");
writeFileSync(
  jsonPath,
  JSON.stringify(
    {
      title: result.title,
      frontmatter: result.frontmatter,
      errors: result.errors,
      blocks: result.blocks.map((block) => ({ id: block.id, type: block.type })),
    },
    null,
    2,
  ),
  "utf8",
);

if (result.errors.length > 0) {
  console.warn(`converted with ${result.errors.length} parser warning(s)`);
}

