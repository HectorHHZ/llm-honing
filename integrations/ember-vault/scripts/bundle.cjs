"use strict";

const fs = require("node:fs");
const path = require("node:path");

const output = process.argv[2];
if (!output) {
  console.error("Usage: node scripts/bundle.cjs OUTPUT.html");
  process.exit(2);
}

const web = path.join(__dirname, "../web");
const read = name => fs.readFileSync(path.join(web, name), "utf8");
let page = read("index.html").replace('<link rel="stylesheet" href="style.css">', () => `<style>\n${read("style.css")}</style>`);
for (const name of ["engine.js", "render.js", "app.js"]) {
  const code = read(name).replace(/<\/script/gi, "<\\/script");
  page = page.replace(`<script src="${name}"></script>`, () => `<script>\n${code}</script>`);
}

fs.writeFileSync(output, page, { flag: "wx" });
console.log(path.resolve(output));
