// Prompt text the controller shows the model, kept in a file so it can be
// replaced without editing code. With no file given, the defaults below are
// exactly the strings the controller used before they were moved here.
import { readFileSync } from "node:fs";

export const DEFAULT_QUESTIONS_PATH = new URL(
    "../../config/questions.json",
    import.meta.url,
);

let text = JSON.parse(readFileSync(DEFAULT_QUESTIONS_PATH, "utf8"));

// Replace the text for this process. Every key the controller uses must be
// present, so a partial file fails at start rather than mid-match.
export function loadQuestions(path) {
    const loaded = JSON.parse(readFileSync(path, "utf8"));
    const missing = Object.keys(text).filter((key) => !(key in loaded));
    if (missing.length)
        throw new Error(
            `Questions file is missing keys: ${missing.join(", ")}`,
        );
    text = loaded;
    return text;
}

// Look up one entry and fill its {name} placeholders.
export function t(key, values = {}) {
    const template = text[key];
    if (typeof template !== "string")
        throw new Error(`Unknown or non-string question text: ${key}`);
    return template.replace(/\{(\w+)\}/g, (_, name) => {
        if (!(name in values))
            throw new Error(`Question text ${key} needs a value for {${name}}`);
        return String(values[name]);
    });
}
