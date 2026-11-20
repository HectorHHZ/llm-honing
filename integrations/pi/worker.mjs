// Run one pi episode from a directory Honing's PiEpisodeRunner prepared.
//
// Runs pi through its SDK and imports nothing from Honing. It applies the agent plan in episode.json as given;
// what each harness file means was decided on Honing's side (honing/backends/pi/harness.py). Nothing is
// discovered: no extensions, context files, prompt templates, or skills beyond the plan's, and settings come
// only from the harness.
//
// Exit code: 0 when the session finished, 2 when it stopped for another reason (part of the harness did not reach
// pi, the runner stopped it, pi aborted it, or the model's provider kept failing), 1 when it could not run. The key
// named in episode.json is handed to pi in memory and removed from the environment the agent's shells inherit.

import { createHash } from "node:crypto";
import { createWriteStream, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import {
	createAgentSession,
	createExtensionRuntime,
	createSyntheticSourceInfo,
	loadSkills,
	ModelRuntime,
	SessionManager,
	SettingsManager,
} from "@earendil-works/pi-coding-agent";

const writeJson = (path, value) => writeFileSync(path, `${JSON.stringify(value, null, 2)}\n`);

/** Builds the skills a plan names, each under the plan's name. */
function buildSkills(plan, harness, cwd, agentDir) {
	const skills = [];
	const diagnostics = [];
	for (const skill of plan.skills) {
		const filePath = join(harness, skill.path);
		if (skill.description !== null) {
			// The file has no frontmatter; Honing described it.
			skills.push({
				name: skill.name,
				description: skill.description,
				filePath,
				baseDir: dirname(filePath),
				sourceInfo: createSyntheticSourceInfo(filePath, { source: "honing" }),
				disableModelInvocation: false,
			});
			continue;
		}
		const loaded = loadSkills({ cwd, agentDir, skillPaths: [filePath], includeDefaults: false });
		diagnostics.push(...loaded.diagnostics);
		skills.push(...loaded.skills.map((found) => ({ ...found, name: skill.name })));
	}
	return { skills, diagnostics };
}

/** A resource loader that holds exactly the plan's prompt and skills. */
function planLoader(plan, harness, skills) {
	const read = (path) => readFileSync(join(harness, path), "utf8");
	return {
		getExtensions: () => ({ extensions: [], errors: [], runtime: createExtensionRuntime() }),
		getSkills: () => skills,
		getPrompts: () => ({ prompts: [], diagnostics: [] }),
		getThemes: () => ({ themes: [], diagnostics: [] }),
		getAgentsFiles: () => ({ agentsFiles: [] }),
		getSystemPrompt: () => (plan.system_prompt === null ? undefined : read(plan.system_prompt)),
		getSystemPromptSource: () =>
			plan.system_prompt === null ? undefined : { path: join(harness, plan.system_prompt) },
		getAppendSystemPrompt: () => plan.append_system_prompt.map(read),
		getAppendSystemPromptSources: () => plan.append_system_prompt.map((path) => ({ path: join(harness, path) })),
		extendResources: () => {},
		reload: async () => {},
	};
}

/** Each harness path that reached pi, with its route and the SHA-256 of the content read. */
function delivered(plan, harness, session, skills) {
	const prompt = session.systemPrompt;
	const tools = [...session.getActiveToolNames()].sort();
	const thinking = plan.settings.defaultThinkingLevel;
	const listed = new Set(
		skills.skills.filter((skill) => prompt.includes(skill.filePath)).map((skill) => skill.name),
	);
	const reached = (route, content) => {
		const [kind, name] = route.split(":");
		if (kind === "system_prompt" || kind === "append_system_prompt") return prompt.includes(content.trim());
		if (kind === "skill" || kind === "skill_resource") return listed.has(name);
		if (kind === "settings") {
			const sameTools = JSON.stringify(tools) === JSON.stringify([...plan.tools].sort());
			// pi clamps a level the model cannot use, so a level counts only if the session kept it.
			return sameTools && (thinking === undefined || session.thinkingLevel === thinking);
		}
		return kind === "workspace_file";
	};
	const files = {};
	const missing = [];
	for (const [path, route] of Object.entries(plan.routes)) {
		const content = readFileSync(join(harness, path), "utf8");
		if (reached(route, content)) {
			files[path] = { route, sha256: createHash("sha256").update(content).digest("hex") };
		} else {
			missing.push(`${path} (${route})`);
		}
	}
	const report = { files, skills: [...listed].sort(), tools, thinking_level: session.thinkingLevel };
	return { ...report, missing, diagnostics: skills.diagnostics.map((d) => `${d.path ?? ""}: ${d.message}`) };
}

/** Why the session stopped, from its last assistant message and its settled event. */
function outcome(session, settled) {
	const last = session.messages.filter((message) => message.role === "assistant").at(-1);
	if (settled?.aborted || last?.stopReason === "aborted") return { status: "aborted", error: last?.errorMessage };
	if (last?.stopReason === "error") return { status: "error", error: last.errorMessage };
	return { status: "finished", error: undefined };
}

/** The tokens and cost of every assistant message. */
function usage(session) {
	const total = { input_tokens: 0, output_tokens: 0, cost_usd: 0 };
	for (const message of session.messages) {
		if (message.role !== "assistant" || !message.usage) continue;
		total.input_tokens += message.usage.input ?? 0;
		total.output_tokens += message.usage.output ?? 0;
		total.cost_usd += message.usage.cost?.total ?? 0;
	}
	return total;
}

async function run(root, stop) {
	const episode = JSON.parse(readFileSync(join(root, "episode.json"), "utf8"));
	const plan = episode.agent;
	const harness = join(root, "harness");
	const agentDir = join(root, "agent");
	const cwd = join(root, "work");

	for (const path of plan.files) {
		const target = join(cwd, ".harness", path);
		mkdirSync(dirname(target), { recursive: true });
		writeFileSync(target, readFileSync(join(harness, path)));
	}
	const skills = buildSkills(plan, harness, cwd, agentDir);
	const modelRuntime = await ModelRuntime.create({
		authPath: join(agentDir, "auth.json"),
		modelsPath: join(agentDir, "models.json"),
	});
	const { provider, id, api_key_env: keyEnv } = episode.model;
	if (keyEnv) {
		// pi holds the key in memory; the shells the agent's tools start inherit process.env, so it must not be there.
		await modelRuntime.setRuntimeApiKey(provider, process.env[keyEnv] ?? "");
		delete process.env[keyEnv];
	}
	const model = modelRuntime.getModel(provider, id);
	if (!model) throw new Error(`models.json defines no model ${provider}/${id}`);
	const { session } = await createAgentSession({
		cwd,
		agentDir,
		model,
		modelRuntime,
		resourceLoader: planLoader(plan, harness, skills),
		tools: plan.tools,
		sessionManager: SessionManager.inMemory(cwd),
		settingsManager: SettingsManager.inMemory(plan.settings),
	});
	stop.session = session;
	// Before the first model call, so an episode that is stopped still says what reached pi.
	const delivery = delivered(plan, harness, session, skills);
	writeJson(join(root, "delivery.json"), delivery);
	if (delivery.missing.length > 0) {
		// An episode without its harness would be scored as if the harness had been tried.
		const said = delivery.diagnostics.length > 0 ? `; pi said: ${delivery.diagnostics.join("; ")}` : "";
		writeJson(join(root, "result.json"), {
			final_output: "",
			status: "undelivered",
			error: `the harness did not reach pi: ${delivery.missing.join(", ")}${said}`,
		});
		session.dispose();
		return 2;
	}

	const transcript = createWriteStream(join(root, "transcript.jsonl"));
	let settled;
	session.subscribe((event) => {
		let record = event;
		if (event.type === "message_update") {
			// As pi's JSON mode writes it: deltas only, without the cumulative message.
			const { partial, ...update } = event.assistantMessageEvent;
			record = { type: event.type, assistantMessageEvent: update };
		}
		if (event.type === "agent_settled") settled = event;
		transcript.write(`${JSON.stringify(record)}\n`);
	});
	let stopped = { status: "failed" };
	try {
		if (!stop.requested) await session.prompt(episode.task.prompt);
	} finally {
		stopped = stop.requested ? { status: "aborted", error: "stopped by the runner" } : outcome(session, settled);
		writeJson(join(root, "result.json"), {
			final_output: session.getLastAssistantText() ?? "",
			...stopped,
			messages: session.messages.length,
			...usage(session),
		});
		session.dispose();
		await new Promise((done) => transcript.end(done));
	}
	return stopped.status === "finished" ? 0 : 2;
}

const root = resolve(process.argv[2]);
// The runner asks the worker to stop with SIGTERM. pi starts each shell command in its own process group, so
// only aborting the session reaches them: it kills every running command's process tree.
const stop = { requested: false, session: undefined };
process.once("SIGTERM", () => {
	stop.requested = true;
	if (stop.session === undefined) process.exit(143);
	stop.session.abort().catch((error) => console.error(error));
});
let code;
try {
	code = await run(root, stop);
} catch (error) {
	try {
		readFileSync(join(root, "result.json"));
	} catch {
		writeJson(join(root, "result.json"), { final_output: "", status: "failed", error: String(error?.stack ?? error) });
	}
	console.error(error);
	code = 1;
}
// pi's HTTP clients can keep the event loop alive after the session is disposed.
process.exit(code);
