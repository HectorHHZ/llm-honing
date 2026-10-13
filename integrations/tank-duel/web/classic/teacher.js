// The planner teacher: answers planned-steering questions from the planner's
// own geometry, with no model. It takes the highest-scoring motion plan (always
// admissible) and fires only when that plan's fresh shot is verified safe.
// It answers through the same /v1/decide request shape a decision server does,
// so a match can use it as a fixed opponent, or log its answers as labelled
// demonstrations of the questions the model is asked.
export const TEACHER_MODEL = "planner-teacher";
const oneHot = (keys, choice) =>
    Object.fromEntries(keys.map((k) => [k, k === choice ? 1 : 0]));
export function plannerTeacher(observed) {
    if (!observed.actions)
        throw Error("The planner teacher answers planned steering only");
    const best = observed.actions.reduce((a, b) => (b.score > a.score ? b : a));
    return async function ask({ id, parts, questions }) {
        const answers = {};
        for (const [head, q] of Object.entries(questions)) {
            const keys = Object.keys(q.criteria);
            let choice;
            if (head === "motion") choice = best.id;
            else if (head === "weapon")
                choice = best.allowFire ? "fire" : "hold";
            else throw Error(`The planner teacher cannot answer ${head}`);
            if (!keys.includes(choice))
                throw Error(
                    `The planner teacher's ${head} answer is not offered`,
                );
            answers[head] = {
                type: "choice",
                choice,
                probabilities: oneHot(keys, choice),
            };
        }
        return {
            id,
            answers,
            stateText: parts.essential.join("\n"),
            trimmed: [],
            tokenAudit: {},
        };
    };
}
