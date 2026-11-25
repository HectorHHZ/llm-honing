// The planner teacher: answers planned-steering questions from the planner's
// own geometry, with no model. It takes the highest-scoring motion plan (always
// admissible) and fires only when that plan's fresh shot is verified safe.
// It answers through the same /v1/decide request shape a decision server does,
// so a match can use it as a fixed opponent, or log its answers as labelled
// demonstrations of the questions the model is asked. Given `motion`, the motion
// a weapon question was asked about, it answers that question for that motion:
// fire only when its fresh shot is verified safe. That labels the weapon
// question on states the model reached with a motion the teacher would not
// have chosen. Without it, the weapon answer is for the teacher's own motion.
export const TEACHER_MODEL = "planner-teacher";
const oneHot = (keys, choice) =>
    Object.fromEntries(keys.map((k) => [k, k === choice ? 1 : 0]));
export function plannerTeacher(observed, { motion } = {}) {
    if (!observed.actions)
        throw Error("The planner teacher answers planned steering only");
    const best = observed.actions.reduce((a, b) => (b.score > a.score ? b : a)),
        asked =
            motion === undefined
                ? best
                : observed.actions.find((a) => a.id === motion);
    if (!asked) throw Error(`The planner teacher has no motion ${motion}`);
    return async function ask({ id, parts, questions }) {
        const answers = {};
        for (const [head, q] of Object.entries(questions)) {
            const keys = Object.keys(q.criteria);
            let choice;
            if (head === "motion") choice = best.id;
            else if (head === "weapon")
                choice = asked.allowFire ? "fire" : "hold";
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
