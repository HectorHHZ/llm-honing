"""Prevent identical encoded decision inputs from crossing episode splits."""

import copy
import hashlib
import json
import random


def identity(item):
    value = json.dumps([item["ids"], item["markers"]], separators=(",", ":"))
    return hashlib.sha256(value.encode()).hexdigest()


def separate_unique(training, validation):
    def unique(rows):
        result = {}
        for row in rows:
            key = identity(row)
            if key in result and result[key]["label"] != row["label"]:
                raise ValueError("Conflicting labels for the same complete model input")
            result.setdefault(key, row)
        return result

    train, valid = unique(training), unique(validation)
    overlap = set(train) & set(valid)
    for key in overlap:
        if train[key]["label"] != valid[key]["label"]:
            raise ValueError("Conflicting labels across episode splits")
    novel = [row for key, row in valid.items() if key not in train]
    return (
        list(train.values()),
        novel,
        {
            "rawTraining": len(training),
            "rawValidation": len(validation),
            "uniqueTraining": len(train),
            "uniqueValidation": len(valid),
            "removedOverlap": len(overlap),
            "novelValidation": len(novel),
        },
    )


def parameter_order_variant(row, seed):
    """Shuffle training parameter descriptions, preserving cancel and answer meaning.

    Call only for training episodes. The seed depends on inputs, never gold
    answers. This is one fixed permutation per request, not extra examples or
    a runtime action-ranking heuristic. Root choices retain their original order.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError("Parameter shuffle seed must be a nonnegative integer")
    result = copy.deepcopy(row)
    if row["stage"] == 1:
        return result
    for key, question in row["questions"].items():
        criteria = question["criteria"]
        keys = list(criteria)
        if question["type"] != "choice" or not keys or keys[0] != "A" or not criteria["A"].startswith("Cancel "):
            raise ValueError("Expected a staged parameter question with cancel A")
        if set(row["answers"][key]) != {"choice"}:
            raise ValueError("Expected an annotation containing only the gold choice")
        chosen = row["answers"][key]["choice"]
        if chosen not in criteria:
            raise ValueError("Gold choice is not offered")
        digest = hashlib.sha256(
            json.dumps([seed, row["parts"], row["stage"], key, question], sort_keys=True).encode()
        ).hexdigest()
        sources = keys[1:]
        random.Random(int(digest, 16)).shuffle(sources)
        sources.insert(0, "A")
        result["questions"][key]["criteria"] = {
            destination: criteria[source] for destination, source in zip(keys, sources, strict=True)
        }
        result["answers"][key]["choice"] = keys[sources.index(chosen)]
    return result


def select_training_variants(training, validation):
    """Apply variants only AFTER the original-input split/deduplication.

    A variant colliding with retained validation is removed from training; it
    cannot shrink validation or reintroduce an original-input overlap. Reject
    inconsistent labels, then deduplicate the actual training encodings again.
    """
    held_out = {identity(item): item for item in validation}
    selected = []
    removed = 0
    for original in training:
        item = original.get("trainingVariant", original)
        key = identity(item)
        if key in held_out:
            if item["label"] != held_out[key]["label"]:
                raise ValueError("Variant conflicts with validation labels")
            removed += 1
            continue
        selected.append({key: value for key, value in item.items() if key != "trainingVariant"})
    unique, _, audit = separate_unique(selected, [])
    return unique, {
        "originalTraining": len(training),
        "removedVariantValidationOverlap": removed,
        "removedVariantDuplicates": len(selected) - len(unique),
        "actualTraining": audit["uniqueTraining"],
        "validationUnchanged": True,
    }
