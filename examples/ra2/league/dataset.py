"""Prevent identical encoded decision inputs from crossing episode splits."""

import hashlib
import json


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
