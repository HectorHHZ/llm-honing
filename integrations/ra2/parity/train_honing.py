"""Train with Honing's LocalJevTrainer on the same matches #49's train_laya.py reads."""

import json
import os
import sys
from pathlib import Path

from honing.backends.jev.laya import laya_snapshot
from honing.backends.jev.trainer import JevTrainerSettings, LocalJevTrainer
from honing.backends.ra2 import RA2DemonstrationProvider
from honing.campaign import rules
from honing.core import TrainJobSpec, requirements_for

PR49 = Path(os.environ["PR49_DIR"])
deterministic = "--deterministic" in sys.argv
argv = [a for a in sys.argv if a != "--deterministic"]
out, gpu, epochs, matches = Path(argv[1]), argv[2], int(argv[3]), [Path(p) for p in argv[4:]]
record_path = PR49 / ".cache/models/typed-decisions.json"
record = json.loads(record_path.read_text())
protocol = json.loads((PR49 / "league/protocol.json").read_text())
job = TrainJobSpec("sft", steps=epochs, contract="typed_decisions")
trainer = LocalJevTrainer(
    JevTrainerSettings(
        laya_source=Path(os.environ.get("LAYA_SOURCE_DIR", str(PR49 / "deps/laya"))),
        pins_path=PR49 / "pins.json",
        model_record=record_path,
        work_dir=out,
        gpu=gpu,
        deterministic=deterministic,
    )
)
rules.algorithm_requirements_met(job, trainer.support)
batch = RA2DemonstrationProvider(matches, protocol).next_batch(requirements_for(job).data_contract())
rules.batch_meets_contract(job, batch)
result = trainer.train(job, laya_snapshot(Path(record["path"])), batch)
print(
    json.dumps(
        {
            "checkpoint": result.snapshot.checkpoint.location,
            "checksum": result.snapshot.checkpoint.checksum,
            "parameters": result.snapshot.parameters,
            "metrics": dict(result.metrics),
        }
    )
)
