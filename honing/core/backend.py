"""What a training backend declares it can do.

Campaign rules read a :class:`BackendSupport` before a job starts or a plan edits
weights, so a request the backend cannot serve is refused up front. v1 backends
declare ``uniform`` architectures only: Slime builds its Megatron model from one
configuration shared by every layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

from honing.core.data import ContractKind
from honing.core.errors import require, require_choice
from honing.core.numerics import NumericsMode
from honing.core.training import AlgorithmName
from honing.core.weights import WeightFormat

AlgorithmSupport = Literal["native", "hook", "unsupported"]
LANGUAGE_MODEL_CONTRACTS: frozenset[ContractKind] = frozenset(
    {"token_sequences", "on_policy_samples", "grouped_on_policy_samples", "preference_pairs"}
)
ArchitectureSupport = Literal["uniform", "heterogeneous"]


@dataclass(frozen=True)
class BackendSupport:
    """The capabilities one backend declares.

    ``hook`` means the backend runs the algorithm through an extension point
    rather than a built-in path, for example Slime's custom-loss hook.
    ``contracts`` are the data contracts it can train on. The default is every
    language-model contract; a backend that trains decision models adds
    ``typed_decisions``, so a job never reaches a trainer that cannot read its data.
    """

    name: str
    architectures: ArchitectureSupport
    train_numerics: frozenset[NumericsMode]
    weight_formats: frozenset[WeightFormat]
    algorithms: tuple[tuple[AlgorithmName, AlgorithmSupport], ...]
    contracts: frozenset[ContractKind] = LANGUAGE_MODEL_CONTRACTS

    def __post_init__(self) -> None:
        require(bool(self.name), "BackendSupport.name must be set")
        require_choice(self.architectures, get_args(ArchitectureSupport), "BackendSupport.architectures")
        names = [name for name, _ in self.algorithms]
        require(len(set(names)) == len(names), "a backend declares each algorithm once")
        for name, support in self.algorithms:
            require_choice(name, get_args(AlgorithmName), "BackendSupport algorithm")
            require_choice(support, get_args(AlgorithmSupport), f"support for {name}")
        require(len(self.contracts) > 0, "a backend trains on at least one data contract")
        for contract in self.contracts:
            require_choice(contract, get_args(ContractKind), "BackendSupport contract")

    def support_for(self, algorithm: AlgorithmName) -> AlgorithmSupport:
        return dict(self.algorithms).get(algorithm, "unsupported")
