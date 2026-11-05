# Draft RFC: Space Invaders action replay format

Status: draft, pending a GitHub RFC issue and explicit maintainer acceptance.
`CONTRIBUTING.md` requires acceptance before merging a persisted format.

The optional integration records a single complete ALE episode as JSON schema
1. The header pins `ALE/SpaceInvaders-v5`, ALE/Gymnasium versions, seed, pooled/native observation mode,
action repeat 4, and sticky-action probability 0.25. Each row records a native
action index, reward, lives, terminal flags, and SHA-256 of the RGB observation.
The footer contains cumulative reward and natural-end status. Replay reruns the
same actions and compares every physical outcome. Truncated or capped games
cannot pass verification.

No controller or model choice is persisted as part of this format. No external
ROM, checkpoint, or sealed incumbent artifact is distributed. Compatibility
across ALE/ROM versions is not promised. A future schema change requires
another review.
