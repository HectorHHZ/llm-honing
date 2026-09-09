# Security Policy

## Reporting a vulnerability

Do not report security vulnerabilities through public GitHub issues,
discussions, or pull requests.

Report privately through
[GitHub Security Advisories](https://github.com/HectorHHZ/llm-honing/security/advisories/new).
Include the affected version or commit, a description of the impact, and the
smallest reproduction you have. You will get an acknowledgement within five
working days and an assessment of severity and remediation after triage.

Please do not share exploit details publicly until a fix is available and a
disclosure timeline has been agreed.

## Scope

Honing runs training and pruning workloads that read model checkpoints,
datasets, and agent-produced traffic. Reports are in scope when they concern:

- execution of untrusted input as code (dataset contents, harness proposals,
  recorded traffic, or a checkpoint's embedded configuration);
- privilege or trust-boundary escapes from a training or episode sandbox;
- leakage of credentials, tokens, or user data through logs, artifacts,
  manifests, or telemetry; or
- publication of an artifact that bypasses the gate or the commit protocol.

Denial of service caused by an operator's own resource configuration, and
issues that require an already-privileged account on the host, are normally
out of scope.
