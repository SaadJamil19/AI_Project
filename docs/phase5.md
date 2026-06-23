# Phase 5: Path Verification and Policy Auditing

Phase 5 adds trusted Rust gates that run after slot binding and before command
preview or execution. Python retrieval and LLM parsing remain untrusted; Rust
owns filesystem boundaries, policy allowlists, and terminal-safe rendering.

## Component Path Validation

`cmd/cli-agent/src/path_validate.rs` validates an untrusted path against a
trusted canonical workspace root.

The validator:

- canonicalizes the workspace root first
- rejects `..` / `Component::ParentDir` before touching the filesystem
- rejects unsupported prefix/root components that cannot be safely interpreted
- walks existing components one by one with `symlink_metadata`
- blocks any existing component that is a symbolic link
- canonicalizes each existing component and checks `Path::starts_with`
- supports non-existent targets by finding the nearest existing parent, then
  appending only normal child components

This prevents `src/../../outside`, absolute-path escapes, and hidden symlink
pivots from crossing the workspace boundary.

## Policy Auditing

`cmd/cli-agent/src/policy.rs` loads a policy row from `policy_rules` and audits
the final compiled argv plus inherited environment.

The policy engine checks:

- executable binary identity or explicit binary allowlist
- expected subcommand path
- flag allowlists and blocklists
- environment variable allowlists and blocklists
- privilege alteration tokens such as `sudo`, `--privileged`, or root user args
- network-like arguments such as URLs and publish/network flags
- recursive destructive signatures such as `-rf`, `--recursive`, and `--force`

Any infraction returns a typed `PolicyError` and blocks downstream execution.

## Terminal Preview Sanitization

Terminal preview must not render raw model or user tokens directly.
`sanitize_terminal_preview_token` preserves printable Unicode while escaping
dangerous control characters.

Examples:

- `ESC` becomes `\x1B`
- NUL becomes `\0`
- newline and carriage return become `\n` and `\r`
- other hidden controls become `\u{XXXX}`
- bidirectional override controls such as RLO `U+202E`, LRO `U+202D`,
  LRE `U+202A`, RLE `U+202B`, and PDF `U+202C` are escaped as `\u{202E}` style
  sequences

Valid international Unicode paths remain readable; the sanitizer only escapes
control characters.

## Request-Bound Policy Failures

`audit-policy` now requires a request id:

```bash
cli-agent audit-policy <request-id> <rule-id> <argv-json> [env-json]
```

If the trusted policy engine returns any `PolicyError`, the CLI opens a writable
SQLite connection and transitions the matching `request_records.execution_status`
to `SECURITY_BLOCKED` inside a transaction before returning the error. This keeps
policy blocks auditable.

## Host Environment Audit

The policy engine audits both the explicit environment JSON passed to the CLI and
the active host process environment from `std::env::vars()`. Blocklisted runtime
variables such as loader injection hooks are rejected even if they were not
included in the JSON payload.

## CLI Verification Commands

The Rust client exposes testable gates:

```bash
cli-agent validate-path <workspace-root> <path>
cli-agent audit-policy req_123 policy_git_restore '["git","restore","src/main.rs"]' '{"PATH":"/usr/bin"}'
cli-agent preview-token $'src/file.rs\x1b[2J'
cli-agent preview-argv '["git","restore","src/international-file.rs"]'
```

These commands are diagnostic wrappers around the same trusted Rust modules the
eventual execution path should call.

## Verification

Run:

```bash
CARGO_TARGET_DIR=/tmp/semantic-cli-agent-target-phase5 cargo test
```

The Phase 5 tests assert that directory traversal is blocked and ANSI escape
sequences are neutralized while printable Unicode is preserved.
