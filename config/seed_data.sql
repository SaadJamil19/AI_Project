-- Seed data for semantic-cli-agent: core git development templates.
--
-- Usage after creating the database with `cli-agent init`:
--   sqlite3 "$SEMANTIC_CLI_AGENT_HOME/cli-agent.db" < config/seed_data.sql
--
-- Idempotent: INSERT OR IGNORE keeps repeated loads from duplicating rows.

PRAGMA foreign_keys = ON;

BEGIN;

INSERT OR IGNORE INTO policy_rules (
    rule_id,
    binary_name,
    subcommand_path,
    fast_path_allowed,
    required_confirmation_count,
    executable_path_policy_json,
    env_variable_inheritance_json,
    positional_argument_rules_json,
    path_slot_policies_json,
    package_manager_risk_level,
    privilege_risk_level,
    destructive_recursive_level
)
VALUES (
    'policy_git_core',
    'git',
    '',
    1,
    1,
    '{"allowed_binaries":["/usr/bin/git","git"]}',
    '{"allow":[],"block":["LD_PRELOAD","LD_LIBRARY_PATH"]}',
    '{"allowed_flags":["status","restore","commit","-m","checkout","-b","stash","push"],"blocked_flags":[]}',
    '{"allow_network_args":false}',
    'ALLOW',
    'ALLOW',
    'ALLOW'
);

INSERT OR IGNORE INTO unified_documents (
    doc_id,
    source_type,
    binary_name,
    subcommand_path,
    intent_description,
    template_argv_json,
    slot_schema_json,
    policy_rule_id,
    trust_state
)
VALUES (
    'seed_git_status',
    'STATIC_DOCS',
    'git',
    'status',
    'show me the current working directory repository changes and untracked files status',
    '["git","status"]',
    '[]',
    'policy_git_core',
    'STATIC_VERIFIED'
);

INSERT OR IGNORE INTO unified_documents (
    doc_id,
    source_type,
    binary_name,
    subcommand_path,
    intent_description,
    template_argv_json,
    slot_schema_json,
    policy_rule_id,
    trust_state
)
VALUES (
    'seed_git_restore',
    'STATIC_DOCS',
    'git',
    'restore',
    'undo changes or restore a target file back to its last committed state',
    '["git","restore","$target_file"]',
    '[{"name":"target_file","kind":"string","required":true,"max_bytes":256,"allowed_formats":["relative_path"]}]',
    'policy_git_core',
    'STATIC_VERIFIED'
);

INSERT OR IGNORE INTO unified_documents (
    doc_id,
    source_type,
    binary_name,
    subcommand_path,
    intent_description,
    template_argv_json,
    slot_schema_json,
    policy_rule_id,
    trust_state
)
VALUES (
    'seed_git_checkout_branch',
    'STATIC_DOCS',
    'git',
    'checkout -b',
    'create and switch to a new local git development branch branch_name',
    '["git","checkout","-b","$branch_name"]',
    '[{"name":"branch_name","kind":"string","required":true,"max_bytes":128,"allowed_formats":["safe_token"]}]',
    'policy_git_core',
    'STATIC_VERIFIED'
);

INSERT OR IGNORE INTO unified_documents (
    doc_id,
    source_type,
    binary_name,
    subcommand_path,
    intent_description,
    template_argv_json,
    slot_schema_json,
    policy_rule_id,
    trust_state
)
VALUES (
    'seed_git_stash_push',
    'STATIC_DOCS',
    'git',
    'stash push',
    'stash current uncommitted working tree changes for later restoration',
    '["git","stash","push"]',
    '[]',
    'policy_git_core',
    'STATIC_VERIFIED'
);

-- Narrowly-scoped curl policy: allows read-style requests with a method,
-- headers, and a literal data body, but deliberately blocks every flag
-- that writes to or reads from the local filesystem (-o/-O/--output,
-- -T/--upload-file, -K/--config, -F/--form), proxies/redirects traffic
-- through another endpoint (--proxy/-x, --resolve, --connect-to,
-- --unix-socket), or could exfiltrate a local file as a request body
-- (--data-binary, --data-urlencode here; the @filename pattern on ANY
-- data flag is additionally blocked by cmd/cli-agent/src/policy.rs's
-- is_data_flag/is_local_file_reference check, not just by omission here).
-- There is deliberately no host allowlist: any host is reachable once a
-- curl command passes this policy and is confirmed at the y/N prompt.
INSERT OR IGNORE INTO policy_rules (
    rule_id,
    binary_name,
    subcommand_path,
    fast_path_allowed,
    required_confirmation_count,
    executable_path_policy_json,
    env_variable_inheritance_json,
    positional_argument_rules_json,
    path_slot_policies_json,
    package_manager_risk_level,
    privilege_risk_level,
    destructive_recursive_level
)
VALUES (
    'policy_curl_core',
    'curl',
    '',
    0,
    1,
    '{"allowed_binaries":["/usr/bin/curl","curl"]}',
    '{"allow":[],"block":["LD_PRELOAD","LD_LIBRARY_PATH"]}',
    '{"allowed_flags":["-s","-S","--silent","-X","--request","-H","--header","-d","--data","--data-raw","-G","--get","-A","--user-agent","-i","-I","--max-time","--connect-timeout","-L","--location"],"blocked_flags":["-o","-O","--output","-T","--upload-file","-K","--config","--resolve","--connect-to","-F","--form","--unix-socket","-x","--proxy","--data-binary","--data-urlencode"]}',
    '{"allow_network_args":true}',
    'BLOCK',
    'BLOCK',
    'BLOCK'
);

COMMIT;
