use rusqlite::{params, Connection};
use serde::de::DeserializeOwned;
use serde::Deserialize;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum PolicyError {
    #[error("policy rule JSON is invalid in {field}: {message}")]
    InvalidJson { field: &'static str, message: String },
    #[error("compiled argv is empty")]
    EmptyArgv,
    #[error("binary {actual} does not match policy binary {expected}")]
    BinaryMismatch { expected: String, actual: String },
    #[error("subcommand path does not match policy")]
    SubcommandMismatch,
    #[error("flag is not allowlisted: {0}")]
    FlagNotAllowlisted(String),
    #[error("flag is explicitly blocked: {0}")]
    FlagBlocked(String),
    #[error("environment variable is not allowlisted: {0}")]
    EnvNotAllowlisted(String),
    #[error("privilege alteration argument blocked: {0}")]
    PrivilegeRisk(String),
    #[error("network operation argument blocked: {0}")]
    NetworkRisk(String),
    #[error("recursive destructive argument blocked: {0}")]
    DestructiveRisk(String),
    #[error("data-flag argument reads a local file for exfiltration: {0} {1}")]
    DataFlagFileRead(String, String),
    #[error("policy rule not found: {0}")]
    RuleNotFound(String),
    #[error("failed to mark request SECURITY_BLOCKED: {0}")]
    RequestStatusUpdate(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyRule {
    pub rule_id: String,
    pub binary_name: String,
    pub subcommand_path: String,
    pub executable_path_policy: ExecutablePathPolicy,
    pub env_policy: EnvPolicy,
    pub positional_policy: PositionalPolicy,
    pub path_slot_policy: PathSlotPolicy,
    pub package_manager_risk_level: RiskLevel,
    pub privilege_risk_level: RiskLevel,
    pub destructive_recursive_level: RiskLevel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskLevel {
    Allow,
    Block,
}

#[derive(Debug, Clone, Deserialize, Default, PartialEq, Eq)]
pub struct ExecutablePathPolicy {
    #[serde(default)]
    pub allowed_binaries: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Default, PartialEq, Eq)]
pub struct EnvPolicy {
    #[serde(default)]
    pub allow: Vec<String>,
    #[serde(default)]
    pub block: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Default, PartialEq, Eq)]
pub struct PositionalPolicy {
    #[serde(default)]
    pub allowed_flags: Vec<String>,
    #[serde(default)]
    pub blocked_flags: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Default, PartialEq, Eq)]
pub struct PathSlotPolicy {
    #[serde(default)]
    pub allow_network_args: bool,
}

pub fn load_policy_rule(conn: &Connection, rule_id: &str) -> Result<PolicyRule, PolicyError> {
    let row = conn
        .query_row(
            r#"
            SELECT
                rule_id,
                binary_name,
                subcommand_path,
                executable_path_policy_json,
                env_variable_inheritance_json,
                positional_argument_rules_json,
                path_slot_policies_json,
                package_manager_risk_level,
                privilege_risk_level,
                destructive_recursive_level
            FROM policy_rules
            WHERE rule_id = ?1
            LIMIT 1
            "#,
            params![rule_id],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, String>(5)?,
                    row.get::<_, String>(6)?,
                    row.get::<_, String>(7)?,
                    row.get::<_, String>(8)?,
                    row.get::<_, String>(9)?,
                ))
            },
        )
        .map_err(|_| PolicyError::RuleNotFound(rule_id.to_owned()))?;

    Ok(PolicyRule {
        rule_id: row.0,
        binary_name: row.1,
        subcommand_path: row.2,
        executable_path_policy: parse_policy_json("executable_path_policy_json", &row.3)?,
        env_policy: parse_policy_json("env_variable_inheritance_json", &row.4)?,
        positional_policy: parse_policy_json("positional_argument_rules_json", &row.5)?,
        path_slot_policy: parse_policy_json("path_slot_policies_json", &row.6)?,
        package_manager_risk_level: parse_risk_level(&row.7),
        privilege_risk_level: parse_risk_level(&row.8),
        destructive_recursive_level: parse_risk_level(&row.9),
    })
}

pub fn audit_policy(
    policy: &PolicyRule,
    argv: &[String],
    inherited_env: &BTreeMap<String, String>,
) -> Result<(), PolicyError> {
    audit_argv(policy, argv)?;
    audit_environment(policy, inherited_env)?;
    let host_env: BTreeMap<String, String> = std::env::vars().collect();
    audit_environment(policy, &host_env)?;
    Ok(())
}

/// Audits only the compiled argv against policy, with no environment checks.
///
/// Used at template-ingestion time (`ai-learn`), where there is no live
/// execution environment to inspect yet; only the corrected argv itself is
/// being vetted before it can ever be persisted as a reusable template.
pub fn audit_argv_policy(policy: &PolicyRule, argv: &[String]) -> Result<(), PolicyError> {
    audit_argv(policy, argv)
}

pub fn audit_policy_for_request(
    conn: &mut Connection,
    request_id: &str,
    policy: &PolicyRule,
    argv: &[String],
    inherited_env: &BTreeMap<String, String>,
) -> Result<(), PolicyError> {
    match audit_policy(policy, argv, inherited_env) {
        Ok(()) => Ok(()),
        Err(err) => {
            crate::storage::mark_request_security_blocked(conn, request_id).map_err(|status_err| {
                PolicyError::RequestStatusUpdate(format!(
                    "policy error was {}; status update failed with {}",
                    err, status_err
                ))
            })?;
            Err(err)
        }
    }
}

fn audit_argv(policy: &PolicyRule, argv: &[String]) -> Result<(), PolicyError> {
    let binary = argv.first().ok_or(PolicyError::EmptyArgv)?;
    if binary != &policy.binary_name
        && !policy
            .executable_path_policy
            .allowed_binaries
            .iter()
            .any(|allowed| allowed == binary)
    {
        return Err(PolicyError::BinaryMismatch {
            expected: policy.binary_name.clone(),
            actual: binary.clone(),
        });
    }

    let subcommands: Vec<&str> = policy.subcommand_path.split_whitespace().collect();
    if !subcommands.is_empty() {
        let actual: Vec<&str> = argv
            .iter()
            .skip(1)
            .take(subcommands.len())
            .map(String::as_str)
            .collect();
        if actual != subcommands {
            return Err(PolicyError::SubcommandMismatch);
        }
    }

    let allowed_flags: BTreeSet<&str> = policy
        .positional_policy
        .allowed_flags
        .iter()
        .map(String::as_str)
        .collect();
    let blocked_flags: BTreeSet<&str> = policy
        .positional_policy
        .blocked_flags
        .iter()
        .map(String::as_str)
        .collect();

    let positional = &argv[1..];
    for (index, arg) in positional.iter().enumerate() {
        if blocked_flags.contains(arg.as_str()) {
            return Err(PolicyError::FlagBlocked(arg.clone()));
        }
        if arg.starts_with('-')
            && !allowed_flags.is_empty()
            && !allowed_flags.contains(arg.as_str())
        {
            return Err(PolicyError::FlagNotAllowlisted(arg.clone()));
        }
        if policy.privilege_risk_level == RiskLevel::Block && is_privilege_arg(arg) {
            return Err(PolicyError::PrivilegeRisk(arg.clone()));
        }
        if policy.package_manager_risk_level == RiskLevel::Block
            && !policy.path_slot_policy.allow_network_args
            && is_network_arg(arg)
        {
            return Err(PolicyError::NetworkRisk(arg.clone()));
        }
        if policy.destructive_recursive_level == RiskLevel::Block
            && is_destructive_recursive_arg(arg)
        {
            return Err(PolicyError::DestructiveRisk(arg.clone()));
        }
        if is_data_flag(arg) {
            if let Some(value) = positional.get(index + 1) {
                if is_local_file_reference(value) {
                    return Err(PolicyError::DataFlagFileRead(arg.clone(), value.clone()));
                }
            }
        }
    }

    Ok(())
}

fn audit_environment(
    policy: &PolicyRule,
    env_map: &BTreeMap<String, String>,
) -> Result<(), PolicyError> {
    let allowed_env: BTreeSet<&str> = policy.env_policy.allow.iter().map(String::as_str).collect();
    let blocked_env: BTreeSet<&str> = policy.env_policy.block.iter().map(String::as_str).collect();
    for key in env_map.keys() {
        if blocked_env.contains(key.as_str())
            || (!allowed_env.is_empty() && !allowed_env.contains(key.as_str()))
        {
            return Err(PolicyError::EnvNotAllowlisted(key.clone()));
        }
    }

    Ok(())
}

pub fn sanitize_preview_tokens(argv: &[String]) -> Vec<String> {
    argv.iter()
        .map(|token| crate::path_validate::sanitize_terminal_preview_token(token))
        .collect()
}

fn parse_policy_json<T>(field: &'static str, raw: &str) -> Result<T, PolicyError>
where
    T: DeserializeOwned + Default,
{
    if raw.trim().is_empty() {
        return Ok(T::default());
    }

    let value: Value = serde_json::from_str(raw).map_err(|err| PolicyError::InvalidJson {
        field,
        message: err.to_string(),
    })?;
    if matches!(value, Value::Array(ref values) if values.is_empty()) {
        return Ok(T::default());
    }

    serde_json::from_value(value).map_err(|err| PolicyError::InvalidJson {
        field,
        message: err.to_string(),
    })
}

fn parse_risk_level(raw: &str) -> RiskLevel {
    match raw {
        "ALLOW" => RiskLevel::Allow,
        _ => RiskLevel::Block,
    }
}

fn is_privilege_arg(arg: &str) -> bool {
    matches!(arg, "sudo" | "su" | "doas" | "--privileged" | "--user=root" | "--as-root")
}

fn is_network_arg(arg: &str) -> bool {
    arg.starts_with("http://")
        || arg.starts_with("https://")
        || matches!(arg, "--network" | "--net" | "--publish" | "-p")
}

fn is_destructive_recursive_arg(arg: &str) -> bool {
    matches!(arg, "-rf" | "-fr" | "--recursive" | "--force")
}

/// Flags whose *value* (the next argv token) is itself untrusted file/data
/// content, not just another flag. curl-family `@filename` syntax means
/// "read this local file and use its contents as the value" - a classic
/// local-file-read-then-send-to-a-remote-host exfiltration primitive that
/// no other check here inspects, since every other check only looks at
/// flag tokens themselves, never the argument that follows one.
fn is_data_flag(arg: &str) -> bool {
    matches!(
        arg,
        "-d" | "--data" | "--data-raw" | "--data-binary" | "--data-urlencode" | "-F" | "--form"
    )
}

fn is_local_file_reference(value: &str) -> bool {
    value.starts_with('@')
}
