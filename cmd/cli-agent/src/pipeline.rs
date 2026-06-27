use crate::environment::SessionContext;
use crate::ipc::query_sidecar;
use crate::path_validate::validate_workspace_path;
use crate::policy::{audit_policy_for_request, load_policy_rule};
use crate::storage::{
    fetch_trusted_template_by_doc_id, find_observed_correction_for_prompt,
    find_or_create_session_record, find_recent_observing_request, insert_request_record,
    mark_request_observing, mark_request_security_blocked, record_observed_correction,
};
use crate::validate::{bind_template_slots, compile_template_argv, path_like_slot_names, BoundSlots};
use anyhow::{Context, Result};
use rusqlite::Connection;
use std::collections::BTreeMap;
use std::path::Path;

/// How long after a natural-language miss a manual `ai-run` command is still
/// plausibly "the correction" for it. Matches the passive-observation window
/// from the architectural spec: long enough to cover "tried the wrong thing,
/// then immediately ran the right one," short enough that an unrelated
/// command typed minutes later doesn't get attributed to an old miss.
pub const OBSERVATION_WINDOW_MINUTES: i64 = 10;

/// A fully validated, ready-to-confirm command. Everything that produced
/// this has already gone through trusted template reload, slot validation,
/// workspace path boundary checks, and policy audit. The caller still owns
/// the interactive preview/confirm/execute step and final approval marking.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedCommand {
    pub request_id: String,
    pub argv: Vec<String>,
}

/// What `resolve_natural_language_command` decided to do with a prompt.
/// Exactly one of three things happens to every natural-language prompt:
/// it resolves to a real, fully validated command; it has been seen before
/// with a known manual correction, pending the user's explicit confirmation
/// to learn it; or nothing matched and the request is now `OBSERVING`,
/// waiting to see what the user does manually next.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NaturalLanguageOutcome {
    Resolved(ResolvedCommand),
    LearnFromObservedCorrection {
        request_id: String,
        suggested_argv: Vec<String>,
    },
    Observing {
        request_id: String,
    },
}

/// Carries a natural-language prompt through the entire guardrail pipeline:
///
/// 1. records the session and the request,
/// 2. asks the untrusted Python sidecar for a candidate template + slots,
/// 3. discards all trust in that answer and reloads the real template by id,
/// 4. validates the untrusted slots against the trusted schema,
/// 5. compiles the trusted template + bound slots into a literal argv,
/// 6. re-checks any path-shaped slot against the workspace boundary, and
/// 7. audits the compiled argv and inherited environment against policy.
///
/// Before any of that: if this exact prompt text was seen before and a
/// manual correction was observed for it (see `observe_manual_command`),
/// that takes priority and the sidecar is never even queried — there is
/// nothing left to ask it. If the sidecar comes back with no candidate at
/// all, the request becomes `OBSERVING` rather than `SECURITY_BLOCKED`:
/// nothing unsafe was attempted, the system just doesn't know what was
/// meant yet.
///
/// Every other failure path here marks the request `SECURITY_BLOCKED`
/// before returning an error, so a rejected prompt is always auditable.
/// Nothing in this function executes a process or prompts the user — that
/// stays in the interactive caller so this function can be exercised
/// directly in tests.
pub fn resolve_natural_language_command(
    conn: &mut Connection,
    context: &SessionContext,
    socket_path: &Path,
    raw_prompt: &str,
) -> Result<NaturalLanguageOutcome> {
    if let Some(correction) = find_observed_correction_for_prompt(conn, raw_prompt)
        .context("failed to check for a previously observed correction")?
    {
        return Ok(NaturalLanguageOutcome::LearnFromObservedCorrection {
            request_id: correction.request_id,
            suggested_argv: correction.argv,
        });
    }

    let session_id = find_or_create_session_record(conn, context)
        .context("failed to record session before natural-language lookup")?;
    let request_id = insert_request_record(conn, &session_id, raw_prompt)
        .context("failed to record request before natural-language lookup")?;

    let proposal = query_sidecar(socket_path, &request_id, raw_prompt, 5)
        .context("failed to reach the sidecar for natural-language interpretation")?;

    let candidate_template_id = match proposal.intent_proposal.candidate_template_id {
        Some(id) => id,
        None => {
            mark_request_observing(conn, &request_id).with_context(|| {
                format!(
                    "sidecar found no matching template; failed to mark request_id={} OBSERVING",
                    request_id
                )
            })?;
            return Ok(NaturalLanguageOutcome::Observing { request_id });
        }
    };

    let template = fetch_trusted_template_by_doc_id(conn, &candidate_template_id)
        .context("failed to reload the trusted template named by the sidecar")?;

    let raw_slots_json = serde_json::to_string(&proposal.intent_proposal.raw_untrusted_slots)
        .context("failed to re-encode sidecar slot map for trusted validation")?;

    let bound = match bind_template_slots(
        &template.trust_state,
        &template.slot_schema_json,
        &raw_slots_json,
    ) {
        Ok(bound) => bound,
        Err(err) => {
            mark_request_security_blocked(conn, &request_id).with_context(|| {
                format!(
                    "slot validation failed with {}; failed to mark request_id={} SECURITY_BLOCKED",
                    err, request_id
                )
            })?;
            return Err(err).context("sidecar slots failed trusted schema validation");
        }
    };

    let final_argv = match compile_template_argv(&template.template_argv_json, &bound) {
        Ok(argv) => argv,
        Err(err) => {
            mark_request_security_blocked(conn, &request_id).with_context(|| {
                format!(
                    "template compilation failed with {}; failed to mark request_id={} SECURITY_BLOCKED",
                    err, request_id
                )
            })?;
            return Err(err).context("trusted template could not be compiled into a final command");
        }
    };

    let workspace_root = context
        .canonical_git_root
        .clone()
        .unwrap_or_else(|| context.canonical_cwd.clone());
    if let Err(err) = validate_path_slots(&workspace_root, &template.slot_schema_json, &bound) {
        mark_request_security_blocked(conn, &request_id).with_context(|| {
            format!(
                "path validation failed with {}; failed to mark request_id={} SECURITY_BLOCKED",
                err, request_id
            )
        })?;
        return Err(err).context("a path-like slot escaped the workspace boundary");
    }

    let policy = load_policy_rule(conn, &template.policy_rule_id)
        .context("failed to load the trusted policy rule for this template")?;
    let inherited_env: BTreeMap<String, String> = std::env::vars().collect();
    audit_policy_for_request(conn, &request_id, &policy, &final_argv, &inherited_env)
        .context("compiled command failed the trusted policy audit")?;

    Ok(NaturalLanguageOutcome::Resolved(ResolvedCommand {
        request_id,
        argv: final_argv,
    }))
}

/// Called after a literal `ai-run <command>` actually executes. If this
/// terminal session has a recent `OBSERVING` request (a natural-language
/// prompt that just missed), this command is recorded as its candidate
/// correction — but only the first such command, and only ever inside
/// `OBSERVATION_WINDOW_MINUTES`. Returns the original prompt text when a
/// *new* correction was recorded, so the caller can let the user know it
/// was noted, or `None` when there was nothing to observe or a correction
/// was already recorded for that miss.
///
/// This never executes anything itself and never blocks the command that
/// already ran: any failure here is the caller's to decide whether to
/// surface, not a reason to treat the just-run command as having failed.
pub fn observe_manual_command(
    conn: &Connection,
    context: &SessionContext,
    argv: &[String],
) -> Result<Option<String>> {
    let session_id = find_or_create_session_record(conn, context)
        .context("failed to resolve session while observing a manual command")?;

    let Some(observing) = find_recent_observing_request(conn, &session_id, OBSERVATION_WINDOW_MINUTES)
        .context("failed to look up an active OBSERVING request")?
    else {
        return Ok(None);
    };

    let recorded = record_observed_correction(conn, &observing.request_id, argv)
        .context("failed to record observed correction")?;

    Ok(recorded.then_some(observing.raw_user_prompt))
}

fn validate_path_slots(
    workspace_root: &Path,
    slot_schema_json: &str,
    bound: &BoundSlots,
) -> Result<()> {
    for name in path_like_slot_names(slot_schema_json)? {
        if let Some(value) = bound.get(&name) {
            validate_workspace_path(workspace_root, value)
                .with_context(|| format!("path slot '{}' failed workspace boundary validation", name))?;
        }
    }
    Ok(())
}
