use serde::Deserialize;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

pub const DEFAULT_MAX_SLOT_BYTES: usize = 512;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum SlotValidationError {
    #[error("template lifecycle state blocks execution: {0}")]
    TemplateRevoked(String),
    #[error("slot schema JSON is invalid: {0}")]
    InvalidSchemaJson(String),
    #[error("raw slot JSON is invalid: {0}")]
    InvalidRawSlotJson(String),
    #[error("slot definition at index {index} is invalid: {reason}")]
    InvalidSlotDefinition { index: usize, reason: String },
    #[error("duplicate slot schema entry: {0}")]
    DuplicateSlot(String),
    #[error("required slot is missing: {0}")]
    MissingRequiredSlot(String),
    #[error("unknown untrusted slot was supplied: {0}")]
    UnknownSlot(String),
    #[error("slot {slot} exceeds byte limit {max_bytes}; got {actual_bytes}")]
    SlotTooLong {
        slot: String,
        max_bytes: usize,
        actual_bytes: usize,
    },
    #[error("slot {slot} does not match any allowed format")]
    FormatRejected { slot: String },
    #[error("slot {slot} must be an integer JSON primitive or integer string")]
    IntegerExpected { slot: String },
    #[error("slot {slot} integer value {value} is outside allowed bounds")]
    IntegerOutOfBounds { slot: String, value: i64 },
    #[error("slot {slot} must be a JSON string")]
    NonStringSlot { slot: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundSlots {
    values: BTreeMap<String, String>,
}

impl BoundSlots {
    pub fn get(&self, key: &str) -> Option<&str> {
        self.values.get(key).map(String::as_str)
    }

    pub fn into_inner(self) -> BTreeMap<String, String> {
        self.values
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum SlotSchemaRoot {
    Array(Vec<SlotRule>),
    Object { slots: Vec<SlotRule> },
}

#[derive(Debug, Deserialize)]
pub struct SlotRule {
    name: String,
    #[serde(default = "default_slot_kind")]
    kind: String,
    #[serde(default)]
    required: bool,
    #[serde(default)]
    max_bytes: Option<usize>,
    #[serde(default)]
    min_value: Option<i64>,
    #[serde(default)]
    max_value: Option<i64>,
    #[serde(default)]
    allowed_formats: Vec<String>,
}

fn default_slot_kind() -> String {
    "string".to_owned()
}

pub fn bind_and_validate_slots(
    slot_schema_json: &str,
    raw_untrusted_slots_json: &str,
) -> Result<BoundSlots, SlotValidationError> {
    let schema = parse_schema(slot_schema_json)?;
    let raw_slots = parse_raw_slots(raw_untrusted_slots_json)?;
    bind_raw_map(schema, raw_slots)
}

pub fn bind_template_slots(
    trust_state: &str,
    slot_schema_json: &str,
    raw_untrusted_slots_json: &str,
) -> Result<BoundSlots, SlotValidationError> {
    validate_template_lifecycle(trust_state)?;
    bind_and_validate_slots(slot_schema_json, raw_untrusted_slots_json)
}

pub fn validate_template_lifecycle(trust_state: &str) -> Result<(), SlotValidationError> {
    match trust_state {
        "DISABLED" | "REVOKED" => Err(SlotValidationError::TemplateRevoked(
            trust_state.to_owned(),
        )),
        _ => Ok(()),
    }
}

pub fn bind_raw_map(
    schema: Vec<SlotRule>,
    raw_slots: BTreeMap<String, Value>,
) -> Result<BoundSlots, SlotValidationError> {
    let mut seen = BTreeSet::new();
    let mut rules_by_name = BTreeMap::new();

    for (index, rule) in schema.into_iter().enumerate() {
        validate_rule(index, &rule)?;
        if !seen.insert(rule.name.clone()) {
            return Err(SlotValidationError::DuplicateSlot(rule.name));
        }
        rules_by_name.insert(rule.name.clone(), rule);
    }

    for key in raw_slots.keys() {
        if !rules_by_name.contains_key(key) {
            return Err(SlotValidationError::UnknownSlot(key.clone()));
        }
    }

    let mut values = BTreeMap::new();
    for (name, rule) in rules_by_name {
        let Some(value) = raw_slots.get(&name) else {
            if rule.required {
                return Err(SlotValidationError::MissingRequiredSlot(name));
            }
            continue;
        };

        let raw = canonicalize_slot_value(&name, &rule, value)?;

        let max_bytes = rule.max_bytes.unwrap_or(DEFAULT_MAX_SLOT_BYTES);
        let actual_bytes = raw.len();
        if actual_bytes > max_bytes {
            return Err(SlotValidationError::SlotTooLong {
                slot: name,
                max_bytes,
                actual_bytes,
            });
        }

        if !rule.allowed_formats.is_empty()
            && !rule
                .allowed_formats
                .iter()
                .any(|format| value_matches_format(&raw, format))
        {
            return Err(SlotValidationError::FormatRejected { slot: name });
        }

        values.insert(name, raw);
    }

    Ok(BoundSlots { values })
}

fn parse_schema(slot_schema_json: &str) -> Result<Vec<SlotRule>, SlotValidationError> {
    let root: SlotSchemaRoot = serde_json::from_str(slot_schema_json)
        .map_err(|err| SlotValidationError::InvalidSchemaJson(err.to_string()))?;
    match root {
        SlotSchemaRoot::Array(slots) | SlotSchemaRoot::Object { slots } => Ok(slots),
    }
}

fn parse_raw_slots(raw_untrusted_slots_json: &str) -> Result<BTreeMap<String, Value>, SlotValidationError> {
    serde_json::from_str(raw_untrusted_slots_json)
        .map_err(|err| SlotValidationError::InvalidRawSlotJson(err.to_string()))
}

fn validate_rule(index: usize, rule: &SlotRule) -> Result<(), SlotValidationError> {
    if rule.name.trim().is_empty() {
        return Err(SlotValidationError::InvalidSlotDefinition {
            index,
            reason: "slot name must not be empty".to_owned(),
        });
    }
    if !matches!(rule.kind.as_str(), "string" | "integer") {
        return Err(SlotValidationError::InvalidSlotDefinition {
            index,
            reason: format!("unsupported slot kind {}", rule.kind),
        });
    }
    if matches!(rule.max_bytes, Some(0)) {
        return Err(SlotValidationError::InvalidSlotDefinition {
            index,
            reason: "max_bytes must be greater than zero".to_owned(),
        });
    }
    if let (Some(min), Some(max)) = (rule.min_value, rule.max_value) {
        if min > max {
            return Err(SlotValidationError::InvalidSlotDefinition {
                index,
                reason: "min_value must be less than or equal to max_value".to_owned(),
            });
        }
    }
    for format in &rule.allowed_formats {
        if !is_supported_format(format) {
            return Err(SlotValidationError::InvalidSlotDefinition {
                index,
                reason: format!("unsupported allowed format {}", format),
            });
        }
    }
    Ok(())
}

fn canonicalize_slot_value(
    slot: &str,
    rule: &SlotRule,
    value: &Value,
) -> Result<String, SlotValidationError> {
    if rule.kind == "integer" || rule.allowed_formats.iter().any(|format| format == "integer") {
        let integer = match value {
            Value::Number(number) => number
                .as_i64()
                .ok_or_else(|| SlotValidationError::IntegerExpected {
                    slot: slot.to_owned(),
                })?,
            Value::String(raw) => raw
                .parse::<i64>()
                .map_err(|_| SlotValidationError::IntegerExpected {
                    slot: slot.to_owned(),
                })?,
            _ => {
                return Err(SlotValidationError::IntegerExpected {
                    slot: slot.to_owned(),
                })
            }
        };

        if rule.min_value.is_some_and(|min| integer < min)
            || rule.max_value.is_some_and(|max| integer > max)
        {
            return Err(SlotValidationError::IntegerOutOfBounds {
                slot: slot.to_owned(),
                value: integer,
            });
        }

        return Ok(integer.to_string());
    }

    value
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| SlotValidationError::NonStringSlot {
            slot: slot.to_owned(),
        })
}

fn is_supported_format(format: &str) -> bool {
    matches!(
        format,
        "ascii" | "no_nul" | "relative_path" | "path" | "filename" | "git_ref" | "flag" | "integer" | "safe_token"
    )
}

fn value_matches_format(value: &str, format: &str) -> bool {
    match format {
        "ascii" => value.is_ascii(),
        "no_nul" => !value.as_bytes().contains(&0),
        "relative_path" => is_relative_path(value),
        "path" => is_path_like(value),
        "filename" => is_filename(value),
        "git_ref" => is_git_ref(value),
        "flag" => is_flag(value),
        "integer" => value.parse::<i64>().is_ok(),
        "safe_token" => is_safe_token(value),
        _ => false,
    }
}

fn is_relative_path(value: &str) -> bool {
    if value.is_empty()
        || value.starts_with('/')
        || value.starts_with('\\')
        || value.contains('\0')
        || value.contains("..")
    {
        return false;
    }
    value
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-' | '/' | '\\'))
}

fn is_path_like(value: &str) -> bool {
    !value.is_empty()
        && !value.contains('\0')
        && value
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-' | '/' | '\\' | ':'))
}

fn is_filename(value: &str) -> bool {
    !value.is_empty()
        && !value.contains('\0')
        && !value.contains('/')
        && !value.contains('\\')
        && value
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
}

fn is_git_ref(value: &str) -> bool {
    if value.is_empty()
        || value.starts_with('-')
        || value.starts_with('/')
        || value.ends_with('/')
        || value.contains("..")
        || value.contains("//")
        || value.contains("@{")
        || value.ends_with(".lock")
        || value.contains('\\')
        || value.contains('\0')
    {
        return false;
    }
    value
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-' | '/'))
}

fn is_flag(value: &str) -> bool {
    value.starts_with("--")
        && value.len() > 2
        && value
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_'))
}

fn is_safe_token(value: &str) -> bool {
    !value.is_empty()
        && value
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-' | ':' | '@'))
}
