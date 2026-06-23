use std::fs;
use std::path::{Component, Path, PathBuf};
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum PathValidationError {
    #[error("workspace root does not exist or cannot be canonicalized: {0}")]
    WorkspaceRootInvalid(String),
    #[error("path contains parent-directory traversal")]
    ParentTraversal,
    #[error("path contains unsupported prefix or root component")]
    UnsupportedComponent,
    #[error("path escapes workspace root")]
    BoundaryEscape,
    #[error("path crosses symbolic link: {0}")]
    SymbolicLink(String),
    #[error("path component is empty")]
    EmptyPath,
    #[error("filesystem metadata error for {path}: {message}")]
    Metadata { path: String, message: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedPath {
    pub workspace_root: PathBuf,
    pub resolved_path: PathBuf,
    pub nearest_existing_parent: PathBuf,
    pub missing_components: Vec<PathBuf>,
}

pub fn validate_workspace_path(
    workspace_root: impl AsRef<Path>,
    raw_untrusted_path: &str,
) -> Result<ValidatedPath, PathValidationError> {
    if raw_untrusted_path.trim().is_empty() {
        return Err(PathValidationError::EmptyPath);
    }

    let canonical_root = fs::canonicalize(workspace_root.as_ref()).map_err(|err| {
        PathValidationError::WorkspaceRootInvalid(format!(
            "{} ({})",
            workspace_root.as_ref().display(),
            err
        ))
    })?;

    reject_unsafe_raw_components(Path::new(raw_untrusted_path))?;

    let candidate = if Path::new(raw_untrusted_path).is_absolute() {
        PathBuf::from(raw_untrusted_path)
    } else {
        canonical_root.join(raw_untrusted_path)
    };

    let relative_components = relative_components(&canonical_root, &candidate)?;
    let mut cursor = canonical_root.clone();
    let mut missing_components = Vec::new();

    for component in relative_components {
        if !missing_components.is_empty() {
            missing_components.push(PathBuf::from(&component));
            cursor.push(&component);
            continue;
        }

        let next = cursor.join(&component);
        match fs::symlink_metadata(&next) {
            Ok(metadata) => {
                if metadata.file_type().is_symlink() {
                    return Err(PathValidationError::SymbolicLink(
                        next.display().to_string(),
                    ));
                }
                let canonical_next = fs::canonicalize(&next).map_err(|err| {
                    PathValidationError::Metadata {
                        path: next.display().to_string(),
                        message: err.to_string(),
                    }
                })?;
                if !canonical_next.starts_with(&canonical_root) {
                    return Err(PathValidationError::BoundaryEscape);
                }
                cursor = canonical_next;
            }
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                missing_components.push(PathBuf::from(&component));
                cursor.push(&component);
            }
            Err(err) => {
                return Err(PathValidationError::Metadata {
                    path: next.display().to_string(),
                    message: err.to_string(),
                })
            }
        }
    }

    let nearest_existing_parent = nearest_existing_parent(&cursor);
    let canonical_parent = fs::canonicalize(&nearest_existing_parent).map_err(|err| {
        PathValidationError::Metadata {
            path: nearest_existing_parent.display().to_string(),
            message: err.to_string(),
        }
    })?;
    if !canonical_parent.starts_with(&canonical_root) || !cursor.starts_with(&canonical_root) {
        return Err(PathValidationError::BoundaryEscape);
    }

    Ok(ValidatedPath {
        workspace_root: canonical_root,
        resolved_path: cursor,
        nearest_existing_parent: canonical_parent,
        missing_components,
    })
}

pub fn sanitize_terminal_preview_token(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    for ch in raw.chars() {
        match ch {
            '\x1b' => out.push_str("\\x1B"),
            '\x00' => out.push_str("\\0"),
            '\u{202A}' => out.push_str("\\u{202A}"),
            '\u{202B}' => out.push_str("\\u{202B}"),
            '\u{202C}' => out.push_str("\\u{202C}"),
            '\u{202D}' => out.push_str("\\u{202D}"),
            '\u{202E}' => out.push_str("\\u{202E}"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push('\t'),
            c if c.is_control() => {
                let code = c as u32;
                out.push_str("\\u{");
                out.push_str(&format!("{code:04X}"));
                out.push('}');
            }
            c => out.push(c),
        }
    }
    out
}

fn reject_unsafe_raw_components(path: &Path) -> Result<(), PathValidationError> {
    for component in path.components() {
        match component {
            Component::ParentDir => return Err(PathValidationError::ParentTraversal),
            Component::Prefix(_) => return Err(PathValidationError::UnsupportedComponent),
            Component::RootDir | Component::CurDir | Component::Normal(_) => {}
        }
    }
    Ok(())
}

fn relative_components(
    canonical_root: &Path,
    candidate: &Path,
) -> Result<Vec<String>, PathValidationError> {
    let relative = if candidate.is_absolute() {
        candidate
            .strip_prefix(canonical_root)
            .map_err(|_| PathValidationError::BoundaryEscape)?
            .to_path_buf()
    } else {
        candidate.to_path_buf()
    };

    let mut components = Vec::new();
    for component in relative.components() {
        match component {
            Component::Normal(value) => components.push(value.to_string_lossy().to_string()),
            Component::CurDir => {}
            Component::ParentDir => return Err(PathValidationError::ParentTraversal),
            Component::Prefix(_) | Component::RootDir => {
                return Err(PathValidationError::UnsupportedComponent)
            }
        }
    }
    Ok(components)
}

fn nearest_existing_parent(path: &Path) -> PathBuf {
    let mut cursor = path;
    while !cursor.exists() {
        let Some(parent) = cursor.parent() else {
            break;
        };
        cursor = parent;
    }
    cursor.to_path_buf()
}
