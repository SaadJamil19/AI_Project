# semantic-cli-agent shell integration for zsh.
# Source this file from ~/.zshrc after building cli-agent.

_semantic_cli_agent_bin() {
  if [[ -n "${SEMANTIC_CLI_AGENT_BIN:-}" ]]; then
    print -r -- "$SEMANTIC_CLI_AGENT_BIN"
  else
    print -r -- "cli-agent"
  fi
}

ai-run() {
  local bin
  bin="$(_semantic_cli_agent_bin)"
  if (( $# == 0 )); then
    print -u2 "usage: ai-run <command> [args...]"
    return 2
  fi
  "$bin" ai-run "$@"
}

ai-learn() {
  local bin
  bin="$(_semantic_cli_agent_bin)"
  if (( $# < 3 )) || [[ "$1" != "--request-id" ]]; then
    print -u2 'usage: ai-learn --request-id <id> "<corrected command>"'
    return 2
  fi
  "$bin" ai-learn "$@"
}
