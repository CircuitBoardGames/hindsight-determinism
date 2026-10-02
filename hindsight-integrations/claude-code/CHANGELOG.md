# Changelog

## [Unreleased]

### Added

- `{user_id}` template variable for `retainTags` and `retainMetadata`, resolved
  from the `HINDSIGHT_USER_ID` env var (empty string if unset). Enables
  machine-independent per-user memory scoping without hardcoding user ids in
  `settings.json`.
- `requestTimeoutSeconds` config (env: `HINDSIGHT_REQUEST_TIMEOUT_SECONDS`) to
  override the per-call HTTP timeout used by recall (10s), retain (15s) and the
  knowledge MCP tools. Defaults to `null`, which preserves current per-call
  behavior. Set this when self-hosted Hindsight legitimately takes longer than
  10s under contention (e.g. parallel recalls) so the client doesn't surface
  `read operation timed out` on requests the server completes successfully.
  Does not affect the health check, which stays at 5s. Fixes #1575.

### Changed

- `retain.py` can serve a second harness (omp): optional hook_input keys `retain_context` and
  `retain_tags` replace `retainContext`/`retainTags` for that call, `skip_cadence` lets the caller own
  the turn cadence, and `source_timestamp` is sent as the memory's `timestamp`. The user's config file
  is not touched. Plugin version 0.7.5-airoboros.2.
- Fixed: in full-session mode with `retainEveryNTurns > 1`, only the LAST turn of each unsent suffix was
  retained; the earlier turns were marked sent and lost.

- Retain (and prior-context recall) now strips harness noise from transcripts, not only memory
  tags: `<system-reminder>`, `<cross-session-message>` peer/cron notices, `<task-notification>`,
  `<session-notify>`, local-command wrappers, and whole turns that are Claude Code internal prompts
  (`[SUGGESTION MODE]`, recap, skill-body expansion). Tool results were already excluded unless
  `retainToolCalls` is on. Port of TencentDB-Agent-Memory's sanitizer (MIT).

- Tags that resolve to an empty namespace content (e.g. `"user:"` when
  `HINDSIGHT_USER_ID` is unset) are now dropped from retain requests. Previously
  such tags were sent as-is. Tags without `:` are unaffected.

## [0.1.0] - 2025-03-23

### Added
- Initial release: Claude Code plugin for Hindsight long-term memory
- Auto-recall on every user prompt via `UserPromptSubmit` hook — injects relevant memories as `additionalContext`
- Auto-retain after every response via async `Stop` hook — extracts and stores conversation transcript
- Session lifecycle hooks (`SessionStart` health check, `SessionEnd` daemon cleanup)
- Three connection modes: external API, auto-managed local daemon (`uvx hindsight-embed`), existing local server
- Dynamic bank IDs with configurable granularity (`agent`, `project`, `session`, `channel`, `user`)
- Channel-agnostic: works with Claude Code Channels (Telegram, Discord, Slack) and interactive sessions
- Zero pip dependencies — pure Python stdlib (`urllib`, `fcntl`, `subprocess`)
- 34 configuration options via `settings.json` with env var overrides
- LLM auto-detection from `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `GROQ_API_KEY`
- Chunked retention with sliding window (`retainEveryNTurns` + `retainOverlapTurns`)
- Memory tag stripping to prevent retain feedback loops
