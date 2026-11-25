# Changelog

All notable changes to this package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **Breaking:** the package now requires Node.js 24.22.3 or later. `engines.node`
  was `>=24`, so a consumer on an earlier 24.x installed it and ran the Brotli
  decoder from before the CVE-2026-111497 fix. npm reports the new floor as
  `EBADENGINE`: a warning by default, an install error under `engine-strict`.
- `SessionMessage.role` for human-authored session-history messages is now documented and emitted as `"user"` instead of `"human"`, matching the live invoke/stream `Message.role` vocabulary (`user`, `assistant`, `tool`, `system`).
- `MemoryClient.buildContext` rejects non-positive, non-integer, and non-finite `maxTokens` with `RangeError` before sending the request.
