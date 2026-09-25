# Release notes format

One file per release: `v{version}.md`. Required before `publish-prod` is dispatched —
`finalize-release` reads this file to populate the GitHub Release body.

## Template

```markdown
## What's new

- Short bullet describing a user-visible change.
- Another change.

## Bug fixes

- Optional section; omit if empty.

## Breaking changes

- Optional section; omit if empty.
```

Keep entries customer-facing: what changed, not how. Link to docs or issues where helpful.
