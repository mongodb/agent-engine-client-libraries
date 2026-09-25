# Agent Engine Client Libraries

This repository publishes the Atlas Agent Engine CLI and the container images
used for local development.

## Install the CLI

Download the binary for your operating system from the
[Releases](../../releases) page. After making the downloaded binary executable,
run `agentengine version` to verify the installed version.

## Container images

Local-development images are published at
`ghcr.io/mongodb/agent-engine-client-libraries`. Pulling those images does not
require a registry login.

## Release archives

The root `.gitattributes` excludes all repository content from GitHub-generated
source archives. Only release binaries are intended for distribution here.
