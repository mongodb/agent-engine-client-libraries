#!/bin/bash
# agentengine-pnpm-workspace.sh: Install and build a TypeScript agent that
# is a package of a pnpm workspace.
#
# The runner base image ships this script, pnpm, and the files next to it.
# It is the one install path for a pnpm workspace agent: the platform builds
# (ECP's generated Dockerfile and the sandbox's install-typescript.sh) and
# `agentengine dev up` are to run it, so a pnpm workspace builds the same
# way in each place. Nothing runs it yet.
#
# Usage:
#   agentengine-pnpm-workspace.sh detect <app_dir> <install_target>
#       Print the workspace root and exit 0 when the agent at
#       <app_dir>/<install_target> is a package of a pnpm workspace inside
#       <app_dir>. Exit 1 when the npm install path applies instead.
#   agentengine-pnpm-workspace.sh install <app_dir> <install_target> [--prod]
#       Install the agent and its workspace dependencies with the runner
#       image's SDK tarballs, then build them. --prod then reinstalls only
#       the agent's production dependencies, as `npm prune --omit=dev` keeps
#       only them on the npm path.
#   agentengine-pnpm-workspace.sh build <app_dir> <install_target>
#       detect, then install --prod when the agent is a workspace package.
#       The platform builds call this: exit 0 means pnpm built the agent,
#       exit 1 means the npm install path applies and nothing ran.
#   agentengine-pnpm-workspace.sh dev <source_dir> <run_dir> <install_target>
#       Copy <source_dir> into <run_dir>, then run install on the copy,
#       keeping devDependencies. `agentengine dev up` passes the developer's
#       checkout as <source_dir>, and nothing here writes to it.
#
# install, build and dev run only the image's pnpm: a workspace that pins
# another package manager, or another pnpm major, fails with exit 2.
#
# detect and build exit 1 only to mean "use npm". Every other failure exits
# non-zero and not 1, and callers must fail the build on it rather than
# fall back to npm.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly script_dir
readonly sdk_dist_dir="${AGENTIC_SDK_DIST_DIR:-/opt/agent-engine-runner-shared/dist}"
readonly pnpmfile="${AGENTIC_PNPMFILE:-${script_dir}/agentengine-sdk.pnpmfile.cjs}"
readonly checks="${script_dir}/agentengine-pnpm-workspace.cjs"
readonly mirror="${script_dir}/agentengine-workspace-mirror.cjs"
readonly store_dir="${AGENTIC_PNPM_STORE_DIR:-/tmp/agentengine-pnpm-store}"
readonly launcher="node_modules/@mongodb-js/agent-engine-runner-shared/dist/launcher.js"

# Every pnpm command reads these, the membership check in resolve_workspace
# included: never download another pnpm for a packageManager pin, and never
# ask the registry for a newer pnpm. install_workspace_agent rejects a pin
# that the image's pnpm cannot honor before it installs anything.
export pnpm_config_pm_on_fail=warn
export pnpm_config_update_notifier=false

die() {
    echo "ERROR: $*" >&2
    exit 2
}

# Sets WORKSPACE_ROOT, AGENT_DIR and AGENT_PACKAGE, or returns 1 when the
# agent is not a package of a pnpm workspace inside app_dir.
#
# An agent takes the pnpm path only when its package.json declares a
# dependency with the workspace: or catalog: protocol: npm cannot install
# either, and every other agent, including a member of a pnpm workspace with
# only registry dependencies, keeps the npm path it has today.
#
# The nearest pnpm-workspace.yaml above the agent is the workspace, as pnpm
# itself resolves it. A pnpm-workspace.yaml in the agent directory makes the
# agent the workspace root, which has no workspace dependencies to install,
# so that layout and an agent that commits a package-lock.json keep the npm
# path. Membership is asked of pnpm rather than parsed here, so the answer
# matches the install.
#
# Callers run this on the left of ||, where bash suspends set -e, so every
# step that can fail says what its failure means: return 1 for "use npm",
# die for anything else.
resolve_workspace() {
    local app_dir=$1 install_target=$2 dir agent listed
    local agent_status=0 member_status=0

    app_dir="$(cd "${app_dir}" && pwd -P)" || die "${1} is not a directory"
    AGENT_DIR="$(cd "${app_dir}/${install_target}" && pwd -P)" || return 1
    case "${AGENT_DIR}/" in
        "${app_dir}/"*) ;;
        *) die "install target ${install_target} is outside ${app_dir}" ;;
    esac
    [[ "${AGENT_DIR}" != "${app_dir}" ]] || return 1
    [[ -f "${AGENT_DIR}/package.json" ]] || return 1
    [[ ! -e "${AGENT_DIR}/package-lock.json" ]] || return 1
    [[ ! -e "${AGENT_DIR}/pnpm-workspace.yaml" ]] || return 1
    # "pnpm" or "npm", a tab, and the package name. An unreadable
    # package.json exits 1 and keeps the npm path, which then fails on it.
    agent="$(node "${checks}" agent "${AGENT_DIR}")" || agent_status=$?
    [[ "${agent_status}" -le 1 ]] || die "could not read ${AGENT_DIR}/package.json"
    [[ "${agent_status}" -eq 0 ]] || return 1
    [[ "${agent%%$'\t'*}" == pnpm ]] || return 1
    AGENT_PACKAGE="${agent#*$'\t'}"

    WORKSPACE_ROOT=
    dir="$(dirname "${AGENT_DIR}")"
    while :; do
        if [[ -f "${dir}/pnpm-workspace.yaml" ]]; then
            WORKSPACE_ROOT="${dir}"
            break
        fi
        [[ "${dir}" != "${app_dir}" ]] || break
        dir="$(dirname "${dir}")"
    done
    [[ -n "${WORKSPACE_ROOT}" ]] || return 1

    [[ -n "${AGENT_PACKAGE}" ]] \
        || die "${AGENT_DIR}/package.json needs a \"name\" to install it from the pnpm workspace at ${WORKSPACE_ROOT}"

    listed="$(cd "${WORKSPACE_ROOT}" \
        && pnpm --filter "${AGENT_PACKAGE}" ls --depth -1 --json)" \
        || die "pnpm could not list the workspace at ${WORKSPACE_ROOT}"
    node "${checks}" member "${listed}" "${AGENT_DIR}" || member_status=$?
    [[ "${member_status}" -le 1 ]] \
        || die "pnpm returned an unreadable project list for ${WORKSPACE_ROOT}"
    return "${member_status}"
}

pnpm_environment() {
    [[ -f "${sdk_dist_dir}/sdk-packages.json" ]] \
        || die "no sdk-packages.json in ${sdk_dist_dir}/ -- rebuild runner-base-typescript-langgraph"
    [[ -f "${pnpmfile}" ]] || die "pnpmfile not found at ${pnpmfile}"
    export AGENTIC_SDK_DIST_DIR="${sdk_dist_dir}"
    export AGENTIC_AGENT_PACKAGE="${AGENT_PACKAGE}"
    export pnpm_config_global_pnpmfile="${pnpmfile}"
    export pnpm_config_store_dir="${store_dir}"
    # Do not let `pnpm run` reinstall between the steps below.
    export pnpm_config_verify_deps_before_run=false
    # pnpm 11 fails the install when a dependency has a build script that the
    # workspace has not approved. The image's SDK can be newer than the one
    # the workspace locked and bring such a dependency, so skip unapproved
    # build scripts with a warning instead. The allowBuilds list in
    # pnpm-workspace.yaml still decides which build scripts run.
    export pnpm_config_strict_dep_builds=false
    # Keep every package in the workspace's own node_modules/.pnpm, linked
    # into each package's node_modules, where the launcher and the SDK
    # checks look. With virtualStoreType: global, or a virtualStoreDir in
    # the store or outside the workspace, packages live where `build`
    # removes them or where the image never copies them, and in `dev` a
    # virtualStoreDir could reach the developer's checkout. With nodeLinker:
    # hoisted, the agent's node_modules has no SDK packages. These variables
    # outrank pnpm-workspace.yaml; a --config flag does not.
    export pnpm_config_virtual_store_type=project
    export pnpm_config_virtual_store_dir=node_modules/.pnpm
    export pnpm_config_node_linker=isolated
}

# Fails unless the runner image's SDK packages are the only ones in the
# workspace and the agent's launcher is in place. `build` runs it again
# after it removes the pnpm store, so a link into the store fails the build.
verify_install() {
    node "${checks}" verify-sdk "${sdk_dist_dir}/sdk-packages.json" "${AGENT_DIR}" "${WORKSPACE_ROOT}" \
        || die "the workspace's Agent Engine SDK packages above are not the runner image's copies. The runner image's copy must be the only one in the workspace. An overrides entry in ${WORKSPACE_ROOT}/pnpm-workspace.yaml is one cause."
    [[ -f "${AGENT_DIR}/${launcher}" ]] \
        || die "the runner launcher is missing from ${AGENT_DIR}/${launcher} after the pnpm install"
}

# Builds again the dependencies that the production install left unbuilt,
# as pnpm lists them in node_modules/.modules.yaml. Workspace packages are
# never named there, so their own lifecycle scripts do not run again.
rebuild_dependencies() {
    local names name
    local -a packages=()
    names="$(node "${checks}" pending-builds "${WORKSPACE_ROOT}/node_modules/.modules.yaml")" \
        || die "cannot read the pending builds pnpm recorded in ${WORKSPACE_ROOT}/node_modules/.modules.yaml"
    while IFS= read -r name; do
        [[ -n "${name}" ]] && packages+=("${name}")
    done <<<"${names}"
    [[ ${#packages[@]} -gt 0 ]] || return 0
    pnpm --filter-prod "${AGENT_PACKAGE}..." rebuild "${packages[@]}"
}

install_workspace_agent() {
    local app_dir=$1 install_target=$2 prod=${3:-} pnpm_version

    resolve_workspace "${app_dir}" "${install_target}" \
        || die "${install_target} is not a package of a pnpm workspace inside ${app_dir}"
    pnpm_environment
    cd "${WORKSPACE_ROOT}"
    pnpm_version="$(pnpm --version)"
    echo "Installing ${AGENT_PACKAGE} and its workspace dependencies from the pnpm workspace at ${WORKSPACE_ROOT} with pnpm ${pnpm_version}"
    node "${checks}" package-manager "${WORKSPACE_ROOT}" "${pnpm_version}" \
        || die "the pnpm workspace at ${WORKSPACE_ROOT} pins a package manager that this runner image cannot build with"

    # Not frozen: the pnpmfile changes the SDK entries, so the committed
    # lockfile is the starting point rather than the exact result. Every
    # other locked version stays as committed.
    pnpm install --no-frozen-lockfile --filter "${AGENT_PACKAGE}..." \
        --config.confirmModulesPurge=false
    # Workspace dependencies first, in dependency order, then the agent on
    # its own so a missing build script fails here as `npm run build` does.
    pnpm --filter "${AGENT_PACKAGE}^..." run --if-present build
    pnpm --filter "${AGENT_PACKAGE}" run build
    if [[ "${prod}" == --prod ]]; then
        # A clean install of the agent's production closure. The install
        # above also brought in what the workspace packages the agent uses
        # only as devDependencies need, and a prune in place would keep it;
        # --filter-prod leaves those packages out. The workspace builds wrote
        # their output beside their sources, not into node_modules.
        find "${WORKSPACE_ROOT}" -name node_modules -type d -prune -exec rm -rf {} +
        # Scripts off: pnpm would run the workspace packages' own lifecycle
        # scripts (prepare, say) again, without the devDependencies they use.
        # A package from the store keeps its build output; a local directory
        # dependency is copied again, so it is built again.
        pnpm install --prod --frozen-lockfile --ignore-scripts --filter-prod "${AGENT_PACKAGE}..." \
            --config.confirmModulesPurge=false
        rebuild_dependencies
    fi
    verify_install
}

dev() {
    local source_dir=$1 run_dir=$2 install_target=$3

    echo "Copying ${source_dir} to ${run_dir}"
    node "${mirror}" "${source_dir}" "${run_dir}"
    install_workspace_agent "${run_dir}" "${install_target}"
}

main() {
    local command=${1:-}

    case "${command}" in
        detect)
            [[ $# -eq 3 ]] || die "usage: $0 detect <app_dir> <install_target>"
            resolve_workspace "$2" "$3" || exit 1
            printf '%s\n' "${WORKSPACE_ROOT}"
            ;;
        install)
            [[ $# -eq 3 || ( $# -eq 4 && "$4" == --prod ) ]] \
                || die "usage: $0 install <app_dir> <install_target> [--prod]"
            install_workspace_agent "$2" "$3" "${4:-}"
            ;;
        build)
            [[ $# -eq 3 ]] || die "usage: $0 build <app_dir> <install_target>"
            resolve_workspace "$2" "$3" || exit 1
            # A separate process keeps set -e in force for the install, and
            # a failing pnpm step can exit 1, so any failure is reported as
            # 2 rather than read as "use npm".
            bash "${BASH_SOURCE[0]}" install "$2" "$3" --prod || exit 2
            # node_modules holds its own links to the packages, so the store
            # only adds size to the image build's layer. pnpm has already
            # built the agent, so a failed removal must not exit 1 ("use npm").
            rm -rf "${store_dir}" || true
            # Nothing the agent runs may have been a link into the store.
            verify_install
            ;;
        dev)
            [[ $# -eq 4 ]] || die "usage: $0 dev <source_dir> <run_dir> <install_target>"
            dev "$2" "$3" "$4"
            ;;
        *)
            die "usage: $0 {detect|install|build|dev} ..."
            ;;
    esac
}

main "$@"
