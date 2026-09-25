#!/bin/sh
set -eu

warning_prefix="WARNING: vendored wheel may be incompatible with platform builds (CPython 3.11 Linux x86_64):"

is_compatible_platform_tag() {
	old_ifs=${IFS}
	IFS=.
	for tag in $1; do
		case "${tag}" in
			any | linux*_x86_64 | manylinux*_x86_64)
				IFS=${old_ifs}
				return 0
				;;
			musllinux*) ;;
		esac
	done
	IFS=${old_ifs}
	return 1
}

is_compatible_python_tag() {
	old_ifs=${IFS}
	IFS=.
	for tag in $1; do
		case "${tag}" in
			py3 | cp311)
				IFS=${old_ifs}
				return 0
				;;
			py3[0-9] | py3[0-9][0-9])
				py_minor=${tag#py3}
				if [ "${py_minor}" -le 11 ] 2>/dev/null; then
					IFS=${old_ifs}
					return 0
				fi
				;;
		esac
	done
	IFS=${old_ifs}
	return 1
}

supports_cp311_abi3() {
	old_ifs=${IFS}
	IFS=.
	for tag in $1; do
		case "${tag}" in
			cp3[0-9] | cp3[0-9][0-9])
				cp_minor=${tag#cp3}
				if [ "${cp_minor}" -le 11 ] 2>/dev/null; then
					IFS=${old_ifs}
					return 0
				fi
				;;
		esac
	done
	IFS=${old_ifs}
	return 1
}

is_compatible_abi_tag() {
	old_ifs=${IFS}
	IFS=.
	for tag in $1; do
		case "${tag}" in
			none | cp311)
				IFS=${old_ifs}
				return 0
				;;
		esac
	done
	IFS=${old_ifs}
	return 1
}

has_compatible_abi3_tag() {
	abi_tag=$1
	python_tag=$2
	old_ifs=${IFS}
	IFS=.
	for tag in ${abi_tag}; do
		if [ "${tag}" = "abi3" ] && supports_cp311_abi3 "${python_tag}"; then
			IFS=${old_ifs}
			return 0
		fi
	done
	IFS=${old_ifs}
	return 1
}

is_compatible_wheel() {
	wheel_name=${1##*/}
	wheel_stem=${wheel_name%.whl}
	case "${wheel_stem}" in
		*-*-*-*-*) ;;
		*) return 0 ;;
	esac

	platform_tag=${wheel_stem##*-}
	wheel_stem=${wheel_stem%-${platform_tag}}
	abi_tag=${wheel_stem##*-}
	wheel_stem=${wheel_stem%-${abi_tag}}
	python_tag=${wheel_stem##*-}

	is_compatible_platform_tag "${platform_tag}" || return 1
	if is_compatible_python_tag "${python_tag}" && is_compatible_abi_tag "${abi_tag}"; then
		return 0
	fi
	has_compatible_abi3_tag "${abi_tag}" "${python_tag}"
}

for vendored_wheel_dir in "$@"; do
	for vendored_wheel in "${vendored_wheel_dir}"/*.whl; do
		[ -e "${vendored_wheel}" ] || continue
		if ! is_compatible_wheel "${vendored_wheel}"; then
			echo "${warning_prefix} ${vendored_wheel}" >&2
		fi
	done
done
