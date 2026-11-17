"""Scoped filesystem workspace with fail-closed path resolution.

Every filesystem tool must go through :meth:`Workspace.resolve`. The resolver
walks the path one component at a time on an open directory descriptor and
refuses to follow a symlink at any level, so a path that looks relative cannot
escape the root and a pre-planted link cannot redirect a write.

The tool layer declares *which* roots a tool may touch via ``PathScope``; this
module enforces *how* a path inside those roots is resolved.
"""

from __future__ import annotations

import errno
import fnmatch
import os
import stat
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path, PurePosixPath

PATH_MODES = frozenset({"allow", "deny", "interrupt"})

MAX_PATH_BYTES = 1024
MAX_WRITE_BYTES = 256 * 1024


class WorkspaceError(RuntimeError):
    """A path failed resolution. Always fail-closed."""


@dataclass(frozen=True)
class PathScope:
    """Declared filesystem scope of one tool.

    ``allow``     the tool may touch paths under ``roots``.
    ``deny``      the tool is registered but every path call is refused.
    ``interrupt`` the tool never runs unattended: a call returns
                  ``approval_required`` and the caller must decide.

    ``deny_globs`` are checked against the *resolved relative* path, so they
    apply after the scope check and cannot be bypassed by ``./x/../y``.
    """

    mode: str
    roots: tuple[str, ...] = ()
    deny_globs: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.mode not in PATH_MODES:
            raise ValueError("path_scope.mode 必须是 allow / deny / interrupt")
        if self.mode == "allow" and not self.roots:
            raise ValueError("allow 模式必须至少声明一个 root")
        for root in self.roots:
            _validate_scope_root(root)
        for pattern in self.deny_globs:
            _validate_deny_glob(pattern)


def _reject(reason: str) -> "WorkspaceError":
    return WorkspaceError(reason)


def _validate_scope_root(root: str) -> tuple[str, ...]:
    if (
        not isinstance(root, str)
        or not root
        or "\x00" in root
        or root.startswith("/")
        or root.startswith("\\")
        or (":" in root.split("/")[0] and os.name == "nt")
    ):
        raise ValueError("root 必须是相对路径且不含 ..")
    raw_parts = root.split("/")
    if ".." in raw_parts:
        raise ValueError("root 必须是相对路径且不含 ..")
    return tuple(part for part in raw_parts if part not in ("", "."))


def _validate_deny_glob(pattern: str) -> tuple[str, ...]:
    if (
        not isinstance(pattern, str)
        or not pattern
        or "\x00" in pattern
        or pattern.startswith("/")
        or pattern.startswith("\\")
    ):
        raise ValueError("deny_glob 必须是有效相对模式")
    raw_parts = pattern.split("/")
    if ".." in raw_parts:
        raise ValueError("deny_glob 必须是有效相对模式")
    parts = tuple(part for part in raw_parts if part not in ("", "."))
    if not parts:
        raise ValueError("deny_glob 必须是有效相对模式")
    return parts


def _require_open_flag(name: str) -> int:
    flag = getattr(os, name, 0)
    if not flag:
        raise _reject("platform_unsupported")
    return int(flag)


def _relative_parts(relative: str) -> tuple[str, ...]:
    """Split a caller-supplied path into vetted components, or raise."""
    if not isinstance(relative, str) or not relative:
        raise _reject("path_invalid")
    if "\x00" in relative:
        raise _reject("path_invalid")
    if len(relative.encode("utf-8")) > MAX_PATH_BYTES:
        raise _reject("path_too_long")
    if relative.startswith("/") or relative.startswith("\\"):
        raise _reject("path_outside_scope")
    if ":" in relative.split("/")[0] and os.name == "nt":
        raise _reject("path_outside_scope")
    raw_parts = relative.split("/")
    parts: list[str] = []
    for part in raw_parts:
        if part in ("", "."):
            continue
        if part == "..":
            raise _reject("path_outside_scope")
        parts.append(part)
    if not parts:
        raise _reject("path_invalid")
    return tuple(parts)


class Workspace:
    """One root directory. All reads and writes are confined to it."""

    def __init__(self, root: Path) -> None:
        _require_open_flag("O_DIRECTORY")
        _require_open_flag("O_NOFOLLOW")
        root = Path(root)
        if root.is_symlink():
            raise _reject("root_symlink")
        abs_root = Path(os.path.abspath(root))
        if abs_root.is_symlink():
            raise _reject("root_symlink")
        if not abs_root.is_dir():
            raise _reject("root_not_directory")
        self._root = abs_root
        try:
            self._root_real = abs_root.resolve(strict=True)
        except OSError as exc:
            raise _reject("root_unreadable") from exc

    @property
    def root(self) -> Path:
        return self._root

    def _open_root(self) -> int:
        flags = (
            os.O_RDONLY
            | _require_open_flag("O_DIRECTORY")
            | _require_open_flag("O_NOFOLLOW")
            | getattr(os, "O_CLOEXEC", 0)
        )
        try:
            fd = os.open(self._root, flags)
        except OSError as exc:  # pragma: no cover - root re-checked in __init__
            if exc.errno == errno.ELOOP:
                raise _reject("root_symlink") from exc
            raise _reject("root_unreadable") from exc
        try:
            if not stat.S_ISDIR(os.fstat(fd).st_mode):  # pragma: no cover - O_DIRECTORY enforced
                raise _reject("root_not_directory")
        except Exception:
            os.close(fd)
            raise
        return fd

    def _stat_at(self, dir_fd: int, name: str) -> os.stat_result:
        try:
            return os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
        except FileNotFoundError as exc:
            raise _reject("path_missing") from exc
        except OSError as exc:
            raise _reject("path_unreadable") from exc

    def _resolve_with_parent(self, relative: str, *, must_exist: bool = True) -> tuple[int, str, Path]:
        """Walk relative components on dir_fd and return ``(parent_fd, last_name, resolved_path)``."""
        parts = _relative_parts(relative)
        fd = self._open_root()
        try:
            dir_flags = (
                os.O_RDONLY
                | _require_open_flag("O_DIRECTORY")
                | _require_open_flag("O_NOFOLLOW")
                | getattr(os, "O_CLOEXEC", 0)
            )
            for part in parts[:-1]:
                info = self._stat_at(fd, part)
                if stat.S_ISLNK(info.st_mode):
                    raise _reject("path_symlink")
                if not stat.S_ISDIR(info.st_mode):
                    raise _reject("path_not_directory")
                try:
                    next_fd = os.open(part, dir_flags, dir_fd=fd)
                except OSError as exc:
                    if exc.errno == errno.ELOOP:
                        raise _reject("path_symlink") from exc
                    raise _reject("path_unreadable") from exc
                os.close(fd)
                fd = next_fd

            last = parts[-1]
            last_info: os.stat_result | None
            try:
                last_info = os.stat(last, dir_fd=fd, follow_symlinks=False)
            except FileNotFoundError:
                if must_exist:
                    raise _reject("path_missing") from None
                last_info = None
            except OSError as exc:
                raise _reject("path_unreadable") from exc
            if last_info is not None:
                if stat.S_ISLNK(last_info.st_mode):
                    raise _reject("path_symlink")
                if not stat.S_ISREG(last_info.st_mode):
                    raise _reject("path_not_regular_file")

            candidate = self._root.joinpath(*parts)
            try:
                resolved = candidate.parent.resolve(strict=True).joinpath(last)
            except OSError as exc:
                raise _reject("path_unreadable") from exc
            if not resolved.is_relative_to(self._root_real):
                raise _reject("path_outside_scope")
            return fd, last, resolved
        except Exception:
            os.close(fd)
            raise

    def resolve(self, relative: str, *, must_exist: bool = True) -> Path:
        """Resolve a relative path inside the root without following symlinks."""
        fd, _last, resolved = self._resolve_with_parent(relative, must_exist=must_exist)
        os.close(fd)
        return resolved

    def relative_of(self, path: Path) -> str:
        return path.relative_to(self._root_real).as_posix()

    def read_text(self, relative: str, *, max_bytes: int = MAX_WRITE_BYTES) -> tuple[str, int]:
        dir_fd, last, _resolved = self._resolve_with_parent(relative, must_exist=True)
        try:
            flags = (
                os.O_RDONLY
                | _require_open_flag("O_NOFOLLOW")
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NONBLOCK", 0)
            )
            try:
                file_fd = os.open(last, flags, dir_fd=dir_fd)
            except FileNotFoundError as exc:
                raise _reject("path_missing") from exc
            except OSError as exc:
                if exc.errno == errno.ELOOP:
                    raise _reject("path_symlink") from exc
                raise _reject("path_unreadable") from exc
            try:
                info = os.fstat(file_fd)
                if not stat.S_ISREG(info.st_mode):
                    raise _reject("path_not_regular_file")
                if info.st_size > max_bytes:
                    raise _reject("file_too_large")
                chunks = bytearray()
                while len(chunks) <= max_bytes:
                    block = os.read(file_fd, min(64 * 1024, max_bytes + 1 - len(chunks)))
                    if not block:
                        break
                    chunks.extend(block)
                if len(chunks) > max_bytes:
                    raise _reject("file_too_large")
                data = bytes(chunks)
            except WorkspaceError:
                raise
            except OSError as exc:
                raise _reject("path_unreadable") from exc
            finally:
                os.close(file_fd)
        finally:
            os.close(dir_fd)
        try:
            return data.decode("utf-8"), len(data)
        except UnicodeDecodeError as exc:
            raise _reject("file_not_utf8") from exc

    def write_text(self, relative: str, text: str, *, overwrite: bool = False) -> int:
        if not isinstance(text, str):
            raise _reject("write_payload_invalid")
        payload = text.encode("utf-8")
        if len(payload) > MAX_WRITE_BYTES:
            raise _reject("file_too_large")
        dir_fd, last, _resolved = self._resolve_with_parent(relative, must_exist=False)
        try:
            flags = (
                os.O_WRONLY
                | os.O_CREAT
                | _require_open_flag("O_NOFOLLOW")
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NONBLOCK", 0)
                | (0 if overwrite else os.O_EXCL)
            )
            try:
                descriptor = os.open(last, flags, 0o600, dir_fd=dir_fd)
            except FileExistsError as exc:
                raise _reject("file_exists") from exc
            except OSError as exc:
                if exc.errno == errno.ELOOP:
                    raise _reject("path_symlink") from exc
                if exc.errno in (errno.ENXIO, errno.EISDIR):
                    raise _reject("path_not_regular_file") from exc
                raise _reject("write_failed") from exc
            try:
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    raise _reject("path_not_regular_file")
                if overwrite:
                    try:
                        os.ftruncate(descriptor, 0)
                    except OSError as exc:
                        raise _reject("write_failed") from exc
            except Exception:
                os.close(descriptor)
                raise
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
            except OSError as exc:
                raise _reject("write_failed") from exc
        finally:
            os.close(dir_fd)
        return len(payload)


def _glob_matches(parts: tuple[str, ...], joined: str, pattern: str) -> bool:
    if PurePosixPath(joined).match(pattern):
        return True
    raw_pattern = _validate_deny_glob(pattern)
    collapsed: list[str] = []
    for segment in raw_pattern:
        if segment == "**" and collapsed and collapsed[-1] == "**":
            continue
        collapsed.append(segment)

    @lru_cache(maxsize=None)
    def match_from(path_index: int, glob_index: int) -> bool:
        if glob_index == len(collapsed):
            return path_index == len(parts)
        segment = collapsed[glob_index]
        if segment == "**":
            if glob_index == len(collapsed) - 1:
                return path_index < len(parts)
            return match_from(path_index, glob_index + 1) or (
                path_index < len(parts) and match_from(path_index + 1, glob_index)
            )
        if path_index == len(parts):
            return False
        return fnmatch.fnmatchcase(parts[path_index], segment) and match_from(
            path_index + 1, glob_index + 1
        )

    return match_from(0, 0)


def scope_permits(scope: PathScope, relative: str) -> bool:
    """Containment check against a declared scope.

    Raises ``WorkspaceError`` for a malformed path (so a caller cannot smuggle
    an absolute path past the check), returns False for a well-formed path that
    falls outside the roots or matches a deny glob.
    """
    if scope.mode != "allow":
        return False
    parts = _relative_parts(relative)
    joined = "/".join(parts)
    for pattern in scope.deny_globs:
        if _glob_matches(parts, joined, pattern):
            return False
    for root in scope.roots:
        root_parts = _validate_scope_root(root)
        if not root_parts:
            return True
        norm_root = "/".join(root_parts)
        if joined == norm_root or joined.startswith(norm_root + "/"):
            return True
    return False
