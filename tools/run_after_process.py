#!/usr/bin/env python3
"""Wait for one Linux process to exit, then execute a command in its place."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import select
import sys


def process_start_ticks(pid):
    text = Path(f"/proc/{pid}/stat").read_text()
    # The command name in parentheses may contain whitespace or parentheses.
    return int(text[text.rfind(")") + 2:].split()[19])


def log(message):
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pid", type=int)
    parser.add_argument("--cwd", type=Path, required=True)
    parser.add_argument("--expect-start-ticks", type=int)
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    # Options precede the PID; the command follows a -- separator.
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if args.pid <= 0 or not command:
        parser.error("Provide a positive PID and a command after --")
    os.chdir(args.cwd)
    try:
        fd = os.pidfd_open(args.pid)
    except ProcessLookupError:
        fd = None
        log(f"PID {args.pid} has already exited")
    if fd is not None:
        try:
            try:
                start_ticks = process_start_ticks(args.pid)
            except FileNotFoundError:
                start_ticks = None
            if args.expect_start_ticks is not None and start_ticks != args.expect_start_ticks:
                log(f"Original PID {args.pid} has exited; its PID is no longer the original process")
            else:
                log(f"Waiting for PID {args.pid}; watcher PID {os.getpid()}")
                if args.ready_file:
                    args.ready_file.write_text(str(os.getpid()) + "\n")
                poller = select.poll()
                poller.register(fd, select.POLLIN)
                poller.poll()
                log(f"PID {args.pid} exited")
        finally:
            os.close(fd)
    if args.ready_file and not args.ready_file.exists():
        args.ready_file.write_text(str(os.getpid()) + "\n")
    log(f"Starting in {Path.cwd()}: {command!r}")
    # Replace the watcher, keeping its PID and redirected log output.
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
