"""Memory context load + distillation cycle for the MR1 root orchestrator.

`_load_memory_context` reads the on-disk memory file used to seed the
system prompt. `trigger_memdltr` runs the dump-and-distill cycle that
compresses long-running conversation state, then restarts the brain.

Both were originally methods on `MR1`; extracted as free functions
taking `root` (the MR1 instance) as first parameter. `MR1` keeps thin
wrappers so existing callers and tests that patch `MR1.start` continue
to work.
"""

from mr1.orchestrator.root import _CONTEXT_PATH, _DUMP_COMPLETE_SIGNAL


def load_memory_context(root) -> str:
    """Read the memory context file if it exists."""
    if _CONTEXT_PATH.exists():
        try:
            return _CONTEXT_PATH.read_text(encoding="utf-8")
        except OSError:
            pass
    return ""


def trigger_memdltr(root) -> str:
    """
    Trigger the full compression + restart cycle:
      1. Ask MR1 to dump context to mr1_context.md
      2. Wait for the dump completion signal
      3. Run mem_dltr.distill()
      4. Kill MR1 process
      5. Spawn a fresh MR1 process
    """
    response = root._send_to_brain(
        "[SYSTEM:MEMDLTR] Dump everything important about this conversation "
        "to memory/active/mr1_context.md. Include: full conversation summary, "
        "active tasks, user preferences, key decisions. After writing the file, "
        "end your response with exactly: [MR1:DUMP_COMPLETE]"
    )

    dump_confirmed = _DUMP_COMPLETE_SIGNAL in response
    root._emit_event(
        "system_event",
        lane="system",
        summary="memory distillation started",
        agent_type="mr1",
    )

    # Run distillation.
    from mr1.mini.mem_dltr import distill
    dltr_result = distill(logger=root._logger)

    # Kill and restart.
    root._process.kill()
    root._state.set_claude_session_id(None)
    root.start()

    status = "confirmed" if dump_confirmed else "unconfirmed"
    message = (
        f"Memory compressed (dump {status}). "
        f"Distilled: {dltr_result.forgotten} forgotten, "
        f"{dltr_result.dumped} dumped, {dltr_result.rag_chunks} RAG chunks. "
        f"MR1 restarted with fresh context."
    )
    root._emit_event(
        "system_event",
        lane="system",
        summary=message,
        agent_type="mem_dltr",
    )
    return message
