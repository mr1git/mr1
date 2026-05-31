"""Confirm: 'rename X' via NL does not actually rename anything."""
import tempfile
from pathlib import Path
from mr1.runtime_test_cli import RuntimePaths, RuntimeTestSession, _patched_runtime_paths

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    paths = RuntimePaths(
        isolated=True,
        runtime_root=root,
        workflow_root=root / "workflows",
        state_path=root / "active" / "mr1_state.json",
        context_path=root / "active" / "mr1_context.md",
        dumps_root=root / "dumps",
        rag_root=root / "rag",
    )
    with _patched_runtime_paths(paths):
        s = RuntimeTestSession(paths)
        try:
            for i, turn in enumerate(
                [
                    "/agent create Librarian",
                    "/agents",
                    "rename Librarian to PaperLibrarian",
                    "/agents",
                ],
                1,
            ):
                p = s.handle_input(turn, request_index=i)
                print(f"=== turn {i}: {turn!r} ===")
                print(f"  ok={p['ok']}")
                print(f"  resp={p['response_text'][:400]!r}")
                if p["agents"]["updated"]:
                    print(
                        f"  agents_updated: "
                        f"{[(a['agent_id'], a.get('title')) for a in p['agents']['updated']]}"
                    )
                print()
        finally:
            s.shutdown()
