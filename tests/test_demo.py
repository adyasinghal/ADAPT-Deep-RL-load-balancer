"""The asyncio demo: same ServerNode, real message passing, a crash in the middle."""

import asyncio

from demo.run_demo import Demo, DemoSettings, run_demo


def test_demo_survives_a_crash_without_violations():
    settings = DemoSettings(duration=12.0, speed=20.0, tty=False, crash_at=6.0, crash_node=2)
    summary = asyncio.run(run_demo(settings))
    assert summary["created"] > 100
    assert summary["violations"] == 0
    assert summary["outstanding"] == 0
    assert summary["completed"] + summary["blocked"] + summary["lost"] == summary["created"]


def test_live_view_renders():
    async def render():
        demo = Demo(DemoSettings(), asyncio.get_running_loop())
        demo.crash(3)
        return demo.render()

    text = asyncio.run(render())
    assert "CRASHED" in text and "invariant violations 0" in text
    assert len(text.splitlines()) == 3 + 8 + 5
