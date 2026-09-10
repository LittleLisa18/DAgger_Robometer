"""Read-only playback browser QA; execute and save screenshots only on hw."""

import json
import tempfile
from pathlib import Path

import requests
from playwright.sync_api import sync_playwright

from autodagger_libero.tests.test_collection import Tests


with tempfile.TemporaryDirectory() as root:
    _, store, dashboard, *_ = Tests().episode(root)
    empty = Path(root) / "empty"
    empty.mkdir()
    (empty / "metadata.json").write_text(
        json.dumps(
            {"episode_id": "empty", "steps": 0, "task": "empty test", "samples": []}
        )
    )
    dashboard.refresh()
    before = {str(p): p.read_bytes() for p in Path(root).rglob("*") if p.is_file()}
    dashboard.config.dashboard_port = 0
    dashboard.start()
    url = f"http://127.0.0.1:{dashboard.server.server_port}"
    try:
        assert requests.get(url + "/api/replay/episode0/99").status_code == 400
        assert requests.get(url + "/api/replay/unknown/0").status_code == 404
        assert (
            requests.get(url + "/api/replay/episode0/0").json()["collect"] == "rollout"
        )
        assert (
            requests.get(url + "/api/replay/episode0/2").json()["collect"] == "teacher"
        )
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
            page = browser.new_page(viewport={"width": 1440, "height": 1400})
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(url + "/player.html?episode=episode0")
            page.wait_for_function(
                "document.getElementById('position').textContent.includes('帧 1/6')"
            )
            page.wait_for_function(
                "['image','image2'].every(k=>document.getElementById(k).naturalWidth===256)"
            )
            assert page.locator("#actor").inner_text() == "rollout"
            page.locator("#takeover").click()
            page.wait_for_function(
                "document.getElementById('position').textContent.includes('step 2')"
            )
            assert page.locator("#actor").inner_text() == "teacher"
            assert "0.7" in page.locator("#action").inner_text()
            page.locator("#prev").click()
            page.wait_for_function(
                "document.getElementById('position').textContent.includes('step 1')"
            )
            assert page.locator("#actor").inner_text() == "rollout"
            page.evaluate(
                "for(const i of [5,0,4,1,3]){const s=document.getElementById('seek');s.value=i;s.dispatchEvent(new Event('input'));}"
            )
            page.wait_for_function(
                "document.getElementById('position').textContent.includes('step 3')"
            )
            page.locator("#play").click()
            page.wait_for_function(
                "document.getElementById('position').textContent.includes('帧 6/6') && document.getElementById('play').textContent==='播放'"
            )
            page.locator("#episode").select_option("empty")
            page.wait_for_function(
                "document.getElementById('position').textContent.includes('没有已保存')"
            )
            assert page.locator("#play").is_disabled()
            page.locator("#episode").select_option("episode0")
            page.wait_for_function(
                "document.getElementById('position').textContent.includes('帧 1/6')"
            )
            page.context.set_offline(True)
            page.locator("#next").click()
            page.wait_for_function(
                "document.getElementById('error').textContent.includes('加载失败')"
            )
            page.context.set_offline(False)
            page.locator("#takeover").click()
            page.wait_for_function(
                "document.getElementById('position').textContent.includes('step 2') && !document.getElementById('error').textContent"
            )
            artifacts = Path("autodagger_libero/validation")
            artifacts.mkdir(exist_ok=True)
            page.screenshot(
                path=str(artifacts / "trajectory-player.png"), full_page=True
            )
            assert not errors, errors
            browser.close()
        assert before == {
            str(p): p.read_bytes() for p in Path(root).rglob("*") if p.is_file()
        }
    finally:
        dashboard.close()
print(
    "PASS: synchronized cameras/actions, seeking, takeover, playback end, empty episode, network recovery, read-only files"
)
