"""Headless Chromium dashboard QA, all execution and screenshots on hw."""

import json
import tempfile
from pathlib import Path
from playwright.sync_api import sync_playwright
from autodagger_libero.tests.test_collection import Tests

artifacts = Path("autodagger_libero/validation")
artifacts.mkdir(exist_ok=True)
with tempfile.TemporaryDirectory() as root:
    _, _, dashboard, meta, *_ = Tests().episode(root)
    dashboard.config.dashboard_port = 0
    dashboard.start()
    url = f"http://127.0.0.1:{dashboard.server.server_port}"
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1440, "height": 1500})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(url)
        page.wait_for_function(
            "document.getElementById('image').naturalWidth > 0 && document.getElementById('image2').naturalWidth > 0"
        )
        page.wait_for_function(
            "document.getElementById('actor').textContent === 'teacher'"
        )
        page.get_by_role("button", name="查看", exact=True).click()
        page.wait_for_function(
            "document.getElementById('detail-title').textContent.includes('episode0')"
        )
        page.wait_for_function(
            "document.getElementById('coverage').textContent.includes('历史')"
        )
        page.wait_for_function(
            "document.getElementById('takeover-reason').textContent.includes('历史') && document.getElementById('takeover-reason').textContent.includes('强制接管（测试） · step 2')"
        )
        page.screenshot(path=str(artifacts / "dashboard-history.png"), full_page=True)
        page.get_by_role("button", name="返回实时曲线").click()
        page.wait_for_function(
            "document.getElementById('coverage').textContent.includes('实时')"
        )
        for reason, label in [
            ("progress_stalled", "进度停滞"),
            ("progress_regression", "进度持续退步"),
        ]:
            dashboard.publish(takeover_reason=reason)
            page.wait_for_function(
                "label => document.getElementById('takeover-reason').textContent.includes('实时接管原因：'+label)",
                arg=label,
            )
        dashboard.publish(takeover_step=None, takeover_reason=None)
        page.wait_for_function(
            "document.getElementById('takeover-reason').textContent.includes('未接管')"
        )
        dashboard.publish(takeover_step=2, takeover_reason="forced_test")
        page.context.set_offline(True)
        page.wait_for_function(
            "document.getElementById('connection').textContent.includes('连接中断')"
        )
        page.context.set_offline(False)
        page.wait_for_function(
            "document.getElementById('connection').textContent.includes('已连接')"
        )
        assert not errors, errors
        assert page.locator("canvas").count() == 2
        assert page.get_by_role("button").count() == 2
        page.screenshot(path=str(artifacts / "dashboard-live.png"), full_page=True)
        browser.close()
    dashboard.close()
print(
    json.dumps(
        {
            "result": "PASS",
            "checks": [
                "two cameras",
                "two charts",
                "teacher badge",
                "history metadata",
                "live/history takeover reasons and reset",
                "history/live switching",
                "offline recovery",
                "no JS errors",
                "read-only UI",
            ],
            "screenshots": str(artifacts),
        }
    )
)
