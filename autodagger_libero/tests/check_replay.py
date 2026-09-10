"""Capture a read-only replay of the hw real-LIBERO smoke artifact."""

from pathlib import Path
from playwright.sync_api import sync_playwright
from autodagger_libero.view_run import open_run

artifacts = Path("autodagger_libero/validation")
run = sorted(artifacts.glob("simulation_*"))[-1]
dashboard = open_run(run, port=0)
try:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1440, "height": 1500})
        page.goto(f"http://127.0.0.1:{dashboard.server.server_port}")
        page.wait_for_function(
            "document.getElementById('image').naturalWidth > 0 && document.getElementById('image2').naturalWidth > 0"
        )
        page.wait_for_function(
            "document.getElementById('test-notice').textContent.includes('模拟评分器')"
        )
        page.screenshot(
            path=str(artifacts / "dashboard-real-replay.png"), full_page=True
        )
        browser.close()
finally:
    dashboard.close()
print("PASS: read-only replay of real LIBERO run, explicit simulated-score banner")
