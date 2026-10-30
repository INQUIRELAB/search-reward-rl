"""Render figures/framework_v2.html to source/framework.pdf (vector, like the original Chrome-printed figure) and a PNG preview.
Usage: python render_framework.py
"""
from pathlib import Path

from playwright.sync_api import sync_playwright

W = Path(__file__).resolve().parents[1]
html = (W / "figures" / "framework_v2.html").resolve()
(W / "source").mkdir(exist_ok=True)
with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome")
    pg = b.new_page(viewport={"width": 1100, "height": 1356}, device_scale_factor=2)
    pg.goto(html.as_uri())
    pg.wait_for_load_state("networkidle")
    pg.evaluate("document.fonts.ready")
    pg.screenshot(path=str(W / "figures" / "framework_v2_preview.png"), full_page=False)
    pg.pdf(path=str(W / "source" / "framework.pdf"), width="1100px", height="1356px", print_background=True,
           margin={"top": "0", "bottom": "0", "left": "0", "right": "0"})
    b.close()
print("written", W / "source" / "framework.pdf")
