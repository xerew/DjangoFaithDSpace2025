"""Build the printable Trust AI Lab AI/ML roadmap HTML from Markdown."""

from pathlib import Path

import markdown

from generate_beginner_guide import CSS


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "ai_ml_platform_roadmap.md"
OUTPUT = ROOT / "docs" / "ai_ml_platform_roadmap.html"


ROADMAP_CSS = CSS + r"""
.document > h1:first-of-type { margin-top: 0; }

pre {
  margin: 3mm 0 5mm;
  padding: 3.5mm 4mm;
  overflow-wrap: anywhere;
  white-space: pre-wrap;
  color: #dbeafe;
  background: #0f2347;
  border: 1px solid #1e3a8a;
  border-radius: 6px;
  break-inside: avoid-page;
}

pre code {
  padding: 0;
  color: inherit;
  background: transparent;
  font-size: 8.7pt;
}

@media print {
  h1 { break-before: page; }
  .cover-page + .page-break + h1 { break-before: auto; }
  h2, h3 { break-after: avoid-page; }
}
"""


def main() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    body = markdown.markdown(
        source,
        extensions=["extra", "sane_lists", "toc", "md_in_html"],
        extension_configs={
            "toc": {"permalink": False, "toc_depth": "1-3"},
        },
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Trust AI Lab - AI, Machine Learning &amp; Platform Improvements</title>
  <style>{ROADMAP_CSS}</style>
</head>
<body>
  <article class="document">{body}</article>
</body>
</html>
"""
    OUTPUT.write_text(html, encoding="utf-8")
    print(f"Generated {OUTPUT}")


if __name__ == "__main__":
    main()
