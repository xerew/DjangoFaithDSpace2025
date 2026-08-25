"""Build the printable Trust AI Lab beginner guide HTML from Markdown."""

from pathlib import Path

import markdown


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "beginner_user_guide.md"
OUTPUT = ROOT / "docs" / "beginner_user_guide.html"


CSS = r"""
:root {
  --navy: #0b2556;
  --blue: #1a56db;
  --blue-dark: #1e3a8a;
  --blue-pale: #eff6ff;
  --ink: #1f2937;
  --muted: #64748b;
  --line: #dbe4f0;
  --green: #15803d;
  --green-pale: #f0fdf4;
  --amber: #b45309;
  --amber-pale: #fffbeb;
  --red: #b91c1c;
  --red-pale: #fef2f2;
}

* { box-sizing: border-box; }

html {
  font-family: "Segoe UI", Arial, sans-serif;
  color: var(--ink);
  font-size: 10.6pt;
  line-height: 1.52;
}

body {
  margin: 0;
  background: #eef2f7;
}

.document {
  width: 210mm;
  margin: 0 auto;
  padding: 16mm 18mm 18mm;
  background: white;
  box-shadow: 0 8px 35px rgba(15, 23, 42, .12);
}

h1, h2, h3 {
  color: var(--navy);
  line-height: 1.2;
  break-after: avoid-page;
}

h1 {
  margin: 0 0 5mm;
  padding-bottom: 2.5mm;
  font-size: 24pt;
  border-bottom: 2px solid var(--blue);
}

h2 {
  margin: 7mm 0 2.5mm;
  font-size: 16pt;
}

h3 {
  margin: 5mm 0 2mm;
  font-size: 12.5pt;
}

p { margin: 0 0 3mm; }

strong { color: #16284b; }

a { color: var(--blue); text-decoration: none; }

ul, ol { margin: 1.5mm 0 3.5mm 6mm; padding-left: 4mm; }
li { margin: 1.1mm 0; }

table {
  width: 100%;
  border-collapse: collapse;
  margin: 3mm 0 5mm;
  font-size: 9.5pt;
  break-inside: avoid-page;
}

thead { display: table-header-group; }

th {
  padding: 2.2mm 2.5mm;
  color: white;
  background: var(--blue-dark);
  border: 1px solid var(--blue-dark);
  text-align: left;
}

td {
  padding: 2.2mm 2.5mm;
  border: 1px solid var(--line);
  vertical-align: top;
}

tbody tr:nth-child(even) td { background: #f8fafc; }

blockquote {
  margin: 4mm 0;
  padding: 3mm 4mm;
  color: #273955;
  background: var(--amber-pale);
  border: 1px solid #fde68a;
  border-left: 4px solid #f59e0b;
  border-radius: 0 6px 6px 0;
  break-inside: avoid-page;
}

blockquote p:last-child { margin-bottom: 0; }

code {
  padding: .2mm 1mm;
  border-radius: 3px;
  color: #1e3a8a;
  background: #eff6ff;
  font-family: Consolas, monospace;
  font-size: 9pt;
}

hr { margin: 7mm 0 4mm; border: 0; border-top: 1px solid var(--line); }

.cover-page {
  height: 252mm;
  margin: -16mm -18mm -18mm;
  padding: 27mm 24mm 22mm;
  color: white;
  background:
    radial-gradient(circle at 82% 20%, rgba(96, 165, 250, .55), transparent 23%),
    radial-gradient(circle at 15% 80%, rgba(45, 212, 191, .2), transparent 27%),
    linear-gradient(145deg, #071b3e 0%, #123f8f 55%, #1a56db 100%);
  position: relative;
  overflow: hidden;
  break-after: page;
}

.cover-page::after {
  content: "";
  position: absolute;
  width: 115mm;
  height: 115mm;
  right: -45mm;
  bottom: -38mm;
  border: 1px solid rgba(255,255,255,.22);
  border-radius: 50%;
  box-shadow: 0 0 0 12mm rgba(255,255,255,.035), 0 0 0 28mm rgba(255,255,255,.025);
}

.cover-mark {
  font-size: 9pt;
  font-weight: 700;
  letter-spacing: 2.2px;
  color: #bfdbfe;
}

.cover-logo {
  display: block;
  width: 34mm;
  height: 34mm;
  object-fit: contain;
  margin: 28mm 0 8mm;
  padding: 4mm;
  border-radius: 9mm;
  background: white;
  box-shadow: 0 8px 32px rgba(0,0,0,.22);
}

.cover-page h1 {
  max-width: 145mm;
  margin: 0 0 5mm;
  padding: 0;
  color: white;
  border: 0;
  font-size: 37pt;
  letter-spacing: -.6px;
}

.cover-subtitle {
  max-width: 145mm;
  margin: 0;
  color: #dbeafe;
  font-size: 18pt;
  line-height: 1.35;
}

.cover-flow {
  display: flex;
  align-items: center;
  gap: 4mm;
  margin-top: 24mm;
  font-size: 9pt;
  letter-spacing: .8px;
}

.cover-flow span {
  padding: 2.5mm 3.5mm;
  border: 1px solid rgba(255,255,255,.35);
  border-radius: 20px;
  background: rgba(255,255,255,.1);
}

.cover-flow b { color: #93c5fd; font-size: 14pt; }

.cover-meta {
  position: absolute;
  left: 24mm;
  bottom: 23mm;
  z-index: 1;
  color: #dbeafe;
  font-size: 9.5pt;
  line-height: 1.7;
}

.cover-meta strong { color: white; margin-right: 2mm; }

.toc {
  columns: 2;
  column-gap: 10mm;
  margin-top: 4mm;
  padding: 5mm 6mm;
  background: #f8fafc;
  border: 1px solid var(--line);
  border-radius: 8px;
}

.toc ul { margin: 0; padding: 0; list-style: none; }
.toc li { margin: 1.2mm 0; break-inside: avoid; }
.toc li li { padding-left: 3mm; font-size: 9pt; }
.toc a { color: #274060; }

.process-grid {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 2.2mm;
  margin: 5mm 0;
  break-inside: avoid-page;
}

.process-grid div {
  min-height: 28mm;
  padding: 3.5mm;
  color: #1e3a8a;
  background: var(--blue-pale);
  border: 1px solid #bfdbfe;
  border-top: 4px solid var(--blue);
  border-radius: 6px;
}

.process-grid b, .process-grid span { display: block; }
.process-grid b { font-size: 10pt; margin-bottom: 2mm; }
.process-grid span { color: #52657c; font-size: 8.5pt; line-height: 1.35; }

.hierarchy {
  max-width: 135mm;
  margin: 5mm auto;
  text-align: center;
  break-inside: avoid-page;
}

.h-level {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 3mm 5mm;
  color: white;
  border-radius: 6px;
}

.h-level b { color: white; letter-spacing: .7px; }
.h-level span { font-size: 9pt; }
.h-one { background: #0b2556; }
.h-two { margin: 0 8mm; background: #1e40af; }
.h-three { margin: 0 16mm; background: #2563eb; }
.h-four { margin: 0 24mm; background: #0f766e; }
.h-arrow { height: 7mm; color: #94a3b8; font-size: 15pt; }

.route-diagram {
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 3mm;
  margin: 4mm 0;
  padding: 5mm;
  background: #f8fafc;
  border: 1px solid var(--line);
  border-radius: 7px;
  break-inside: avoid-page;
}

.route-diagram span, .branch-root {
  padding: 2.5mm 4mm;
  color: #1e3a8a;
  background: white;
  border: 1.5px solid #93c5fd;
  border-radius: 6px;
  font-weight: 700;
}

.route-diagram b { color: var(--blue); font-size: 15pt; }

.branch-diagram {
  display: grid;
  grid-template-columns: 38mm 1fr;
  align-items: center;
  gap: 8mm;
  margin: 4mm 0;
  padding: 5mm;
  background: #f8fafc;
  border: 1px solid var(--line);
  border-radius: 7px;
  break-inside: avoid-page;
}

.branch-root { text-align: center; }
.branch-lines div { margin: 2mm 0; padding: 2mm 3mm; border-left: 3px solid var(--blue); background: white; }
.branch-lines b { display: inline-block; min-width: 20mm; }

.task-list-item { list-style: none; margin-left: -5mm; }
.task-list-item input { margin-right: 2mm; accent-color: var(--blue); }

.page-break { break-before: page; height: 0; }

@page {
  size: A4;
  margin: 14mm 15mm 16mm;
}

@media print {
  html { font-size: 9.7pt; }
  body { background: white; }
  .document { width: auto; margin: 0; padding: 0; box-shadow: none; }
  .cover-page {
    height: 268mm;
    margin: -14mm -15mm -16mm;
    padding: 27mm 24mm 22mm;
  }
  h1 { font-size: 22pt; }
  h2 { font-size: 15pt; }
  h3 { font-size: 11.5pt; }
  a { color: inherit; }
  table, blockquote, .process-grid, .hierarchy, .route-diagram, .branch-diagram { break-inside: avoid; }
  p, li { orphans: 3; widows: 3; }
}
"""


def main() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    body = markdown.markdown(
        source,
        extensions=[
            "extra",
            "sane_lists",
            "toc",
            "md_in_html",
        ],
        extension_configs={
            "toc": {"permalink": False, "toc_depth": "1-3"},
        },
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Trust AI Lab — Beginner User Guide</title>
  <style>{CSS}</style>
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
