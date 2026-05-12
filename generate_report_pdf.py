#!/usr/bin/env python3
"""Generate a professional PDF from report.md using WeasyPrint."""

import markdown
import re
from weasyprint import HTML, CSS
from pathlib import Path

REPORT_DIR = Path(__file__).parent
MD_FILE    = REPORT_DIR / "report.md"
PDF_FILE   = REPORT_DIR / "project-report-nishant.pdf"

md_text = MD_FILE.read_text(encoding="utf-8")

# ── Extract title + metadata + abstract from the top of the markdown ─────────
lines = md_text.split("\n")

title      = lines[0].lstrip("#").strip()
meta_lines = []
abstract   = ""
body_start = 0

i = 1
while i < len(lines) and not lines[i].startswith("## "):
    meta_lines.append(lines[i])
    i += 1

# Find abstract section
abstract_match = re.search(r"## Abstract\s*\n+(.*?)(?=\n## |\Z)", md_text, re.DOTALL)
if abstract_match:
    abstract = abstract_match.group(1).strip()

# Body starts from Introduction onwards
body_match = re.search(r"(## 1\. Introduction.*)", md_text, re.DOTALL)
body_md = body_match.group(1) if body_match else ""

# ── Build metadata HTML ────────────────────────────────────────────────────
meta_html = ""
for line in meta_lines:
    line = line.strip()
    if not line or line == "---":
        continue
    # Convert **Key:** Value to styled row
    m = re.match(r"\*\*(.+?):\*\*\s*(.*)", line)
    if m:
        meta_html += f'<div class="meta-row"><span class="meta-key">{m.group(1)}:</span> <span class="meta-val">{m.group(2)}</span></div>\n'

# ── Convert abstract and body ─────────────────────────────────────────────
md_conv = markdown.Markdown(extensions=["tables", "fenced_code", "nl2br"])
abstract_html = md_conv.convert(abstract)

md_conv.reset()
body_html = md_conv.convert(body_md)

# Fix image paths to be absolute
def fix_img_paths(html):
    return re.sub(
        r'src="(results/[^"]+)"',
        lambda m: f'src="{REPORT_DIR / m.group(1)}"',
        html
    )

body_html    = fix_img_paths(body_html)
abstract_html = fix_img_paths(abstract_html)

# ── Full HTML ─────────────────────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
</head>
<body>

<div class="title-page">
  <div class="title-text">{title}</div>
  <div class="subtitle">Course Project Report</div>
  <div class="meta-block">
    {meta_html}
  </div>
</div>

<div class="abstract-box">
  <div class="abstract-label">Abstract</div>
  {abstract_html}
</div>

<div class="body-content">
{body_html}
</div>

</body>
</html>"""

# ── CSS ───────────────────────────────────────────────────────────────────
css = CSS(string="""
@page {
    size: A4;
    margin: 2.2cm 2.4cm 2.5cm 2.4cm;
    @top-left   { content: "Domain-Tuned Speculative Decoding in Code LLMs"; font-size: 8pt; color: #9ca3af; font-family: Helvetica, Arial, sans-serif; }
    @top-right  { content: "Nishant Kumar · IISc CCE · May 2026"; font-size: 8pt; color: #9ca3af; font-family: Helvetica, Arial, sans-serif; }
    @bottom-center { content: counter(page); font-size: 9pt; color: #6b7280; font-family: Helvetica, Arial, sans-serif; }
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
    font-family: Helvetica, Arial, sans-serif;
    font-size: 10.5pt;
    line-height: 1.65;
    color: #1f2937;
}

/* ── Title page ─────────────────────────────────── */
.title-page {
    text-align: left;
    padding: 30pt 0 20pt 0;
    border-bottom: 2px solid #1e3a5f;
    margin-bottom: 16pt;
}

.title-text {
    font-size: 22pt;
    font-weight: 700;
    color: #0f172a;
    line-height: 1.3;
    margin-bottom: 8pt;
}

.subtitle {
    font-size: 12pt;
    color: #6b7280;
    font-weight: 400;
    margin-bottom: 14pt;
    font-style: italic;
}

.meta-block {
    display: block;
    text-align: left;
    font-size: 10pt;
    margin: 0;
}

.meta-row {
    margin-bottom: 4pt;
    text-align: left;
}

.meta-key {
    font-weight: 700;
    color: #374151;
}

.meta-val {
    color: #4b5563;
}

/* ── Abstract box ────────────────────────────────── */
.abstract-box {
    background-color: #f8fafc;
    border: 1px solid #cbd5e1;
    border-left: 4px solid #1e3a5f;
    border-radius: 4pt;
    padding: 10pt 14pt;
    margin-bottom: 18pt;
    page-break-inside: avoid;
}

.abstract-label {
    font-size: 11pt;
    font-weight: 700;
    color: #1e3a5f;
    margin-bottom: 6pt;
}

.abstract-box p {
    font-size: 10pt;
    text-align: justify;
    line-height: 1.6;
    color: #374151;
}

/* ── Body text ───────────────────────────────────── */
.body-content {
    text-align: justify;
}

.body-content p {
    margin-bottom: 7pt;
}

/* ── Headings ────────────────────────────────────── */
h2 {
    font-size: 13pt;
    font-weight: 700;
    color: #1e3a5f;
    margin-top: 20pt;
    margin-bottom: 6pt;
    padding-bottom: 4pt;
    border-bottom: 2px solid #1e3a5f;
    page-break-after: avoid;
}

h3 {
    font-size: 11pt;
    font-weight: 700;
    color: #2563eb;
    margin-top: 12pt;
    margin-bottom: 4pt;
    page-break-after: avoid;
}

h4 {
    font-size: 10.5pt;
    font-weight: 600;
    color: #374151;
    margin-top: 8pt;
    margin-bottom: 3pt;
}

/* ── Tables ──────────────────────────────────────── */
table {
    width: 100%;
    border-collapse: collapse;
    margin: 10pt 0 12pt 0;
    font-size: 9.5pt;
    page-break-inside: avoid;
}

thead tr {
    background-color: #1e3a5f;
    color: #ffffff;
}

thead th {
    padding: 6pt 8pt;
    font-weight: 600;
    text-align: left;
}

tbody tr:nth-child(even) { background-color: #f0f4ff; }
tbody tr:nth-child(odd)  { background-color: #ffffff; }

tbody td {
    padding: 5pt 8pt;
    border-bottom: 1px solid #e5e7eb;
    vertical-align: top;
}

/* ── Blockquotes ─────────────────────────────────── */
blockquote {
    margin: 8pt 0;
    padding: 6pt 10pt;
    background-color: #f8fafc;
    border-left: 4px solid #2563eb;
    font-size: 9.5pt;
    color: #374151;
    font-style: italic;
}

blockquote p { margin: 0; }

/* ── Code ────────────────────────────────────────── */
code {
    font-family: "Courier New", monospace;
    font-size: 9pt;
    background-color: #f1f5f9;
    padding: 1pt 3pt;
    border-radius: 2pt;
    color: #0f172a;
}

pre {
    background-color: #f1f5f9;
    padding: 8pt;
    border-radius: 4pt;
    font-size: 8.5pt;
    page-break-inside: avoid;
    margin: 8pt 0;
}

pre code { background: none; padding: 0; }

/* ── Lists ───────────────────────────────────────── */
ul, ol {
    margin: 4pt 0 8pt 0;
    padding-left: 18pt;
}
li { margin-bottom: 3pt; line-height: 1.5; }

/* ── HR ──────────────────────────────────────────── */
hr {
    border: none;
    border-top: 1px solid #e5e7eb;
    margin: 12pt 0;
}

/* ── Images ──────────────────────────────────────── */
img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 12pt auto 4pt auto;
    page-break-inside: avoid;
}

em {
    display: block;
    text-align: center;
    font-size: 9pt;
    color: #6b7280;
    font-style: italic;
    margin-bottom: 10pt;
}

/* ── Strong ──────────────────────────────────────── */
strong { color: #0f172a; font-weight: 700; }
""")

print(f"Generating PDF...")
HTML(string=html, base_url=str(REPORT_DIR)).write_pdf(PDF_FILE, stylesheets=[css])
print(f"Done → {PDF_FILE}")
