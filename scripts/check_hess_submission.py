"""Check HESS submission files and optionally package them (no upload).

Usage: python scripts/check_hess_submission.py --bundle
Requires pypdf and Pillow from the existing environment. Journal rules were
checked on 9 October 2026; see paper/HESS_SUBMISSION_CHECKLIST.md.
"""

import argparse
import hashlib
import math
from pathlib import Path
import re
import shutil
import zipfile

from PIL import Image
import numpy as np
from pypdf import PdfReader
from pypdf.generic import ContentStream

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
OUT = PAPER / "submission"



def bib_dois_cited(source):
    """DOIs of the bibliography entries cited in the availability statement."""
    bib = (PAPER / "references.bib").read_text(encoding="utf-8")
    block = source.split(r"\codedataavailability", 1)[1].split(r"\authorcontribution", 1)[0]
    keys = {k.strip() for g in re.findall(r"\\cite\w*\{([^}]+)\}", block) for k in g.split(",")}
    entries = {re.match(r"\w+\{([^,]+)", e).group(1): e for e in re.split(r"(?m)^@", bib)[1:]}
    return " ".join(entries.get(k, "") for k in keys)

def font_info(resources, seen=None):
    """Inspect embedded fonts, including fonts inside nested PDF forms."""
    if seen is None:
        seen = set()
    resources = resources.get_object()
    if id(resources) in seen:
        return []
    seen.add(id(resources))
    result = []
    for ref in resources.get("/Font", {}).values():
        font = ref.get_object()
        if "/DescendantFonts" in font:
            font = font["/DescendantFonts"][0].get_object()
        descriptor = font.get("/FontDescriptor", {}).get_object() if "/FontDescriptor" in font else {}
        embedded = any(k in descriptor for k in ("/FontFile", "/FontFile2", "/FontFile3"))
        result.append((str(font.get("/BaseFont", "unknown")), embedded))
    for ref in resources.get("/XObject", {}).values():
        obj = ref.get_object()
        if obj.get("/Subtype") == "/Form":
            result.extend(font_info(obj.get("/Resources", resources), seen))
    return result


def multiply(a, b):
    """Compose PDF affine matrices, with row-vector coordinates."""
    return (a[0]*b[0]+a[1]*b[2], a[0]*b[1]+a[1]*b[3],
            a[2]*b[0]+a[3]*b[2], a[2]*b[1]+a[3]*b[3],
            a[4]*b[0]+a[5]*b[2]+b[4], a[4]*b[1]+a[5]*b[3]+b[5])


def image_dpi(stream, resources, reader, matrix=(1, 0, 0, 1, 0, 0)):
    """Measure raster resolution from each image's actual PDF placement."""
    result, stack = [], []
    for args, op in ContentStream(stream, reader).operations:
        if op == b"q":
            stack.append(matrix)
        elif op == b"Q":
            matrix = stack.pop()
        elif op == b"cm":
            matrix = multiply(tuple(float(x) for x in args), matrix)
        elif op == b"Do":
            obj = resources["/XObject"][args[0]].get_object()
            if obj.get("/Subtype") == "/Image":
                sx, sy = math.hypot(*matrix[:2]), math.hypot(*matrix[2:4])
                result.append((72*int(obj["/Width"])/sx, 72*int(obj["/Height"])/sy))
            elif obj.get("/Subtype") == "/Form":
                transform = tuple(float(x) for x in obj.get("/Matrix", (1, 0, 0, 1, 0, 0)))
                result.extend(image_dpi(obj, obj.get("/Resources", resources), reader,
                                        multiply(transform, matrix)))
    return result


def plain_tex(text):
    """Convert this manuscript's text abstract for the submission form."""
    text = re.sub(r"(?m)^%.*$", "", text)
    text = text.replace(r"\%", "%").replace(r"\,", " ").replace("--", "–")
    text = text.replace("~", " ")
    if "\\" in text:
        raise ValueError("Abstract contains unhandled LaTeX; review its plain-text export")
    return " ".join(text.split())


def colour_previews(figure_names):
    """Create inspection copies only; never alter submission figure files.

    Machado et al. (2009), doi:10.1109/TVCG.2009.113, severity 100 matrices.
    Coefficients verified against the upstream colorspacious data table:
    https://github.com/njsmith/colorspacious/blob/master/colorspacious/cvd.py
    Apply to linear RGB, then encode sRGB for display.
    """
    from PIL import ImageDraw

    matrices = {
        "Protanopia": [[.152286, 1.052583, -.204868], [.114503, .786281, .099216], [-.003882, -.048116, 1.051998]],
        "Deuteranopia": [[.367322, .860646, -.227968], [.280085, .672501, .047413], [-.011820, .042940, .968881]],
        "Tritanopia": [[1.255528, -.076749, -.178779], [-.078411, .930809, .147602], [.004733, .691367, .303900]],
    }
    directory = OUT / "accessibility"
    directory.mkdir(exist_ok=True)
    for name in figure_names:
        with Image.open((ROOT / "figures" / name).with_suffix(".png")) as image:
            small = image.convert("RGB")
            small = small.resize((600, round(image.height*600/image.width)), Image.Resampling.LANCZOS)
        srgb = np.asarray(small, dtype=float)/255
        linear = np.where(srgb <= .04045, srgb/12.92, ((srgb+.055)/1.055)**2.4)
        sheet = Image.new("RGB", (4*600, small.height+30), "white")
        draw = ImageDraw.Draw(sheet)
        sheet.paste(small, (0, 30))
        draw.text((8, 8), "Original", fill="black")
        for column, (label, matrix) in enumerate(matrices.items(), start=1):
            transformed = np.clip(linear @ np.asarray(matrix).T, 0, 1)
            encoded = np.where(transformed <= .0031308, 12.92*transformed,
                               1.055*transformed**(1/2.4)-.055)
            preview = Image.fromarray(np.rint(encoded*255).astype("uint8"))
            sheet.paste(preview, (column*600, 30))
            draw.text((column*600+8, 8), label, fill="black")
        sheet.save(directory / f"{Path(name).stem}_cvd.png")
    (directory / "README.md").write_text(
        "# Colour-vision inspection copies\n\n"
        "Each sheet shows the original, protanopia, deuteranopia and tritanopia simulations. "
        "Machado et al. (2009), severity 100, applied in linear RGB. These are inspection "
        "copies, not submission figures. Check model markers/line styles, map colour-bar "
        "signs, basin categories and significance hatching. A simulation is an approximation, "
        "not a guarantee of every reader's perception.\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", action="store_true", help="Write local submission ZIPs after checks pass")
    args = parser.parse_args()
    checks, warnings, rows = [], [], []

    def check(ok, description):
        checks.append((bool(ok), description))

    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    source = re.sub(r"(?m)(?<!\\)%.*$", "", tex)
    check(r"\documentclass[hess, manuscript]{copernicus}" in source,
          "HESS manuscript class: one column, default line and page numbers")
    check(not re.search(r"\\(?:newcommand|renewcommand|patchcmd|paragraph|subfloat|href)\b", source),
          "No custom command definitions, class patches, paragraph headings or subfloats")
    labels = re.findall(r"\\label\{([^}]+)\}", source)
    refs = re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", source)
    check(len(labels) == len(set(labels)) and all(not re.search(r"\s", x) for x in labels)
          and set(refs) <= set(labels), "Unique labels without spaces; all internal references defined")
    order = [source.find(m) for m in (r"\noappendix", r"\codedataavailability", r"\authorcontribution",
                                      r"\competinginterests", r"\begin{acknowledgements}", r"\financialsupport")]
    check(min(order) >= 0 and order == sorted(order), "End-matter order follows HESS submission instructions")
    availability = source.split(r"\codedataavailability", 1)[1].split(r"\authorcontribution", 1)[0]
    # A deposit counts only once a Zenodo DOI is cited, not when one is promised.
    check("being prepared" not in availability and bool(re.search(r"10\.5281/zenodo\.\d+", availability + bib_dois_cited(source))),
          "Availability statement cites the deposited archive by DOI")
    for name in ("copernicus.cls", "copernicus.cfg", "copernicus.bst"):
        # Normalize checkout line endings, without tolerating content changes.
        check((PAPER / name).read_text(encoding="utf-8", errors="replace")
              == (PAPER / "template" / name).read_text(encoding="utf-8", errors="replace"),
              f"{name} matches the official downloaded package")

    summary_block = tex.split("%% HESS SHORT SUMMARY", 1)[1].split(r"\begin{abstract}", 1)[0]
    summary = " ".join(line[3:].strip() for line in summary_block.splitlines()
                       if line.startswith("%% ") and not line.startswith("%% --"))
    check(0 < len(summary) <= 500, f"Non-technical summary: {len(summary)} characters including spaces")
    check(not re.search(r"\b[A-Z][A-Za-z]*[A-Z]\w*", summary), "Short summary contains no uppercase abbreviations")
    abstract = plain_tex(source.split(r"\begin{abstract}", 1)[1].split(r"\end{abstract}", 1)[0])
    check(not re.search(r"\\cite\w*", source.split(r"\begin{abstract}", 1)[1].split(r"\end{abstract}", 1)[0]),
          "Abstract contains no reference citations")
    bib = (PAPER / "references.bib").read_text(encoding="utf-8")
    keys = set(re.findall(r"(?m)^@\w+\{([^,]+)", bib))
    citations = {k.strip() for group in re.findall(r"\\cite\w*(?:\[[^]]*\])*\{([^}]+)\}", source)
                 for k in group.split(",")}
    check(citations <= keys, f"All {len(citations)} cited bibliography keys exist")
    for entry in re.split(r"(?m)^@", bib)[1:]:
        kind, key = re.match(r"(\w+)\{([^,]+)", entry).groups()
        if key in citations and kind == "article":
            check(bool(re.search(r"\bpages\s*=", entry)), f"Reference {key} has pages or an article identifier")

    figure_names = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", source)
    inputs = [PAPER / name for name in ("main.tex", "references.bib", "copernicus.cls", "copernicus.cfg", "copernicus.bst")]
    inputs.extend(ROOT / "figures" / name for name in figure_names)
    check((PAPER / "main.pdf").stat().st_mtime >= max(path.stat().st_mtime for path in inputs),
          "Manuscript PDF is newer than its text, bibliography, template and figure inputs")
    check([re.match(r"fig(\d+)", name).group(1) for name in figure_names]
          == [f"{n:02}" for n in range(1, 8)], "Seven figures numbered in source order; each panel set in one file")
    total = (PAPER / "main.pdf").stat().st_size
    for name in figure_names:
        path = ROOT / "figures" / name
        pdf = PdfReader(path)
        check(len(pdf.pages) == 1, f"{name}: one-page composite figure")
        page = pdf.pages[0]
        width = float(page.mediabox.width)*2.54/72
        check(width >= 8, f"{name}: width {width:.1f} cm (minimum 8 cm)")
        check(path.stat().st_size <= 2_000_000, f"{name}: PDF below 2 MB")
        check(all(embedded for _, embedded in font_info(page["/Resources"])), f"{name}: all fonts embedded")
        fonts = sorted({f.split("+")[-1] for f, _ in font_info(page["/Resources"])})
        families = {re.split(r"[-,]", f.lstrip("/"))[0].removesuffix("MT") for f in fonts}
        check(len(families) == 1, f"{name}: one font family ({', '.join(sorted(families))})")
        dpi = image_dpi(page.get_contents(), page["/Resources"], pdf)
        check(all(min(pair) >= 299.5 for pair in dpi), f"{name}: embedded raster layers at least 300 dpi (rounding tolerance 0.5)")
        image = Image.open(path.with_suffix(".png"))
        png_dpi = image.info.get("dpi", (0, 0))
        check(min(png_dpi) >= 299.5 and image.width/(width/2.54) >= 299.5,
              f"{name}: PNG export at 300 dpi, consistent physical width")
        check(path.with_suffix(".png").stat().st_size <= 5_000_000, f"{name}: PNG below 5 MB")
        check("/OCProperties" not in pdf.trailer["/Root"], f"{name}: no optional hidden layers")
        text = page.extract_text()
        check(not re.search(r"\d%", text), f"{name}: space between numbers and %")
        rows.append(f"| {name} | {width:.1f} | {path.stat().st_size/1e6:.3f} | {image.width} × {image.height} | {'; '.join(fonts)} |")
        total += path.stat().st_size

    pdf = PdfReader(PAPER / "main.pdf")
    sizes = {(float(p.mediabox.width), float(p.mediabox.height), p.get("/Rotate", 0)) for p in pdf.pages}
    check(len(sizes) == 1 and all(w < h and rotation == 0 for w, h, rotation in sizes),
          f"Manuscript: {len(pdf.pages)} pages with consistent portrait dimensions")
    check(all(embedded for page in pdf.pages for _, embedded in font_info(page["/Resources"])),
          "Manuscript: all fonts embedded, including figures")
    check((PAPER / "main.pdf").stat().st_size <= 50_000_000, "Review PDF below 50 MB")
    check(total <= 30_000_000, f"Manuscript plus separate vector figures: {total/1e6:.2f} MB (below 30 MB)")
    log = (PAPER / "main.log").read_text(errors="replace")
    check(not re.search(r"undefined references|Citation .* undefined|Reference .* undefined|multiply defined", log),
          "LaTeX log: no undefined or duplicate references")
    overfull = re.findall(r"Overfull \\hbox \(([^)]+)\).*?at lines ([\d-]+)", log)
    if overfull:
        warnings.append("LaTeX overfull boxes need visual review: " + "; ".join(f"{size}, lines {lines}" for size, lines in overfull))

    OUT.mkdir(exist_ok=True)
    report = ["# HESS technical checks", "", "Generated by scripts/check_hess_submission.py. Passing technical checks does not mean the paper is ready to submit; see ../HESS_SUBMISSION_CHECKLIST.md.", "",
              "| Figure | Width (cm) | PDF (MB) | PNG pixels (300 dpi) | Embedded fonts |",
              "|---|---:|---:|---|---|", *rows, "", "## Automated checks", ""]
    report.extend(f"- {'PASS' if ok else 'FAIL'}: {description}" for ok, description in checks)
    report.extend(["", "## Items requiring review", ""] + [f"- {w}" for w in warnings])
    (OUT / "technical_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    # The two paste-in texts track the manuscript on every run; the packet waits for a clean pass.
    (OUT / "abstract.txt").write_text(abstract + "\n", encoding="utf-8")
    (OUT / "short_summary.txt").write_text(summary + "\n", encoding="utf-8")
    failed = [description for ok, description in checks if not ok]
    print(f"{len(checks)-len(failed)}/{len(checks)} technical checks passed; {len(warnings)} review items.")
    for description in failed:
        print("FAIL:", description)
    if failed:
        raise SystemExit(1)
    if args.bundle:
        colour_previews(figure_names)
        shutil.copy2(PAPER / "main.pdf", OUT / "manuscript.pdf")
        with zipfile.ZipFile(OUT / "figures.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for name in figure_names:
                archive.write(ROOT / "figures" / name, name)
        with zipfile.ZipFile(OUT / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            # Make the source archive compile independently after unpacking.
            archive.writestr("main.tex", tex.replace(r"\graphicspath{{../figures/}}", r"\graphicspath{{./}}"))
            for name in ("references.bib", "main.bbl", "copernicus.cls", "copernicus.cfg", "copernicus.bst"):
                archive.write(PAPER / name, name)
            for name in figure_names:
                archive.write(ROOT / "figures" / name, name)
        manifest = ["file,sha256"]
        for path in sorted(OUT.rglob("*")):
            if path.is_file() and path.name != "SHA256.csv":
                manifest.append(f"{path.relative_to(OUT).as_posix()},{hashlib.sha256(path.read_bytes()).hexdigest()}")
        (OUT / "SHA256.csv").write_text("\n".join(manifest) + "\n", encoding="utf-8")
        print("Local draft submission packet written to paper/submission/ (not uploaded).")


if __name__ == "__main__":
    main()
