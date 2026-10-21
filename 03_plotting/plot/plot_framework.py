#!/usr/bin/env python3
"""Export archived paper artwork; this is not reconstruction from raw media."""
from pathlib import Path
import shutil
import pymupdf


def main():
    bundle = Path(__file__).resolve().parents[2]
    source = bundle / "02_data/plot_data/paper_assets/framework.pdf"
    output = bundle / "03_plotting/output"
    output.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source,output/source.name)
    with pymupdf.open(source) as document:
        page = document[0]
        scale = min(300/72,1600/max(page.rect.width,page.rect.height))
        page.get_pixmap(matrix=pymupdf.Matrix(scale,scale),alpha=False).save(output/(source.stem+".png"))
    print("Archived artwork export:",output/source.name)


if __name__ == "__main__":
    main()
