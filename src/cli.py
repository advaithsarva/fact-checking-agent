"""CLI: a text file in, a fact-check report out.

    python -m src.cli samples/article.txt -o report.html --json report.json
"""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

from src.agent import verify_document
from src.report import render


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=Path("report.html"))
    parser.add_argument("--json", type=Path)
    parser.add_argument("-k", type=int, default=3, help="evidence items per claim")
    args = parser.parse_args()

    report = verify_document(args.file.read_text(encoding="utf-8"), k=args.k)

    args.output.write_text(render(report, title=f"Fact-Check: {args.file.name}"), encoding="utf-8")
    print(f"credibility: {report.document_credibility}  "
          f"({len(report.claim_verdicts)} claims checked)")
    print(f"wrote {args.output}")

    if args.json:
        args.json.write_text(json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
