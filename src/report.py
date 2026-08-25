"""Renders a VerificationReport to one self-contained HTML page."""
import html
from src.agent import VerificationReport

_VERDICT_COLOR = {"Supported": "#d4edda", "Contradicted": "#f8d7da", "Unverifiable": "#fff3cd"}
_STANCE_COLOR = {"supports": "#4caf50", "contradicts": "#f44336", "neutral": "#9e9e9e"}

_STYLE = """
body { font-family: -apple-system, Segoe UI, sans-serif; max-width: 800px; margin: 2rem auto; padding: 0 1rem; line-height: 1.5; }
.claim { border: 1px solid #ddd; border-radius: 6px; padding: 1rem; margin-bottom: 1rem; }
.badge { display: inline-block; padding: 2px 10px; border-radius: 10px; font-size: 0.85rem; font-weight: 600; }
.evidence { margin: 0.5rem 0 0 1rem; padding: 0.5rem; border-left: 3px solid #ccc; font-size: 0.9rem; }
.stance { font-weight: 600; }
.mode { color: #888; font-size: 0.8rem; }
"""


def _claim_html(v) -> str:
    color = _VERDICT_COLOR.get(v.verdict, "#eee")
    evidence_html = "".join(
        f'<div class="evidence"><span class="stance" style="color:{_STANCE_COLOR.get(e.stance, "#000")}">'
        f'{html.escape(e.stance)}</span> (confidence {e.stance_confidence}) &mdash; '
        f'<em>{html.escape(e.source)}</em> [{html.escape(e.reliability_tier)}, reliability {e.reliability_score}]'
        f'<br>{html.escape(e.snippet[:300])}</div>'
        for e in v.evidence
    ) or '<p><em>No evidence found above the relevance threshold.</em></p>'

    return (
        f'<div class="claim"><p>{html.escape(v.claim)}</p>'
        f'<span class="badge" style="background:{color}">{html.escape(v.verdict)}</span> '
        f'<span class="mode">confidence {v.confidence}, {html.escape(v.search_mode)} search</span>'
        f'{evidence_html}</div>'
    )


def render(report: VerificationReport, title: str = "Fact-Check Report") -> str:
    claims_html = "".join(_claim_html(v) for v in report.claim_verdicts) or "<p>No checkable claims found.</p>"
    return (
        f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
        f"<style>{_STYLE}</style></head><body>"
        f"<h1>{html.escape(title)}</h1>"
        f"<p><strong>Document credibility score:</strong> {report.document_credibility} "
        f"({len(report.claim_verdicts)} claim(s) checked)</p>"
        f"{claims_html}</body></html>"
    )
