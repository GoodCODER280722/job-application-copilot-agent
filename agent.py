from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime, UTC
from pathlib import Path
from typing import List, Dict, Any, Tuple

VERSION = "v0.3.1 - Stable Scoring Patch"

# -----------------------------
# Defaults / Config
# -----------------------------
DEFAULT_RESUME_PATH = Path("resume.txt")
DEFAULT_JOB_PATH = Path("job_posting.txt")
DEFAULT_OUT_BASE = Path("output")

# Simple keyword banks (expand over time)
TECH_KEYWORDS = [
    "python", "sql", "api", "rest", "fastapi", "flask", "docker", "git",
    "aws", "azure", "gcp", "linux", "windows", "powershell",
    "servicenow", "jira", "incident", "sla", "intune", "mdm",
    "troubleshooting", "automation", "llm", "prompt", "validation"
]

RED_FLAG_PHRASES = [
    "unpaid", "pay to apply", "gift card", "telegram", "whatsapp",
    "kindly", "wire transfer", "check deposit", "remote equipment fee"
]


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class MatchScores:
    keyword_coverage: float
    overlap_keywords: List[str]
    missing_keywords: List[str]
    red_flags: List[str]
    confidence: float


@dataclass
class SuggestedEdit:
    target: str  # "summary" or "bullet"
    original: str
    suggestion: str
    rationale: str
    needs_confirmation: bool = False


@dataclass
class OutputPacket:
    role_summary: str
    match_scores: MatchScores
    suggested_edits: List[SuggestedEdit]
    interview_questions: List[str]
    talking_points: List[str]


# -----------------------------
# Helpers
# -----------------------------
TECH_SIGNAL_WORDS = {
    "windows", "linux", "mac", "microsoft", "azure", "aws", "gcp",
    "active", "directory", "vpn", "mfa", "intune", "servicenow",
    "jira", "ticket", "incident", "sla", "endpoint", "desktop",
    "hardware", "software", "network", "printer", "outlook",
    "exchange", "o365", "office", "sql", "python", "api",
    "powershell", "security", "firewall", "router", "switch",
    "server", "deployment", "imaging", "autopilot", "sccm"
}


STOPWORDS = {
    "the", "and", "or", "a", "an", "to", "of", "in", "for", "with", "on", "at", "by", "from", "as", "is", "are", "be",
    "this", "that", "these", "those", "you", "your", "we", "our", "will", "can", "may", "must", "should", "have",
    "has", "had", "their", "they", "them", "it", "its", "not", "but", "if", "than", "then", "about", "into", "over",
    "within", "across", "per", "including", "etc", "join", "ability", "perform", "routine", "handle", "maintain",
    "collaborate", "improvement", "knowledge"
}


def read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path.resolve()}")
    return path.read_text(encoding="utf-8", errors="replace").strip()


def read_text_optional(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace").strip()


def normalize(text: str) -> str:
    text = "" if text is None else str(text)
    return re.sub(r"\s+", " ", text).strip().lower()


def collect_pasted_input(prompt: str) -> str:
    """
    Paste-first UX: user pastes content; we end on TWO consecutive blank lines.
    """
    print(prompt)
    print("(Press Enter twice to submit)\n")

    lines: List[str] = []
    empty_count = 0

    while True:
        line = input()
        if line.strip() == "":
            empty_count += 1
            if empty_count >= 2:
                break
        else:
            empty_count = 0
            lines.append(line)

    return "\n".join(lines).strip()


def make_run_dir(out_base: Path) -> Path:
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = out_base / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def extract_keywords(text: str, keyword_list: List[str]) -> List[str]:
    t = normalize(text)
    found = []
    for kw in keyword_list:
        if re.search(rf"\b{re.escape(kw.lower())}\b", t):
            found.append(kw)
    return sorted(set(found), key=str.lower)


def find_red_flags(job_text: str) -> List[str]:
    t = normalize(job_text)
    return [p for p in RED_FLAG_PHRASES if p in t]


def split_bullets(resume_text: str) -> List[str]:
    lines = [ln.strip() for ln in resume_text.splitlines()]
    bullets = []
    for ln in lines:
        if re.match(r"^(\-|\*|•)\s+", ln):
            bullets.append(re.sub(r"^(\-|\*|•)\s+", "", ln).strip())
    if not bullets:
        sentences = re.split(r"(?<=[.!?])\s+", resume_text.strip())
        bullets = [s.strip() for s in sentences if len(s.strip()) > 40][:8]
    return bullets[:20]


def safe_rewrite_bullet(bullet: str, target_keywords: List[str]) -> SuggestedEdit | None:
    b = bullet.strip()
    if not b or len(b) < 20:
        return None

    tightened = re.sub(r"\bvery\b|\breally\b|\bjust\b", "", b, flags=re.IGNORECASE).strip()
    tightened = re.sub(r"\s{2,}", " ", tightened)

    add_kw = None
    for kw in target_keywords:
        if kw.lower() in normalize(b):
            continue
        if kw.lower() in {"troubleshooting", "automation", "incident", "sla", "api", "sql", "python", "git"}:
            add_kw = kw
            break

    suggestion = tightened
    needs_confirmation = False
    rationale = "Tightened wording for clarity."

    if add_kw:
        if add_kw.lower() in {"python", "sql", "api", "git"}:
            suggestion = f"{suggestion} (Tools used where applicable: {add_kw.upper() if add_kw=='sql' else add_kw}.)"
            needs_confirmation = True
            rationale += f" Added a tool mention to align with the posting—confirm you actually used {add_kw}."
        else:
            suggestion = f"{suggestion} (Focus: {add_kw}.)"
            rationale += f" Added focus keyword '{add_kw}' to better match the posting."

    if suggestion == b:
        return None

    return SuggestedEdit(
        target="bullet",
        original=b,
        suggestion=suggestion,
        rationale=rationale,
        needs_confirmation=needs_confirmation
    )


def build_interview_pack(job_text: str, overlap: List[str], missing: List[str]) -> Tuple[List[str], List[str]]:
    questions = [
        "Walk me through your background and how it fits this role.",
        "Tell me about a tough troubleshooting issue you solved—what was your process?",
        "How do you prioritize multiple incoming requests or tickets?",
        "Describe a time you improved a process or automated something.",
        "How do you communicate with non-technical stakeholders during an incident?",
        "What’s your approach to documenting work and building repeatable playbooks?",
        "Tell me about a time you made a mistake—how did you handle it?",
        "How do you ensure accuracy and avoid assumptions when diagnosing problems?",
        "What tools or systems have you used to manage work (ticketing, docs, version control)?",
        "What would your first 30 days look like in this role?"
    ]
    talking_points: List[str] = []
    if overlap:
        talking_points.append(f"Emphasize matching keywords you already have: {', '.join(overlap[:10])}.")
    if missing:
        talking_points.append(f"Prepare honest answers for missing areas: {', '.join(missing[:10])} (frame as 'learning plan').")
    talking_points.append("Use STAR format for behavioral answers (Situation, Task, Action, Result).")
    talking_points.append("Keep claims grounded—no tool name-dropping unless you’ve used it.")
    return questions, talking_points


def tokenize_words(text: str) -> List[str]:
    t = normalize(text)
    words = re.findall(r"[a-z0-9\+\#\.]+", t)
    return [w for w in words if w and w not in STOPWORDS and len(w) > 2]


def rerank_phrases_diverse(phrases: List[str], max_per_token: int = 3) -> List[str]:
    counts: Dict[str, int] = {}
    out: List[str] = []
    for p in phrases:
        tokens = p.split()
        anchor = max(tokens, key=len) if tokens else p
        counts[anchor] = counts.get(anchor, 0) + 1
        if counts[anchor] <= max_per_token:
            out.append(p)
    return out


def is_good_phrase(p: str) -> bool:
    words = p.lower().split()
    return any(w in TECH_SIGNAL_WORDS for w in words)


def extract_phrases(text: str, n: int = 2, top_k: int = 25) -> List[str]:
    words = tokenize_words(text)
    phrases = []
    for i in range(len(words) - n + 1):
        phrases.append(" ".join(words[i:i+n]))

    freq: Dict[str, int] = {}
    for p in phrases:
        freq[p] = freq.get(p, 0) + 1

    items = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    filtered = [p for p, cnt in items if cnt > 1]
    filtered = [p for p in filtered if is_good_phrase(p)]

    if len(filtered) < 10:
        filtered = [p for p, _ in items]
        filtered = [p for p in filtered if is_good_phrase(p)]

    filtered = rerank_phrases_diverse(filtered, max_per_token=2)
    return filtered[:top_k]


def phrase_coverage(resume_text: str, job_text: str) -> Tuple[float, List[str], List[str]]:
    job_phrases = extract_phrases(job_text, n=3, top_k=25)
    resume_norm = normalize(resume_text)

    overlap = [p for p in job_phrases if p in resume_norm]
    missing = [p for p in job_phrases if p not in resume_norm]

    coverage = (len(overlap) / len(job_phrases)) if job_phrases else 0.0
    return coverage, overlap, missing


# -----------------------------
# Core "agent"
# -----------------------------
def run_agent(resume_text: str, job_text: str) -> OutputPacket:
    resume_kws = extract_keywords(resume_text, TECH_KEYWORDS)
    job_kws = extract_keywords(job_text, TECH_KEYWORDS)

    overlap = sorted(set(resume_kws).intersection(job_kws), key=str.lower)
    missing = sorted(set(job_kws).difference(resume_kws), key=str.lower)

    bank_cov = (len(overlap) / len(job_kws)) if job_kws else 0.0
    phrase_cov, phrase_overlap, phrase_missing = phrase_coverage(resume_text, job_text)

    red_flags = find_red_flags(job_text)
    confidence = min(0.95, 0.45 + 0.5 * (min(1.0, len(job_kws) / 12.0)))

    blended = (bank_cov * 0.35) + (phrase_cov * 0.65)

    scores = MatchScores(
        keyword_coverage=round(blended, 3),
        overlap_keywords=overlap + [f"[PHRASE] {p}" for p in phrase_overlap[:10]],
        missing_keywords=missing + [f"[PHRASE] {p}" for p in phrase_missing[:15]],
        red_flags=red_flags,
        confidence=round(confidence, 3),
    )

    bullets = split_bullets(resume_text)
    suggested_edits: List[SuggestedEdit] = []

    for b in bullets[:8]:
        edit = safe_rewrite_bullet(b, target_keywords=missing)
        if edit:
            suggested_edits.append(edit)

    role_summary = (
        "This agent compares your resume text to the job posting, "
        "identifies keyword overlap/gaps, and suggests safe, honest edits "
        "that improve alignment without inventing experience."
    )

    interview_questions, talking_points = build_interview_pack(job_text, overlap, missing)

    return OutputPacket(
        role_summary=role_summary,
        match_scores=scores,
        suggested_edits=suggested_edits,
        interview_questions=interview_questions,
        talking_points=talking_points,
    )


# -----------------------------
# Output writers
# -----------------------------

def input_stats(label: str, text: str) -> dict:
    text = text or ""
    return {
        "label": label,
        "chars": len(text),
        "words": len(text.split()),
        "preview": text[:200]
    }


def write_outputs(packet: OutputPacket, out_dir: Path, resume_stats: dict, job_stats: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


    md: List[str] = []
    md.append("# Job Application Copilot Packet\n")
    md.append("## Role Fit Summary\n")
    md.append(packet.role_summary + "\n")

    ms = packet.match_scores
    md.append("## Match Scores\n")
    md.append(f"- Keyword coverage: **{ms.keyword_coverage:.0%}**\n")
    md.append(f"- Confidence: **{ms.confidence:.0%}**\n")
    if ms.red_flags:
        md.append(f"- Red flags detected in posting: **{', '.join(ms.red_flags)}**\n")

    md.append("\n### Overlap Keywords\n")
    md.append(", ".join(ms.overlap_keywords) if ms.overlap_keywords else "_None detected from the keyword bank._")
    md.append("\n\n### Missing Keywords\n")
    md.append(", ".join(ms.missing_keywords) if ms.missing_keywords else "_None — strong keyword match._")
    md.append("\n")

    md.append("\n## Suggested Resume Edits (Safe + Honest)\n")
    if not packet.suggested_edits:
        md.append("_No suggested edits generated (try adding bullets or expanding resume text)._")
    else:
        for i, e in enumerate(packet.suggested_edits, 1):
            md.append(f"\n### Edit {i}\n")
            md.append(f"**Original:** {e.original}\n\n")
            md.append(f"**Suggestion:** {e.suggestion}\n\n")
            md.append(f"**Rationale:** {e.rationale}\n\n")
            if e.needs_confirmation:
                md.append("⚠️ **Needs confirmation**: this mentions a tool—only keep if true.\n")

    md.append("\n## Interview Prep\n")
    md.append("\n### Likely Questions\n")
    for q in packet.interview_questions:
        md.append(f"- {q}")

    md.append("\n\n### Talking Points\n")
    for tp in packet.talking_points:
        md.append(f"- {tp}")

    md.append("")

    (out_dir / "packet.md").write_text("\n".join(md), encoding="utf-8")

    changes = [asdict(e) for e in packet.suggested_edits]
    (out_dir / "changes.json").write_text(json.dumps(changes, indent=2), encoding="utf-8")
    score_payload = {
    "version": VERSION,
    "scores": asdict(packet.match_scores)
}

    (out_dir / "score.json").write_text(
    json.dumps(score_payload, indent=2),
    encoding="utf-8"
)



# -----------------------------
# Main
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Job Application Co-Pilot Agent (Mock Mode): compare resume vs job posting and generate a packet."
    )
    parser.add_argument("--resume", "-r", help="Path to a resume .txt file (optional).")
    parser.add_argument("--job", "-j", help="Path to a job posting .txt file (optional).")
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT_BASE),
        help="Output base directory (default: output). Each run is saved into output/run_YYYYMMDD_HHMMSS/."
    )
    parser.add_argument(
        "--fallback-default-files",
        action="store_true",
        help="If paste mode is used and you paste nothing, fall back to resume.txt/job_posting.txt if present."
    )
    args = parser.parse_args()

    out_base = Path(args.out)
    run_out_dir = make_run_dir(out_base)

    # Option A: paste mode unless BOTH files provided
    if args.resume and args.job:
        resume_text = read_text(Path(args.resume))
        job_text = read_text(Path(args.job))
    else:
        resume_text = collect_pasted_input("Paste RESUME text now:")
        job_text = collect_pasted_input("Paste JOB POSTING text now:")

        if args.fallback_default_files:
            if not resume_text:
                resume_text = read_text_optional(DEFAULT_RESUME_PATH)
            if not job_text:
                job_text = read_text_optional(DEFAULT_JOB_PATH)

    if not resume_text or not job_text:
        raise ValueError(
            "Missing inputs. Either paste resume + job text, OR provide --resume and --job file paths."
        )

    resume_stats = input_stats("resume", resume_text)
    job_stats = input_stats("job", job_text)

    packet = run_agent(resume_text, job_text)
    write_outputs(packet, run_out_dir, resume_stats, job_stats)

    print("✅ Done. Generated:")
    print(f"- {run_out_dir / 'packet.md'}")
    print(f"- {run_out_dir / 'changes.json'}")
    print(f"- {run_out_dir / 'score.json'}")
    print("Version:", VERSION)
    

if __name__ == "__main__":
    main()
