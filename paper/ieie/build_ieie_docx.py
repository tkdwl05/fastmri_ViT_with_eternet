#!/usr/bin/env python
"""IEIE(대한전자공학회) 투고용 양식 초안 빌더 — stdlib 만 사용 (python-docx/pandoc 불필요).

입력  : paper/ieie/draft_ieie_v7.src.md         (본문 소스: 디렉티브 + 마크다운 일부 — 학술대회 빌더와 **같은 소스**)
        paper/references.bib                    (서지 — [@key] 인용을 IEIE 영문 형식 [n] 으로 변환)
        paper/ieie/template_ieie_2021.docx      (학회 투고용 논문 양식 2021 — 스타일/머리글/섹션 설정을 그대로 재사용)
        paper/figs/*.png                        (그림)
출력  : paper/ieie/draft_ieie_v7.md             (읽기용 마크다운, 번호·참고문헌 확정본)
        paper/ieie/draft_ieie_v7.docx           (양식 적용 docx = 서면 심사용. 논문지 양식엔 저자란이 없고, 본문의
                                                 저자·소속 단서는 check_blind() 가 막는다 — 발견 시 탈락 규정)

실행  : CUDA_VISIBLE_DEVICES="" python paper/ieie/build_ieie_docx.py [--src <x.src.md>] [--out <stem>] [--no-blind-check]
        (09-09 교수님 지시: 프로시딩 게재용과 내용 동일, 저자·소속만 삭제 → 같은 소스를 build_ieie_conf_docx.py 로도 빌드.
         옛 v1 소스·산출물은 paper/ieie/archive/.
         09-14 교수님 지시(v7): 캡션은 **영문**(@cap_en 필수, @cap_ko 는 선택 — 있으면 국문 줄을 앞에 병기), 본문의 수식 기호는
         인라인 수식 $…$ 로 써서 첨자를 Word 수식 객체(OMML)로 낸다(교수님이 y_c 를 직접 수식 객체로 고친 방식과 동일),
         표 주석은 본문 크기.)

소스 디렉티브 (draft_ieie_v7.src.md):
  %% 주석                       빌더가 무시
  @title_ko: / @title_en: / @keywords:
  @abstract_ko: / @abstract_en:   다음 @ 디렉티브 전까지의 블록
  @body:                          이후 본문
  # 장 제목   ## 절 제목   ### 항 제목   (장 = Ⅰ. Ⅱ. …, 절 = 1. 2. …, 항 = 가. 나. …;  "# REFERENCES" 는 자동 목록)
  $$ latex $$                     한 줄 display 수식 → OMML (LaTeX 부분집합)
  $latex$                         본문 안 인라인 수식(첨자·기호: $\tilde{y}_c$, $f_\theta$, $d_{\mathrm{inner}}$) → 인라인 OMML
  @figure: path | page|col | scale   + @cap_ko: / @cap_en:
  @table: page|col | w1,w2,...(twips) + @cap_ko: / @cap_en: + 마크다운 표 행 + (@note:) + @end
  [@key; @key2]                   서지 인용 → 첫 등장 순 [n] (본문은 위첨자, 표 안은 일반)
  [TBD ...]                       미확정 표시 → docx 노란 형광
"""
from __future__ import annotations

import os
import re
import struct
import sys
import zipfile
import xml.dom.minidom as minidom
from xml.etree import ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IEIE = os.path.join(ROOT, "paper", "ieie")
SRC = os.path.join(IEIE, "draft_ieie_v7.src.md")
BIB = os.path.join(ROOT, "paper", "references.bib")
TEMPLATE = os.path.join(IEIE, "template_ieie_2021.docx")
OUT_MD = os.path.join(IEIE, "draft_ieie_v7.md")
OUT_DOCX = os.path.join(IEIE, "draft_ieie_v7.docx")

PAGE_W = 9637   # 본문 폭 (twips): 11906 - 1134 - 1135
COL_W = 4563    # 2단 한 단 폭 (twips): (9637 - 510) / 2
EMU_PER_TWIP = 635
ROMAN = ["Ⅰ", "Ⅱ", "Ⅲ", "Ⅳ", "Ⅴ", "Ⅵ", "Ⅶ", "Ⅷ", "Ⅸ", "Ⅹ"]
HANGUL = ["가", "나", "다", "라", "마", "바", "사", "아", "자", "차"]


# ----------------------------------------------------------------------------- bib
_TEX_ACCENTS = {
    '\\"': {"a": "ä", "o": "ö", "u": "ü", "A": "Ä", "O": "Ö", "U": "Ü", "e": "ë", "i": "ï"},
    "\\'": {"a": "á", "e": "é", "i": "í", "o": "ó", "u": "ú", "c": "ć", "n": "ń", "y": "ý", "s": "ś", "A": "Á", "E": "É"},
    "\\`": {"a": "à", "e": "è", "i": "ì", "o": "ò", "u": "ù"},
    "\\^": {"a": "â", "e": "ê", "i": "î", "o": "ô", "u": "û"},
    "\\~": {"a": "ã", "o": "õ", "n": "ñ"},
    "\\c": {"c": "ç", "C": "Ç", "s": "ş"},
    "\\v": {"c": "č", "s": "š", "z": "ž", "C": "Č", "S": "Š", "Z": "Ž", "r": "ř", "e": "ě"},
    "\\u": {"a": "ă", "g": "ğ"},
    "\\H": {"o": "ő", "u": "ű"},
    "\\.": {"z": "ż"},
    "\\=": {"a": "ā", "e": "ē", "i": "ī", "o": "ō", "u": "ū"},
}


def clean_tex(s: str) -> str:
    """LaTeX 악센트/보호 중괄호/특수 시퀀스를 유니코드 평문으로."""
    def acc(m):
        cmd, ch = m.group(1), m.group(2)
        return _TEX_ACCENTS.get(cmd, {}).get(ch, ch)
    # {\"u}  \"{u}  \"u   {\c{c}}  \c{c}
    s = re.sub(r"\{?(\\[\"'`^~cvuH.=])\{?([A-Za-z])\}?\}?", acc, s)
    s = s.replace("{\\o}", "ø").replace("\\o", "ø").replace("{\\ss}", "ß").replace("\\&", "&").replace("\\%", "%")
    s = s.replace("--", "-").replace("~", " ")
    s = re.sub(r"\\textit\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\emph\{([^}]*)\}", r"\1", s)
    s = s.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", s).strip()


def parse_bib(path: str) -> dict:
    txt = open(path, encoding="utf-8").read()
    entries = {}
    for m in re.finditer(r"@(\w+)\s*\{", txt):
        typ = m.group(1).lower()
        if typ in ("comment", "preamble", "string"):
            continue
        start = m.end() - 1
        depth, i = 0, start
        while True:
            c = txt[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        body = txt[start + 1:i]
        key, rest = body.split(",", 1)
        key = key.strip()
        fields, pos = {}, 0
        while pos < len(rest):
            fm = re.compile(r"\s*(\w+)\s*=\s*").match(rest, pos)
            if not fm:
                break
            name, pos = fm.group(1).lower(), fm.end()
            if pos >= len(rest):
                break
            if rest[pos] == "{":
                d, j = 0, pos
                while True:
                    if rest[j] == "{":
                        d += 1
                    elif rest[j] == "}":
                        d -= 1
                        if d == 0:
                            break
                    j += 1
                val, pos = rest[pos + 1:j], j + 1
            elif rest[pos] == '"':
                j = rest.index('"', pos + 1)
                val, pos = rest[pos + 1:j], j + 1
            else:
                j = re.compile(r"[^,]*").match(rest, pos).end()
                val, pos = rest[pos:j].strip(), j
            fields[name] = val
            pos = re.compile(r"\s*,?").match(rest, pos).end()
        entries[key] = (typ, fields)
    return entries


def fmt_person(name: str) -> str:
    name = clean_tex(name)
    if name.lower() == "others":
        return "et al."
    if "," in name:
        last, first = [p.strip() for p in name.split(",", 1)]
    else:
        parts = name.split()
        last, first = parts[-1], " ".join(parts[:-1])
    initials = []
    for tok in first.split():
        initials.append("-".join(p[0] + "." for p in tok.split("-") if p))
    return (" ".join(initials) + " " + last).strip()


def fmt_authors(field: str) -> str:
    people = [fmt_person(p) for p in re.split(r"\s+and\s+", field.replace("\n", " "))]
    if len(people) > 6:
        return people[0] + " et al."
    if people[-1] == "et al.":
        return ", ".join(people[:-1]) + " et al."
    if len(people) == 1:
        return people[0]
    if len(people) == 2:
        return f"{people[0]} and {people[1]}"
    return ", ".join(people[:-1]) + ", and " + people[-1]


def fmt_pages(p: str) -> str:
    # v3(사용자 편집본) 서지 표기: 쪽 범위는 en dash — "pp. 64–73"
    p = clean_tex(p)
    if "-" in p:
        return "pp. " + re.sub(r"\s*-+\s*", "\u2013", p)
    return f"Art. no. {p}"


# 09-09 감사 반영: IEIE 참고문헌 관례(약어 저널명·Proc. 학회명)로만 축약. 공유 bib(references.bib) 는 MDPI 초안이
# 함께 쓰므로 손대지 않고 이 빌더(학술대회판도 import) 에서만 치환한다. clean_tex 뒤의 문자열 기준, 미등록 = 원문 유지.
VENUE_ABBR = {
    "Medical Physics": "Med. Phys.",
    "Medical Image Analysis": "Med. Image Anal.",
    "Magnetic Resonance in Medicine": "Magn. Reson. Med.",
    "Journal of Imaging": "J. Imaging",
    "Medical Image Computing and Computer Assisted Intervention (MICCAI 2020)": "Proc. MICCAI",
    "Advances in Neural Information Processing Systems (NeurIPS)": "Proc. NeurIPS",
    "IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)": "Proc. IEEE/CVF WACV",
    "Computer Vision - ECCV 2024": "Proc. ECCV",
    "Computer Vision -- ECCV 2024": "Proc. ECCV",
}


def abbr_venue(name: str) -> str:
    name = clean_tex(name)
    return VENUE_ABBR.get(name, VENUE_ABBR.get(name.replace("\u2013", "-").replace("\u2014", "-"), name))


def fmt_reference(typ: str, f: dict) -> list:
    """IEIE 영문 서지 → [(text, italic)] 런 목록."""
    runs = []
    auth = fmt_authors(f.get("author", "")) if f.get("author") else ""
    title = clean_tex(f.get("title", ""))
    year = clean_tex(f.get("year", ""))
    head = f'{auth}, "{title}," ' if auth else f'"{title}," '
    runs.append((head, False))
    tail = []
    if typ == "article":
        journal = abbr_venue(f.get("journal", ""))
        if journal:
            runs.append((journal, True))
        if f.get("volume"):
            tail.append("vol. " + clean_tex(f["volume"]))
        if f.get("number"):
            tail.append("no. " + clean_tex(f["number"]))
        if f.get("pages"):
            tail.append(fmt_pages(f["pages"]))
    elif typ in ("inproceedings", "incollection", "conference"):
        book = abbr_venue(f.get("booktitle", ""))
        runs.append(("in ", False))
        if book:
            runs.append((book, True))
        if f.get("series"):
            tail.append(clean_tex(f["series"]))
        if f.get("volume"):
            tail.append("vol. " + clean_tex(f["volume"]))
        if f.get("pages"):
            tail.append(fmt_pages(f["pages"]))
    elif typ == "misc":
        if f.get("eprint"):
            runs.append(("arXiv preprint arXiv:" + clean_tex(f["eprint"]), False))
        elif f.get("howpublished"):
            runs.append((clean_tex(f["howpublished"]), False))
        elif f.get("url"):
            runs.append((clean_tex(f["url"]), False))
    elif typ in ("book",):
        if f.get("publisher"):
            runs.append((clean_tex(f["publisher"]), False))
    else:
        if f.get("journal") or f.get("booktitle"):
            runs.append((abbr_venue(f.get("journal") or f.get("booktitle")), True))
    if year:
        tail.append(year)
    if tail:
        runs.append((", " + ", ".join(tail) + ".", False))
    else:
        runs.append((".", False))
    # 마지막 텍스트 정리: '," , Vol.' 같은 이중 구두점 방지
    return runs


# ----------------------------------------------------------------------------- source
class Doc:
    def __init__(self):
        self.meta = {}
        self.blocks = []   # dicts: kind = chapter/section/subsection/para/eq/figure/table


def parse_src(path: str) -> Doc:
    doc = Doc()
    lines = [ln.rstrip("\n") for ln in open(path, encoding="utf-8")]
    lines = [ln for ln in lines if not ln.startswith("%%")]
    # front matter
    i = 0
    cur_key = None
    body_start = None
    while i < len(lines):
        ln = lines[i]
        m = re.match(r"@(\w+):\s*(.*)$", ln)
        if m and m.group(1) in ("title_ko", "title_en", "keywords", "abstract_ko", "abstract_en", "body"):
            if m.group(1) == "body":
                body_start = i + 1
                break
            cur_key = m.group(1)
            doc.meta[cur_key] = m.group(2).strip()
        elif cur_key:
            doc.meta[cur_key] = (doc.meta[cur_key] + "\n" + ln).strip()
        i += 1
    for k in ("abstract_ko", "abstract_en"):
        doc.meta[k] = [p.replace("\n", " ").strip() for p in re.split(r"\n\s*\n", doc.meta.get(k, "")) if p.strip()]
    assert body_start is not None, "@body: 디렉티브가 없습니다"

    blocks = doc.blocks
    para: list[str] = []

    def flush():
        if para:
            blocks.append({"kind": "para", "text": " ".join(s.strip() for s in para)})
            para.clear()

    i = body_start
    while i < len(lines):
        ln = lines[i]
        if not ln.strip():
            flush(); i += 1; continue
        if ln.startswith("### "):
            flush(); blocks.append({"kind": "subsection", "title": ln[4:].strip()}); i += 1; continue
        if ln.startswith("## "):
            flush(); blocks.append({"kind": "section", "title": ln[3:].strip()}); i += 1; continue
        if ln.startswith("# "):
            flush(); blocks.append({"kind": "chapter", "title": ln[2:].strip()}); i += 1; continue
        if ln.startswith("$$"):
            flush()
            latex = ln.strip()[2:-2].strip() if ln.strip().endswith("$$") else ln.strip()[2:].strip()
            blocks.append({"kind": "eq", "latex": latex}); i += 1; continue
        if ln.startswith("@figure:"):
            flush()
            parts = [p.strip() for p in ln[len("@figure:"):].split("|")]
            fig = {"kind": "figure", "path": parts[0], "width": parts[1] if len(parts) > 1 else "col",
                   "scale": float(parts[2]) if len(parts) > 2 else 1.0, "cap_ko": "", "cap_en": ""}
            i += 1
            while i < len(lines) and lines[i].startswith("@cap_"):
                k, v = lines[i].split(":", 1)
                fig[k[1:]] = v.strip(); i += 1
            blocks.append(fig); continue
        if ln.startswith("@table:"):
            flush()
            parts = [p.strip() for p in ln[len("@table:"):].split("|")]
            tbl = {"kind": "table", "width": parts[0], "colw": [int(x) for x in parts[1].split(",")] if len(parts) > 1 else None,
                   "cap_ko": "", "cap_en": "", "rows": [], "note": ""}
            i += 1
            while i < len(lines) and not lines[i].startswith("@end"):
                s = lines[i]
                if s.startswith("@cap_"):
                    k, v = s.split(":", 1); tbl[k[1:]] = v.strip()
                elif s.startswith("@note:"):
                    tbl["note"] = s[len("@note:"):].strip()
                elif s.strip().startswith("|"):
                    cells = [c.strip() for c in s.strip().strip("|").split("|")]
                    if all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                        pass  # 구분선
                    else:
                        tbl["rows"].append(cells)
                i += 1
            i += 1  # @end
            blocks.append(tbl); continue
        para.append(ln); i += 1
    flush()
    return doc


# ----------------------------------------------------------------------------- numbering
class Numberer:
    def __init__(self, bib: dict):
        self.bib = bib
        self.order: list[str] = []

    def num(self, key: str) -> int:
        if key not in self.bib:
            raise KeyError(f"references.bib 에 없는 키: {key}")
        if key not in self.order:
            self.order.append(key)
        return self.order.index(key) + 1

    @staticmethod
    def compress(nums: list[int]) -> str:
        nums = sorted(set(nums))
        out, i = [], 0
        while i < len(nums):
            j = i
            while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
                j += 1
            if j - i >= 2:
                out.append(f"{nums[i]}-{nums[j]}")
            else:
                out.extend(str(n) for n in nums[i:j + 1])
            i = j + 1
        return ", ".join(out)


CITE_RE = re.compile(r"\[(@[^\]]+)\]")
TBD_RE = re.compile(r"\[TBD[^\]]*\]")
MATH_RE = re.compile(r"\$([^$]+?)\$")   # 인라인 수식 $…$ (display 수식 "$$ … $$" 줄은 parse_src 가 줄 단위로 먼저 뗀다)


def inline_runs(text: str, nb: Numberer, cite_sup: bool = True) -> list:
    """텍스트 → [(text, {sup, hl, b, u, math})] 런. 인라인 수식 $…$ 를 먼저 떼어 math 런으로 만들고(수식 안의 _ * 가
    굵게/밑줄/TBD 마크업에 닿지 않도록), 나머지 구간에서 인용 번호를 확정(첫 등장 순)한다."""
    assert text.count("$") % 2 == 0, f"인라인 수식 $ 짝이 맞지 않음: {text[:80]}"
    runs = []
    pos = 0
    for m in MATH_RE.finditer(text):
        if m.start() > pos:
            runs.extend(_cite_split(text[pos:m.start()], nb, cite_sup))
        runs.append((m.group(1).strip(), {"math": True}))
        pos = m.end()
    if pos < len(text):
        runs.extend(_cite_split(text[pos:], nb, cite_sup))
    return runs


def _cite_split(text: str, nb: Numberer, cite_sup: bool) -> list:
    runs = []
    pos = 0
    for m in CITE_RE.finditer(text):
        if m.start() > pos:
            runs.extend(_tbd_split(text[pos:m.start()]))
        keys = [k.strip().lstrip("@") for k in m.group(1).split(";") if k.strip()]
        nums = [nb.num(k) for k in keys]
        label = "[" + nb.compress(nums) + "]"
        runs.append((label, {"sup": cite_sup}))
        pos = m.end()
    if pos < len(text):
        runs.extend(_tbd_split(text[pos:]))
    return runs


MARK_RE = re.compile(r"\*\*(?=\S)([^*]+?)(?<=\S)\*\*|__(?=\S)([^_]+?)(?<=\S)__")   # **굵게** / __밑줄__ (표 셀의 최고값·차선값 표시; 저자 자리표시자 "***" 는 매치 안 됨)


def _tbd_split(s: str) -> list:
    out, pos = [], 0
    for m in TBD_RE.finditer(s):
        if m.start() > pos:
            out.extend(_mark_split(s[pos:m.start()]))
        out.append((m.group(0), {"hl": True}))
        pos = m.end()
    if pos < len(s):
        out.extend(_mark_split(s[pos:]))
    return out


def _mark_split(s: str) -> list:
    out, pos = [], 0
    for m in MARK_RE.finditer(s):
        if m.start() > pos:
            out.append((s[pos:m.start()], {}))
        if m.group(1) is not None:
            out.append((m.group(1), {"b": True}))
        else:
            out.append((m.group(2), {"u": True}))
        pos = m.end()
    if pos < len(s):
        out.append((s[pos:], {}))
    return out


def runs_md(runs: list) -> str:
    """md 출력용: 굵게/밑줄 런을 마크다운 표기로 되돌리고 인라인 수식은 $…$ 로 유지한다."""
    return "".join(f"${t}$" if f.get("math") else f"**{t}**" if f.get("b") else f"<u>{t}</u>" if f.get("u") else t for t, f in runs)


def runs_text(runs: list) -> str:
    return "".join(f"${t}$" if f.get("math") else t for t, f in runs)


# ----------------------------------------------------------------------------- OMML (LaTeX 부분집합)
GREEK = {"alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε", "theta": "θ", "lambda": "λ", "mu": "μ",
         "phi": "φ", "sigma": "σ", "tau": "τ", "omega": "ω", "Delta": "Δ", "Sigma": "Σ", "Omega": "Ω", "Phi": "Φ",
         "Theta": "Θ", "Lambda": "Λ", "pi": "π", "rho": "ρ", "eta": "η", "xi": "ξ", "kappa": "κ", "nu": "ν"}
SYMBOLS = {"odot": "⊙", "ldots": "…", "cdots": "⋯", "times": "×", "cdot": "·", "in": "∈", "infty": "∞", "pm": "±",
           "le": "≤", "ge": "≥", "neq": "≠", "approx": "≈", "rightarrow": "→", "leftarrow": "←", "to": "→",
           "mid": "|", "circ": "∘", "ast": "∗", "star": "⋆", "partial": "∂", "nabla": "∇"}
SPACES = {"quad": "\u2003", "qquad": "\u2003\u2003", ",": "\u2009", ";": "\u2005", " ": " ", "!": ""}
FUNCS = {"exp", "log", "max", "min", "sin", "cos", "tan", "ln", "det", "arg", "lim"}
ACCENTS = {"hat": "\u0302", "tilde": "\u0303", "vec": "\u20d7", "dot": "\u0307", "ddot": "\u0308", "breve": "\u0306"}
NARY = {"sum": "∑", "prod": "∏", "int": "∫"}


def _xml(t: str) -> str:
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def mr(t: str, sty: str = None, scr: str = None) -> str:
    rpr = ""
    if sty or scr:
        rpr = "<m:rPr>" + (f'<m:scr m:val="{scr}"/>' if scr else "") + (f'<m:sty m:val="{sty}"/>' if sty else "") + "</m:rPr>"
    return f'<m:r>{rpr}<m:t xml:space="preserve">{_xml(t)}</m:t></m:r>'


class Latex:
    """아주 작은 LaTeX → OMML 변환기 (논문 수식 (1)~(6) 에 필요한 부분집합)."""

    def __init__(self, s: str):
        self.toks = self.tokenize(s)
        self.i = 0

    @staticmethod
    def tokenize(s: str) -> list:
        toks, i = [], 0
        while i < len(s):
            c = s[i]
            if c == "\\":
                m = re.compile(r"\\([A-Za-z]+|.)").match(s, i)
                toks.append(("cmd", m.group(1))); i = m.end()
            elif c in "{}_^":
                toks.append((c, c)); i += 1
            elif c.isspace():
                i += 1
            else:
                toks.append(("ch", c)); i += 1
        return toks

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else (None, None)

    def next(self):
        t = self.toks[self.i]; self.i += 1; return t

    def group(self) -> str:
        """{...} 그룹 또는 단일 토큰의 OMML."""
        k, v = self.peek()
        if k == "{":
            self.next()
            out = self.seq(stop="}")
            self.next()  # }
            return out
        return self.atom()

    def seq(self, stop=None) -> str:
        out = []
        while True:
            k, v = self.peek()
            if k is None or (stop and k == stop) or (k == "cmd" and v == "right"):
                break
            out.append(self.atom())
        return "".join(out)

    def atom(self) -> str:
        k, v = self.next()
        base = self._base(k, v)
        # 첨자 (순서 무관, 최대 sub+sup)
        sub = sup = None
        while True:
            pk, pv = self.peek()
            if pk == "_" and sub is None:
                self.next(); sub = self.group()
            elif pk == "^" and sup is None:
                self.next(); sup = self.group()
            else:
                break
        if sub is not None and sup is not None:
            return f"<m:sSubSup><m:e>{base}</m:e><m:sub>{sub}</m:sub><m:sup>{sup}</m:sup></m:sSubSup>"
        if sub is not None:
            return f"<m:sSub><m:e>{base}</m:e><m:sub>{sub}</m:sub></m:sSub>"
        if sup is not None:
            return f"<m:sSup><m:e>{base}</m:e><m:sup>{sup}</m:sup></m:sSup>"
        return base

    def _base(self, k, v) -> str:
        if k == "ch":
            if v == "-":
                return mr("−")
            if v == "*":
                return mr("∗")
            return mr(v)
        if k == "{":
            out = self.seq(stop="}"); self.next(); return out
        if k == "cmd":
            if v in GREEK:
                return mr(GREEK[v])
            if v in SYMBOLS:
                return mr(SYMBOLS[v])
            if v in SPACES:
                return mr(SPACES[v]) if SPACES[v] else ""
            if v in FUNCS:
                return mr(v, sty="p")
            if v in ("{", "}", "|"):
                return mr(v)
            if v == "mathrm":
                return f"<m:r><m:rPr><m:sty m:val=\"p\"/></m:rPr><m:t xml:space=\"preserve\">{_xml(self.rawgroup())}</m:t></m:r>"
            if v == "mathcal":
                return f"<m:r><m:rPr><m:scr m:val=\"script\"/></m:rPr><m:t>{_xml(self.rawgroup())}</m:t></m:r>"
            if v == "mathbf":
                return f"<m:r><m:rPr><m:sty m:val=\"b\"/></m:rPr><m:t>{_xml(self.rawgroup())}</m:t></m:r>"
            if v == "text":
                return f"<m:r><m:rPr><m:sty m:val=\"p\"/></m:rPr><m:t xml:space=\"preserve\">{_xml(self.rawgroup())}</m:t></m:r>"
            if v == "frac":
                num = self.group(); den = self.group()
                return f"<m:f><m:num>{num}</m:num><m:den>{den}</m:den></m:f>"
            if v == "sqrt":
                e = self.group()
                return f'<m:rad><m:radPr><m:degHide m:val="1"/></m:radPr><m:deg/><m:e>{e}</m:e></m:rad>'
            if v in ACCENTS:
                e = self.group()
                return f'<m:acc><m:accPr><m:chr m:val="{ACCENTS[v]}"/></m:accPr><m:e>{e}</m:e></m:acc>'
            if v == "bar":
                e = self.group()
                return f'<m:bar><m:barPr><m:pos m:val="top"/></m:barPr><m:e>{e}</m:e></m:bar>'
            if v in NARY:
                sub = sup = ""
                while True:
                    pk, pv = self.peek()
                    if pk == "_":
                        self.next(); sub = self.group()
                    elif pk == "^":
                        self.next(); sup = self.group()
                    else:
                        break
                pr = f'<m:chr m:val="{NARY[v]}"/><m:limLoc m:val="undOvr"/>'
                if not sub:
                    pr += '<m:subHide m:val="1"/>'
                if not sup:
                    pr += '<m:supHide m:val="1"/>'
                return f"<m:nary><m:naryPr>{pr}</m:naryPr><m:sub>{sub}</m:sub><m:sup>{sup}</m:sup><m:e/></m:nary>"
            if v in ("max", "min") :
                return mr(v, sty="p")
            if v == "left":
                ok, ov = self.next()
                beg = ov if ok == "ch" else {"|": "|", "{": "{", "}": "}"}.get(ov, ov)
                body = self.seq()
                rk, rv = self.next()  # \right
                assert rv == "right", "\\left 에 대응하는 \\right 가 없습니다"
                ek, ev = self.next()
                end = ev if ek == "ch" else {"|": "|", "{": "{", "}": "}"}.get(ev, ev)
                if beg == ".":
                    beg = ""
                if end == ".":
                    end = ""
                pr = f'<m:begChr m:val="{beg}"/><m:endChr m:val="{end}"/>' if (beg, end) != ("(", ")") else ""
                return f"<m:d><m:dPr>{pr}</m:dPr><m:e>{body}</m:e></m:d>"
            if v == "operatorname":
                return mr(self.rawgroup(), sty="p")
            raise ValueError(f"지원하지 않는 LaTeX 명령: \\{v}")
        raise ValueError(f"예상치 못한 토큰: {k} {v}")

    def rawgroup(self) -> str:
        k, v = self.next()
        assert k == "{", "그룹이 필요합니다"
        out = []
        while True:
            k, v = self.next()
            if k == "}":
                break
            out.append(v if k != "cmd" else GREEK.get(v, v))
        return "".join(out)


def latex_to_omml(latex: str, rpr: str = "") -> str:
    """rpr(<w:sz>/<w:szCs> 등 w:rPr 자식)을 주면 각 수식 런 <m:r> 에 <w:rPr> 로 넣는다 — 인라인 수식을 둘러싼 본문 글자
    크기에 맞추는 용도(교수님이 Word 에서 고친 인라인 수식 객체와 같은 구조: <m:r><w:rPr><w:sz/>…</w:rPr><m:t>).
    글꼴(rFonts)은 넣지 않는다 — 두 양식의 settings.xml 이 수식 글꼴을 Cambria Math 로 지정하고 있다."""
    p = Latex(latex)
    body = p.seq()
    assert p.peek()[0] is None, f"수식 파싱 잔여 토큰: {p.toks[p.i:]}"
    body = body.replace('</m:t></m:r><m:r><m:t xml:space="preserve">', "")  # 인접한 평문 런 병합
    if rpr:
        body = re.sub(r"<m:r>((?:<m:rPr>.*?</m:rPr>)?)", lambda m: "<m:r>" + m.group(1) + f"<w:rPr>{rpr}</w:rPr>", body)
    return f"<m:oMath>{body}</m:oMath>"


def latex_to_plain(latex: str) -> str:
    """마크다운용: LaTeX 원문 유지."""
    return latex


# ----------------------------------------------------------------------------- docx XML helpers
def wt(t: str) -> str:
    return f'<w:t xml:space="preserve">{_xml(t)}</w:t>'


def wr(t: str, rpr: str = "") -> str:
    return f"<w:r>{('<w:rPr>' + rpr + '</w:rPr>') if rpr else ''}{wt(t)}</w:r>"


def runs_xml(runs: list, base_rpr: str = "") -> str:
    """base_rpr 는 <w:sz>/<w:szCs> 만 허용 — 스키마 순서(b, i, sz, szCs, highlight, u, vertAlign)를 지켜 조립.
    math 런은 인라인 OMML 로 내며 base_rpr(글자 크기)만 물려받는다(본문은 스타일 크기 상속, 표 셀·주석은 자기 크기)."""
    out = []
    for t, f in runs:
        if f.get("math"):
            out.append(latex_to_omml(t, rpr=base_rpr))
            continue
        rpr = ("<w:b/>" if f.get("b") else "") + ("<w:i/>" if f.get("i") else "") + base_rpr
        if f.get("hl"):
            rpr += '<w:highlight w:val="yellow"/>'
        if f.get("u"):
            rpr += '<w:u w:val="single"/>'
        if f.get("sup"):
            rpr += '<w:vertAlign w:val="superscript"/>'
        out.append(wr(t, rpr))
    return "".join(out)


def para(style: str, inner: str, ppr_extra: str = "") -> str:
    return f'<w:p><w:pPr><w:pStyle w:val="{style}"/>{ppr_extra}</w:pPr>{inner}</w:p>'


def png_size(path: str) -> tuple:
    with open(path, "rb") as fh:
        head = fh.read(24)
    assert head[:8] == b"\x89PNG\r\n\x1a\n", f"PNG 아님: {path}"
    w, h = struct.unpack(">II", head[16:24])
    return w, h


# 부유 표(페이지 폭 그림/표) 바로 뒤의 앵커용 빈 문단 — 높이 1pt 고정이라 본문 흐름에 거의 보이지 않음
ANCHOR_P = '<w:p><w:pPr><w:pStyle w:val="a3"/><w:spacing w:before="0" w:after="0" w:line="20" w:lineRule="exact"/><w:ind w:firstLine="0"/></w:pPr></w:p>'


class DocxBuilder:
    def __init__(self, nb: Numberer):
        self.nb = nb
        self.body: list[str] = []
        self.media: list[tuple] = []   # (partname, bytes)
        self.rels: list[tuple] = []    # (rId, type, target)
        self.next_rid = 14
        self.docpr_id = 100
        self.fig_no = 0
        self.tbl_no = 0
        self.eq_no = 0

    # ---- text blocks
    def chapter(self, idx: int, title: str):
        if title.strip().upper() == "REFERENCES":
            self.body.append(para("a5", wr("REFERENCES")))
            return
        t = re.sub(r"^(\S{2})$", lambda m: m.group(1)[0] + "  " + m.group(1)[1], title.strip())  # 두 글자 제목 → "서  론"
        inner = f'<w:r><w:rPr><w:rFonts w:ascii="바탕"/></w:rPr>{wt(ROMAN[idx])}</w:r>{wr(". ")}{runs_xml(inline_runs(t, self.nb))}'
        self.body.append(para("a5", inner))

    def section(self, idx: int, title: str):
        self.body.append(para("a6", wr(f"{idx}. ") + runs_xml(inline_runs(title, self.nb))))

    def subsection(self, idx: int, title: str):
        self.body.append(para("a7", wr(f"{HANGUL[idx - 1]}. ") + runs_xml(inline_runs(title, self.nb))))

    def paragraph(self, text: str):
        self.body.append(para("a4", runs_xml(inline_runs(text, self.nb))))

    def equation(self, latex: str) -> int:
        self.eq_no += 1
        omml = latex_to_omml(latex)
        ppr = ('<w:tabs><w:tab w:val="clear" w:pos="408"/><w:tab w:val="clear" w:pos="686"/><w:tab w:val="clear" w:pos="756"/>'
               f'<w:tab w:val="center" w:pos="{COL_W // 2 - 150}"/><w:tab w:val="right" w:pos="{COL_W}"/></w:tabs>'
               '<w:spacing w:before="100" w:after="160"/><w:ind w:left="0" w:firstLine="0"/><w:jc w:val="left"/>')
        inner = "<w:r><w:tab/></w:r>" + omml + f"<w:r><w:tab/>{wt(f'({self.eq_no})')}</w:r>"
        self.body.append(para("af6", inner, ppr))
        return self.eq_no

    # ---- captions
    # 캡션: 09-14 교수님 지시 — 그림·표 캡션은 영문(@cap_en 필수; 양식의 붉은 주석 "표와 그림 : 영문" 과도 일치).
    # @cap_ko 가 있으면 양식 순서대로 국문 줄을 앞에 병기한다(현재 소스는 영문만 사용).
    def _fig_caption(self, no: int, ko: str, en: str) -> str:
        lines = []
        if ko.strip():
            lines.append(para("a8", wr("그림") + "<w:r><w:tab/></w:r>" + wr(f"{no}.") + "<w:r><w:tab/></w:r>" + runs_xml(inline_runs(ko, self.nb))))
        if en.strip():
            lines.append(para("a8", wr("Fig.") + "<w:r><w:tab/></w:r>" + wr(f"{no}.") + "<w:r><w:tab/></w:r>" + runs_xml(inline_runs(en, self.nb))))
        assert lines, f"그림 {no}: 캡션(@cap_en)이 없습니다"
        return "".join(lines)

    def _tbl_caption(self, no: int, ko: str, en: str) -> str:
        ppr = '<w:tabs><w:tab w:val="left" w:pos="466"/></w:tabs>'
        lines = []
        if ko.strip():
            lines.append(para("a8", wr("표") + "<w:r><w:tab/></w:r>" + wr(f"{no}.  ") + runs_xml(inline_runs(ko, self.nb)), ppr))
        if en.strip():
            lines.append(para("a8", wr("Table") + "<w:r><w:tab/></w:r>" + wr(f"{no}.  ") + runs_xml(inline_runs(en, self.nb)), ppr))
        assert lines, f"표 {no}: 캡션(@cap_en)이 없습니다"
        return "".join(lines)

    # ---- float wrapper (page-wide items in a 2-column section)
    def _float_wrap(self, inner: str) -> str:
        w = PAGE_W
        return ('<w:tbl><w:tblPr>'
                '<w:tblpPr w:leftFromText="0" w:rightFromText="0" w:topFromText="0" w:bottomFromText="170" '
                'w:vertAnchor="margin" w:horzAnchor="margin" w:tblpXSpec="center" w:tblpYSpec="top"/>'
                '<w:tblOverlap w:val="never"/>'
                f'<w:tblW w:w="{w}" w:type="dxa"/><w:tblLayout w:type="fixed"/>'
                '<w:tblCellMar><w:left w:w="0" w:type="dxa"/><w:right w:w="0" w:type="dxa"/></w:tblCellMar>'
                '<w:tblLook w:val="0000" w:firstRow="0" w:lastRow="0" w:firstColumn="0" w:lastColumn="0" w:noHBand="0" w:noVBand="0"/>'
                f'</w:tblPr><w:tblGrid><w:gridCol w:w="{w}"/></w:tblGrid>'
                f'<w:tr><w:tc><w:tcPr><w:tcW w:w="{w}" w:type="dxa"/></w:tcPr>{inner}</w:tc></w:tr></w:tbl>')

    # ---- figures
    def figure(self, path: str, width: str, scale: float, ko: str, en: str) -> int:
        self.fig_no += 1
        abspath = os.path.join(ROOT, path)
        pw, ph = png_size(abspath)
        avail = PAGE_W if width == "page" else COL_W
        w_tw = int(avail * scale)
        h_tw = int(w_tw * ph / pw)
        cx, cy = w_tw * EMU_PER_TWIP, h_tw * EMU_PER_TWIP
        rid = f"rId{self.next_rid}"; self.next_rid += 1
        part = f"media/fig{self.fig_no}.png"
        self.media.append((f"word/{part}", open(abspath, "rb").read()))
        self.rels.append((rid, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image", part))
        self.docpr_id += 1
        drawing = (
            '<w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0">'
            f'<wp:extent cx="{cx}" cy="{cy}"/><wp:effectExtent l="0" t="0" r="0" b="0"/>'
            f'<wp:docPr id="{self.docpr_id}" name="Figure {self.fig_no}"/>'
            '<wp:cNvGraphicFramePr><a:graphicFrameLocks xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" noChangeAspect="1"/></wp:cNvGraphicFramePr>'
            '<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
            '<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
            '<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
            f'<pic:nvPicPr><pic:cNvPr id="0" name="fig{self.fig_no}.png"/><pic:cNvPicPr/></pic:nvPicPr>'
            f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
            f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>'
            '</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r>')
        img_p = para("a3", drawing, '<w:spacing w:before="60" w:after="60"/><w:ind w:firstLine="0"/><w:jc w:val="center"/>')
        block = img_p + self._fig_caption(self.fig_no, ko, en)
        if width == "page":
            self.body.append(self._float_wrap(block) + ANCHOR_P)
        else:
            self.body.append(block)
        return self.fig_no

    # ---- tables
    def table(self, width: str, colw: list, rows: list, ko: str, en: str, note: str) -> int:
        self.tbl_no += 1
        ncol = max(len(r) for r in rows)
        avail = PAGE_W if width == "page" else COL_W
        if not colw:
            colw = [avail // ncol] * ncol
        assert len(colw) == ncol, f"표 {self.tbl_no}: 열 폭 {len(colw)}개 ≠ 열 {ncol}개"
        total = sum(colw)
        sz = 16
        border = '<w:top w:val="single" w:sz="3" w:space="0" w:color="000000"/><w:left w:val="single" w:sz="3" w:space="0" w:color="000000"/>' \
                 '<w:bottom w:val="single" w:sz="3" w:space="0" w:color="000000"/><w:right w:val="single" w:sz="3" w:space="0" w:color="000000"/>'
        tblpr = ('<w:tblPr><w:tblOverlap w:val="never"/>'
                 + f'<w:tblW w:w="{total}" w:type="dxa"/>'
                 + ('<w:jc w:val="center"/>' if width == "page" else '<w:tblInd w:w="28" w:type="dxa"/>')
                 + f'<w:tblBorders>{border}<w:insideH w:val="single" w:sz="3" w:space="0" w:color="000000"/><w:insideV w:val="single" w:sz="3" w:space="0" w:color="000000"/></w:tblBorders>'
                 '<w:tblLayout w:type="fixed"/>'
                 '<w:tblCellMar><w:left w:w="28" w:type="dxa"/><w:right w:w="28" w:type="dxa"/></w:tblCellMar>'
                 '<w:tblLook w:val="0000" w:firstRow="0" w:lastRow="0" w:firstColumn="0" w:lastColumn="0" w:noHBand="0" w:noVBand="0"/>'
                 '</w:tblPr>')
        grid = "<w:tblGrid>" + "".join(f'<w:gridCol w:w="{w}"/>' for w in colw) + "</w:tblGrid>"
        trs = []
        for ri, row in enumerate(rows):
            row = row + [""] * (ncol - len(row))
            tcs = []
            for ci, cell in enumerate(row):
                cell_runs = inline_runs(cell.replace("<br>", "\n"), self.nb, cite_sup=False)
                if ri == 0:
                    cell_runs = [(t, dict(f, b=True)) for t, f in cell_runs]
                # 셀 내 줄바꿈(\n) → <w:br/>
                pieces = []
                for t, f in cell_runs:
                    segs = t.split("\n")
                    for si, seg in enumerate(segs):
                        if si > 0:
                            pieces.append("<w:r><w:br/></w:r>")
                        if seg:
                            pieces.append(runs_xml([(seg, f)], f'<w:sz w:val="{sz}"/><w:szCs w:val="{sz}"/>'))
                jc = "center"
                ppr = f'<w:pStyle w:val="a3"/><w:wordWrap/><w:spacing w:after="0" w:line="240" w:lineRule="auto"/><w:ind w:firstLine="0"/><w:jc w:val="{jc}"/>'
                tcs.append(f'<w:tc><w:tcPr><w:tcW w:w="{colw[ci]}" w:type="dxa"/><w:tcBorders>{border}</w:tcBorders><w:vAlign w:val="center"/></w:tcPr>'
                           f'<w:p><w:pPr>{ppr}</w:pPr>{"".join(pieces)}</w:p></w:tc>')
            trpr = '<w:trPr><w:trHeight w:val="262"/>' + ('<w:tblHeader/>' if ri == 0 else '') + '</w:trPr>'
            trs.append(f"<w:tr>{trpr}{''.join(tcs)}</w:tr>")
        tbl = f"<w:tbl>{tblpr}{grid}{''.join(trs)}</w:tbl>"
        note_x = ""
        if note:
            # 표 주석은 본문 크기(a3 상속 10pt) — 09-14 교수님 코멘트("Fontsize=8인 이유?"): 캡션이 아니라 본문 성격의 주석이므로 본문과 같은 크기
            note_x = para("a3", runs_xml(inline_runs(note, self.nb, cite_sup=False)),
                          '<w:spacing w:before="40" w:after="120" w:line="240" w:lineRule="auto"/><w:ind w:firstLine="0"/><w:jc w:val="left"/>')
        block = self._tbl_caption(self.tbl_no, ko, en) + tbl + (note_x or para("a3", "", '<w:spacing w:after="120"/>'))
        if width == "page":
            self.body.append(self._float_wrap(block) + ANCHOR_P)
        else:
            self.body.append(block)
        return self.tbl_no

    # ---- references
    def references(self, bib: dict):
        for n, key in enumerate(self.nb.order, 1):
            typ, f = bib[key]
            runs = fmt_reference(typ, f)
            inner = f'<w:r><w:rPr><w:w w:val="100"/></w:rPr>{wt(f"[{n}] ")}</w:r>'
            inner += "".join(wr(t, "<w:i/>" if it else "") for t, it in runs)
            self.body.append(para("af5", inner))

    # ---- front page (template first-page floating table)
    def front_page(self, meta: dict):
        border = '<w:top w:val="single" w:sz="3" w:space="0" w:color="000000"/><w:left w:val="single" w:sz="3" w:space="0" w:color="000000"/>' \
                 '<w:bottom w:val="single" w:sz="3" w:space="0" w:color="000000"/><w:right w:val="single" w:sz="3" w:space="0" w:color="000000"/>'
        none = "".join(f'<w:{s} w:val="nil"/>' for s in ("top", "left", "bottom", "right"))
        ps = []
        # 양식 파일(template_ieie_2021.docx)의 첫 줄 라벨 "투고용 논문 2021" 은 원고 내용이 아니라 서식 표시라
        # 09-11 검토 반영으로 출력하지 않는다(학회 예시 원고 example_conference_2page.docx 에도 없음).
        # 되살리려면 아래 줄의 wr("") 를 wr("투고용 논문 2021") 로 바꾼다.
        ps.append(para("af3", '<w:bookmarkStart w:id="0" w:name="_top"/><w:bookmarkEnd w:id="0"/>' + wr("")))
        ps.append(para("a9", wr(meta["title_ko"])))
        ps.append(para("aa", wr(meta["title_en"], '<w:sz w:val="34"/>')))
        ps.append(para("ad", wr("요") + wr("  ") + wr("약")))
        for p in meta["abstract_ko"]:
            ps.append(para("ae", runs_xml(inline_runs(p, self.nb))))
        ps.append(para("Abstract", wr("Abstract")))
        for p in meta["abstract_en"]:
            ps.append(para("ae", runs_xml(inline_runs(p, self.nb))))
        kw = ('<w:r><w:rPr><w:rStyle w:val="Keywords"/><w:spacing w:val="0"/></w:rPr>' + wt("      ") + "</w:r>"
              '<w:r><w:rPr><w:rStyle w:val="Keywords"/><w:spacing w:val="0"/></w:rPr>' + wt("Keywords") + "</w:r>"
              + wr(" :") + wr(" ")
              + '<w:r><w:rPr><w:rStyle w:val="Keywords0"/><w:spacing w:val="0"/></w:rPr>' + wt(meta["keywords"]) + "</w:r>")
        ps.append(para("a3", kw))
        cell = "".join(ps)
        tbl = ('<w:tbl><w:tblPr><w:tblpPr w:bottomFromText="28" w:vertAnchor="text" w:tblpYSpec="top"/>'
               '<w:tblW w:w="9645" w:type="dxa"/>'
               f'<w:tblBorders>{border}</w:tblBorders><w:tblLayout w:type="fixed"/>'
               '<w:tblCellMar><w:left w:w="0" w:type="dxa"/><w:right w:w="0" w:type="dxa"/></w:tblCellMar>'
               '<w:tblLook w:val="0000" w:firstRow="0" w:lastRow="0" w:firstColumn="0" w:lastColumn="0" w:noHBand="0" w:noVBand="0"/>'
               '</w:tblPr><w:tblGrid><w:gridCol w:w="9645"/></w:tblGrid>'
               '<w:tr><w:trPr><w:trHeight w:val="9717"/></w:trPr>'
               f'<w:tc><w:tcPr><w:tcW w:w="9645" w:type="dxa"/><w:tcBorders>{none}</w:tcBorders></w:tcPr>{cell}</w:tc></w:tr>'
               f'<w:tr><w:tc><w:tcPr><w:tcW w:w="9645" w:type="dxa"/><w:tcBorders>{none}</w:tcBorders></w:tcPr>{para("af3", "")}</w:tc></w:tr>'
               '</w:tbl>')
        self.body.append(tbl)


DOC_ROOT = ('<w:document xmlns:wpc="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas" '
            'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" '
            'xmlns:o="urn:schemas-microsoft-com:office:office" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
            'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" '
            'xmlns:v="urn:schemas-microsoft-com:vml" '
            'xmlns:wp14="http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing" '
            'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
            'xmlns:w10="urn:schemas-microsoft-com:office:word" '
            'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
            'xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml" '
            'xmlns:wpg="http://schemas.microsoft.com/office/word/2010/wordprocessingGroup" '
            'xmlns:wpi="http://schemas.microsoft.com/office/word/2010/wordprocessingInk" '
            'xmlns:wne="http://schemas.microsoft.com/office/word/2006/wordml" '
            'xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape" mc:Ignorable="w14 wp14">')

SECT_PR = ('<w:sectPr><w:headerReference w:type="default" r:id="rId10"/><w:footerReference w:type="default" r:id="rId11"/>'
           '<w:footnotePr><w:numFmt w:val="lowerRoman"/><w:numRestart w:val="eachPage"/></w:footnotePr>'
           '<w:endnotePr><w:numFmt w:val="decimal"/></w:endnotePr><w:pgSz w:w="11906" w:h="16838"/>'
           '<w:pgMar w:top="1984" w:right="1135" w:bottom="1138" w:left="1134" w:header="1134" w:footer="571" w:gutter="0"/>'
           '<w:cols w:num="2" w:space="510"/></w:sectPr>')


# ----------------------------------------------------------------------------- markdown emitter
class MdBuilder:
    def __init__(self, nb: Numberer):
        self.nb = nb
        self.out: list[str] = []
        self.fig_no = self.tbl_no = self.eq_no = 0

    def _t(self, text: str) -> str:
        return runs_text(inline_runs(text, self.nb))

    def _caption(self, ko_label: str, en_label: str, no: int, ko: str, en: str) -> str:
        lines = ([f"{ko_label} {no}. {self._t(ko)}"] if ko.strip() else []) + ([f"{en_label} {no}. {self._t(en)}"] if en.strip() else [])
        assert lines, f"{en_label} {no}: 캡션(@cap_en)이 없습니다"
        return "  \n".join(lines) + "\n"

    def front(self, meta):
        o = self.out
        o.append(f"# {meta['title_ko']}\n")
        o.append(f"**{meta['title_en']}**\n")
        o.append("*(대한전자공학회 투고용 논문 양식 2021 — 저자·소속은 투고 시스템/게재용 양식에서 기입)*\n")
        o.append("## 요 약\n")
        for p in meta["abstract_ko"]:
            o.append(self._t(p) + "\n")
        o.append("## Abstract\n")
        for p in meta["abstract_en"]:
            o.append(self._t(p) + "\n")
        o.append(f"**Keywords :** {meta['keywords']}\n")
        o.append("---\n")

    def chapter(self, idx, title):
        if title.strip().upper() == "REFERENCES":
            self.out.append("## REFERENCES\n")
        else:
            self.out.append(f"## {ROMAN[idx]}. {self._t(title)}\n")

    def section(self, idx, title):
        self.out.append(f"### {idx}. {self._t(title)}\n")

    def subsection(self, idx, title):
        self.out.append(f"#### {HANGUL[idx - 1]}. {self._t(title)}\n")

    def paragraph(self, text):
        self.out.append(self._t(text) + "\n")

    def equation(self, latex):
        self.eq_no += 1
        self.out.append(f"$$ {latex} \\qquad ({self.eq_no}) $$\n")

    def figure(self, path, width, scale, ko, en):
        self.fig_no += 1
        rel = os.path.relpath(os.path.join(ROOT, path), IEIE)
        self.out.append(f"![Fig. {self.fig_no}]({rel})\n")
        self.out.append(self._caption("그림", "Fig.", self.fig_no, ko, en))

    def table(self, width, colw, rows, ko, en, note):
        self.tbl_no += 1
        self.out.append(self._caption("표", "Table", self.tbl_no, ko, en))
        ncol = max(len(r) for r in rows)
        lines = []
        for ri, r in enumerate(rows):
            r = [runs_md(inline_runs(c, self.nb, cite_sup=False)) for c in r] + [""] * (ncol - len(r))
            lines.append("| " + " | ".join(r) + " |")
            if ri == 0:
                lines.append("|" + "---|" * ncol)
        self.out.append("\n".join(lines) + "\n")
        if note:
            self.out.append(runs_text(inline_runs(note, self.nb, cite_sup=False)) + "\n")

    def references(self, bib):
        for n, key in enumerate(self.nb.order, 1):
            typ, f = bib[key]
            s = "".join(f"*{t}*" if it else t for t, it in fmt_reference(typ, f))
            self.out.append(f"[{n}] {s}  ")
        self.out.append("")

    def text(self) -> str:
        return "\n".join(self.out)


# ----------------------------------------------------------------------------- assemble
def walk(doc: Doc, sinks: list, bib: dict):
    ch = sec = sub = 0
    for b in doc.blocks:
        k = b["kind"]
        if k == "chapter":
            ch += 1; sec = sub = 0
            for s in sinks:
                s.chapter(ch - 1, b["title"])
            if b["title"].strip().upper() == "REFERENCES":
                for s in sinks:
                    s.references(bib)
        elif k == "section":
            sec += 1; sub = 0
            for s in sinks:
                s.section(sec, b["title"])
        elif k == "subsection":
            sub += 1
            for s in sinks:
                s.subsection(sub, b["title"])
        elif k == "para":
            for s in sinks:
                s.paragraph(b["text"])
        elif k == "eq":
            for s in sinks:
                s.equation(b["latex"])
        elif k == "figure":
            for s in sinks:
                s.figure(b["path"], b["width"], b["scale"], b["cap_ko"], b["cap_en"])
        elif k == "table":
            for s in sinks:
                s.table(b["width"], b["colw"], b["rows"], b["cap_ko"], b["cap_en"], b["note"])


def build_docx(dx: DocxBuilder, template: str, out: str, title: str, sect_pr: str = None, subject: str = ""):
    """sect_pr/subject 는 학술대회 2쪽 빌더(build_ieie_conf_docx.py)가 마지막 섹션 속성·문서 속성을 바꿔 재사용."""
    zin = zipfile.ZipFile(template)
    names = zin.namelist()
    document_xml = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n' + DOC_ROOT + "<w:body>"
                    + "".join(dx.body) + (sect_pr or SECT_PR) + "</w:body></w:document>")
    # rels: 템플릿의 이미지/OLE(rId7/8/9, 미사용)는 버리고 나머지 유지 + 새 그림 추가
    rels_xml = zin.read("word/_rels/document.xml.rels").decode("utf-8")
    rel_root = ET.fromstring(rels_xml)
    NS = "http://schemas.openxmlformats.org/package/2006/relationships"
    keep = []
    for r in rel_root.findall(f"{{{NS}}}Relationship"):
        tgt = r.get("Target")
        if tgt.startswith("media/") or tgt.startswith("embeddings/"):
            continue
        keep.append((r.get("Id"), r.get("Type"), tgt))
    keep += dx.rels
    rels_out = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
                f'<Relationships xmlns="{NS}">' + "".join(f'<Relationship Id="{i}" Type="{t}" Target="{g}"/>' for i, t, g in keep)
                + "</Relationships>")
    ct = zin.read("[Content_Types].xml").decode("utf-8")
    ct = ct.replace('<Default Extension="bin" ContentType="application/vnd.openxmlformats-officedocument.oleObject"/>', "")
    ct = ct.replace('<Default Extension="jpeg" ContentType="image/jpeg"/>', "")
    if 'Extension="png"' not in ct:
        ct = ct.replace("<Default ", '<Default Extension="png" ContentType="image/png"/><Default ', 1)
    core = zin.read("docProps/core.xml").decode("utf-8")
    core = re.sub(r"<dc:title>.*?</dc:title>", f"<dc:title>{_xml(title)}</dc:title>", core)
    core = re.sub(r"<dc:subject>.*?</dc:subject>", f"<dc:subject>{_xml(subject)}</dc:subject>", core)
    core = re.sub(r"<dc:creator>.*?</dc:creator>", "<dc:creator></dc:creator>", core)
    core = re.sub(r"<cp:lastModifiedBy>.*?</cp:lastModifiedBy>", "<cp:lastModifiedBy></cp:lastModifiedBy>", core)

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
        for n in names:
            if n in ("word/document.xml", "word/_rels/document.xml.rels", "[Content_Types].xml", "docProps/core.xml"):
                continue
            if n.startswith("word/media/") or n.startswith("word/embeddings/"):
                continue
            zout.writestr(n, zin.read(n))
        zout.writestr("[Content_Types].xml", ct)
        zout.writestr("docProps/core.xml", core)
        zout.writestr("word/_rels/document.xml.rels", rels_out)
        zout.writestr("word/document.xml", document_xml)
        for part, data in dx.media:
            zout.writestr(part, data)
    zin.close()


def validate(out: str):
    z = zipfile.ZipFile(out)
    assert z.testzip() is None, "zip 손상"
    parts = set(z.namelist())
    for n in parts:
        if n.endswith(".xml") or n.endswith(".rels"):
            minidom.parseString(z.read(n))  # well-formedness
    rels = ET.fromstring(z.read("word/_rels/document.xml.rels"))
    NS = "http://schemas.openxmlformats.org/package/2006/relationships"
    rid2tgt = {r.get("Id"): r.get("Target") for r in rels.findall(f"{{{NS}}}Relationship")}
    doc = z.read("word/document.xml").decode("utf-8")
    used = set(re.findall(r'r:(?:id|embed)="(rId\d+)"', doc))
    missing = used - set(rid2tgt)
    assert not missing, f"document.xml 의 미해결 관계 ID: {missing}"
    for rid in used:
        tgt = "word/" + rid2tgt[rid]
        assert tgt in parts, f"관계 대상 파트 없음: {tgt}"
    ct = z.read("[Content_Types].xml").decode("utf-8")
    assert 'Extension="png"' in ct
    _check_structure(z.read("word/document.xml"))
    # 요소 개수 요약
    n_p = doc.count("<w:p>") + doc.count("<w:p ")
    n_tbl = doc.count("<w:tbl>")
    n_img = doc.count("<w:drawing>")
    n_math = doc.count("<m:oMath>")
    return dict(parts=len(parts), paragraphs=n_p, tables=n_tbl, images=n_img, omath=n_math, size_kb=os.path.getsize(out) // 1024)


W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
# OOXML 스키마의 자식 순서(부분) — Word 가 순서 위반을 "읽을 수 없는 내용" 으로 취급하는 경우가 있어 검사
_ORDER = {
    "pPr": ["pStyle", "keepNext", "keepLines", "pageBreakBefore", "framePr", "widowControl", "numPr", "suppressLineNumbers",
            "pBdr", "shd", "tabs", "suppressAutoHyphens", "kinsoku", "wordWrap", "overflowPunct", "topLinePunct", "autoSpaceDE",
            "autoSpaceDN", "bidi", "adjustRightInd", "snapToGrid", "spacing", "ind", "contextualSpacing", "mirrorIndents",
            "suppressOverlap", "jc", "textDirection", "textAlignment", "textboxTightWrap", "outlineLvl", "divId", "cnfStyle",
            "rPr", "sectPr", "pPrChange"],
    "rPr": ["rStyle", "rFonts", "b", "bCs", "i", "iCs", "caps", "smallCaps", "strike", "dstrike", "outline", "shadow", "emboss",
            "imprint", "noProof", "snapToGrid", "vanish", "webHidden", "color", "spacing", "w", "kern", "position", "sz", "szCs",
            "highlight", "u", "effect", "bdr", "shd", "fitText", "vertAlign", "rtl", "cs", "em", "lang", "eastAsianLayout",
            "specVanish", "oMath"],
    "tblPr": ["tblStyle", "tblpPr", "tblOverlap", "bidiVisual", "tblStyleRowBandSize", "tblStyleColBandSize", "tblW", "jc",
              "tblCellSpacing", "tblInd", "tblBorders", "shd", "tblLayout", "tblCellMar", "tblLook", "tblCaption", "tblDescription"],
    "tcPr": ["cnfStyle", "tcW", "gridSpan", "hMerge", "vMerge", "tcBorders", "shd", "noWrap", "tcMar", "textDirection",
             "tcFitText", "vAlign", "hideMark"],
    "trPr": ["cnfStyle", "divId", "gridBefore", "gridAfter", "wBefore", "wAfter", "cantSplit", "trHeight", "tblHeader",
             "tblCellSpacing", "jc", "hidden"],
    "sectPr": ["headerReference", "footerReference", "footnotePr", "endnotePr", "type", "pgSz", "pgMar", "paperSrc", "pgBorders",
               "lnNumType", "pgNumType", "cols", "formProt", "vAlign", "noEndnote", "titlePg", "textDirection", "bidi",
               "rtlGutter", "docGrid"],
}


def _check_structure(xml_bytes: bytes):
    root = ET.fromstring(xml_bytes)
    body = root.find(W + "body")
    kids = list(body)
    assert kids[-1].tag == W + "sectPr" and kids[-2].tag == W + "p", "본문 끝은 문단 + sectPr 이어야 함"
    for el in root.iter():
        tag = el.tag[len(W):] if el.tag.startswith(W) else None
        if tag in _ORDER:
            order = _ORDER[tag]
            names = [c.tag[len(W):] for c in el if c.tag.startswith(W)]
            idx = [order.index(n) for n in names if n in order]
            assert idx == sorted(idx), f"<w:{tag}> 자식 순서 위반: {names}"
            unknown = [n for n in names if n not in order]
            assert not unknown, f"<w:{tag}> 미등록 자식: {unknown}"
        if tag == "tc":
            assert len(el) and el[-1].tag == W + "p", "셀은 문단으로 끝나야 함"
        if tag == "tbl":
            grid = el.find(W + "tblGrid")
            ncol = len(grid.findall(W + "gridCol"))
            for tr in el.findall(W + "tr"):
                assert len(tr.findall(W + "tc")) == ncol, f"표 열 수 불일치: grid {ncol}"
    # 인접한 두 표 사이에 문단이 있어야 함
    for a, b in zip(kids, kids[1:]):
        assert not (a.tag == W + "tbl" and b.tag == W + "tbl"), "표가 문단 없이 연속됨"


# 서면 심사용(blind) 점검 — 논문지 양식(template_ieie_2021.docx)에는 저자 블록이 없으므로 본문·초록에 신원 단서만 없으면 된다.
# 학술대회판과 소스를 공유하므로 @author_* 디렉티브는 parse_src 가 무시하고, 여기서는 본문에 남은 단서만 잡는다.
BLIND_PATTERNS = (r"\*\*\*", r"[Uu]niversit", r"대학교", r"대학원", r"연구실", r"연구소", r"e-?mail", r"저자", r"우리 (연구|그룹|팀)",
                  r"[Oo]ur (previous|prior|earlier) (work|study|paper)", r"본 (연구실|그룹)", r"\b(Oh|Choh)\b")


def check_blind(doc) -> list:
    """제목·키워드·초록·본문·캡션·표에서 저자·소속 단서를 찾는다. 참고문헌(REFERENCES)은 검사 대상이 아니다 —
    서면 심사 규정이 금지하는 것은 본문의 저자명·소속 표기이고 참고문헌의 저자명은 통상 허용된다(2026-09-09 검수)."""
    texts = [doc.meta.get("title_ko", ""), doc.meta.get("title_en", ""), doc.meta.get("keywords", "")]
    texts += list(doc.meta.get("abstract_ko", [])) + list(doc.meta.get("abstract_en", []))
    for b in doc.blocks:
        if b["kind"] in ("chapter", "section", "subsection", "para"):
            texts.append(b.get("text", b.get("title", "")))
        elif b["kind"] == "figure":
            texts += [b["cap_ko"], b["cap_en"]]
        elif b["kind"] == "table":
            texts += [b["cap_ko"], b["cap_en"], b["note"]] + [c for r in b["rows"] for c in r]
    hits = []
    for t in texts:
        for pat in BLIND_PATTERNS:
            for m in re.finditer(pat, t):
                hits.append((pat, t[max(0, m.start() - 20):m.end() + 20]))
    return hits


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="IEIE 논문지 양식 docx 빌더")
    ap.add_argument("--src", default=SRC, help="소스 .src.md (기본: draft_ieie_v7.src.md — 학술대회 빌더와 공유)")
    ap.add_argument("--out", default=None, help="출력 stem (기본: 소스 이름에서 .src.md 를 뗀 것) → <stem>.md / <stem>.docx")
    ap.add_argument("--no-blind-check", action="store_true", help="서면 심사용 신원 단서 점검 생략")
    args = ap.parse_args(argv)
    src = args.src if os.path.isabs(args.src) else os.path.join(ROOT, args.src)
    stem = args.out or re.sub(r"\.src\.md$", "", os.path.basename(src))
    out_md = os.path.join(os.path.dirname(src), stem + ".md")
    out_docx = os.path.join(os.path.dirname(src), stem + ".docx")

    bib = parse_bib(BIB)
    doc = parse_src(src)
    if not args.no_blind_check:
        hits = check_blind(doc)
        assert not hits, "서면 심사용 신원 단서 발견(저자·소속 등 — 발견 시 탈락 규정): " + "; ".join(f"{p} → …{c}…" for p, c in hits)
    # 두 빌더가 같은 번호 매기기를 공유하도록 하나의 Numberer 사용 — docx 를 먼저 훑으면서 번호 확정
    nb = Numberer(bib)
    dx = DocxBuilder(nb)
    dx.front_page(doc.meta)
    md = MdBuilder(nb)
    md.front(doc.meta)
    walk(doc, [dx, md], bib)
    build_docx(dx, TEMPLATE, out_docx, doc.meta["title_ko"])
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write(md.text())
    info = validate(out_docx)
    print(f"[md ] {os.path.relpath(out_md, ROOT)}  ({len(md.text())} chars)")
    print(f"[docx] {os.path.relpath(out_docx, ROOT)}  {info}")
    print(f"      figures={dx.fig_no} tables={dx.tbl_no} equations={dx.eq_no} references={len(nb.order)}"
          + ("" if args.no_blind_check else "  blind-check: OK(저자·소속 단서 없음)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
