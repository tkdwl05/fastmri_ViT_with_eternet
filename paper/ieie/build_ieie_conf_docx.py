#!/usr/bin/env python3
"""
대한전자공학회 학술대회 2쪽 논문 양식(paper/ieie/example_conference_2page.docx, "추계학술대회 투고논문 작성예시") 빌더.

    CUDA_VISIBLE_DEVICES="" python paper/ieie/build_ieie_conf_docx.py

입력  paper/ieie/draft_ieie_v7.src.md  (+ paper/references.bib, paper/figs/*.png) — 학술지 빌더와 **같은 소스**
출력  paper/ieie/draft_ieie_v7_conf.md / .docx   (프로시딩 게재용: @author_ko/@affil_ko/@email/@author_en/@affil_en 포함)
      (09-09 교수님 지시: 서면 심사용(학술지 양식)과 내용 동일, 저자·소속만 차이. 규정 Double Column 1~5쪽. 옛 v1 은 archive/.)

투고용 학술지 빌더(build_ieie_docx.py)의 소스 파서·서지 포맷터·인용 번호·OMML·검증기를 그대로 import 하고,
문서 조립만 작성예시의 직접 서식을 재현한다(외부 패키지 없음, stdlib 만):
  - 1단 섹션: 제목 표(9249 twips, 바탕 18pt bold) → 빈 줄 → 저자 표(9362 twips, 높이 3616: 저자/소속/e-mail/영문제목/영문저자/영문소속)
    → 섹션 나누기(cols 1, space 567)
  - 2단 섹션(continuous, cols 2, space 567): "Abstract"(12pt bold, 가운데) → 영문 초록(9pt, 앞 공백 2칸)
    → 장 제목 "Ⅰ. 서론"(12pt, 가운데, bold 아님) → 절 제목 "2.1 …"(10pt, 왼쪽) → 본문 9pt(앞 공백 2칸, 줄간격 288 auto)
    → 그림(단 폭 inline) + "Fig. N. …" 영문 캡션(9pt 가운데; 09-14 교수님 지시 — @cap_en 없으면 국문 "그림 N.") / 표 캡션 "Table N. …" + 단 폭 표(8pt) + 주석(본문 9pt)
    → 인라인 수식 $…$ 는 본문 크기(9pt)의 인라인 OMML(교수님이 y_c 를 수식 객체로 고친 방식과 동일), display 수식도 9pt
    → "참고문헌"(12pt 가운데) → 스타일 "11"(개요 1) 9pt 내어쓰기 308 "[n] …" (IEIE 영문 서지 포맷은 학술지 빌더와 공유)
  - 글꼴: 작성예시와 동일하게 HY신명조(ascii/eastAsia) 직접 지정, 제목만 바탕
  - 인용은 본문 인라인 "[n]"(위첨자 아님), 첫 등장 순 번호 — 학술지 빌더와 동일 Numberer
분량 추정기(estimate_pages)로 2쪽 초과 여부를 대략 점검한다(실제 배치는 Word/한글에서 확인).
"""
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_ieie_docx import (ROOT, IEIE, BIB, ROMAN, EMU_PER_TWIP, parse_bib, parse_src, Numberer, inline_runs,   # noqa: E402
                             runs_text, runs_md, fmt_reference, wt, _xml, png_size, latex_to_omml, latex_to_plain,
                             build_docx, validate, walk, MdBuilder)

SRC = os.path.join(IEIE, "draft_ieie_v7.src.md")
TEMPLATE = os.path.join(IEIE, "example_conference_2page.docx")
OUT_MD = os.path.join(IEIE, "draft_ieie_v7_conf.md")
OUT_DOCX = os.path.join(IEIE, "draft_ieie_v7_conf.docx")

# 작성예시 sectPr: A4, 여백 상하 1701 / 좌우 1134 → 본문 폭 9638, 2단 간격 567 → 단 폭 4535 (twips)
PAGE_W = 11906 - 1134 - 1134
COL_SPACE = 567
COL_W = (PAGE_W - COL_SPACE) // 2

FONT = '<w:rFonts w:ascii="HY신명조" w:eastAsia="HY신명조" w:hint="eastAsia"/>'   # 런
PFONT = '<w:rFonts w:ascii="HY신명조" w:eastAsia="HY신명조"/>'                    # 문단 기호
TITLE_FONT = '<w:rFonts w:ascii="바탕" w:eastAsia="바탕"/>'
SPACING = '<w:spacing w:line="288" w:lineRule="auto"/>'
PG = ('<w:pgSz w:w="11906" w:h="16838"/>'
      '<w:pgMar w:top="1701" w:right="1134" w:bottom="1701" w:left="1134" w:header="1134" w:footer="1134" w:gutter="0"/>')
SECT_1COL = f'<w:sectPr><w:endnotePr><w:numFmt w:val="decimal"/></w:endnotePr>{PG}<w:cols w:space="{COL_SPACE}"/></w:sectPr>'
SECT_2COL = (f'<w:sectPr><w:endnotePr><w:numFmt w:val="decimal"/></w:endnotePr><w:type w:val="continuous"/>{PG}'
             f'<w:cols w:num="2" w:space="{COL_SPACE}"/></w:sectPr>')
AUTHOR_KEYS = ("author_ko", "affil_ko", "email", "author_en", "affil_en")


# ----------------------------------------------------------------------------- 저수준 XML
def crun(t: str, sz: int = None, b: bool = False, i: bool = False, hl: bool = False, sup: bool = False,
         font: str = FONT, extra: str = "", u: bool = False) -> str:
    """rPr 스키마 순서: rFonts, b, i, (w …), sz, szCs, highlight, u, vertAlign."""
    rpr = (font + ("<w:b/>" if b else "") + ("<w:i/>" if i else "") + extra
           + (f'<w:sz w:val="{sz}"/><w:szCs w:val="{sz}"/>' if sz else "")
           + ('<w:highlight w:val="yellow"/>' if hl else "")
           + ('<w:u w:val="single"/>' if u else "")
           + ('<w:vertAlign w:val="superscript"/>' if sup else ""))
    return f"<w:r><w:rPr>{rpr}</w:rPr>{wt(t)}</w:r>"


def math_rpr(sz: int = None) -> str:
    """인라인 OMML 런에 넣을 w:rPr 자식 — 글자 크기만(글꼴은 settings.xml 의 수식 글꼴 Cambria Math 가 맡는다)."""
    return f'<w:sz w:val="{sz}"/><w:szCs w:val="{sz}"/>' if sz else ""


def cruns(runs: list, sz: int = None, b: bool = False) -> str:
    return "".join(latex_to_omml(t, rpr=math_rpr(sz)) if f.get("math") else
                   crun(t, sz, b or bool(f.get("b")), bool(f.get("i")), bool(f.get("hl")), bool(f.get("sup")),
                        u=bool(f.get("u"))) for t, f in runs)


def cpara(inner: str = "", jc: str = None, style: str = "a3", ind: str = "", spacing: str = SPACING,
          mark: str = PFONT, after_rpr: str = "", before_runs: str = "") -> str:
    """pPr 스키마 순서: pStyle, wordWrap, spacing, ind, jc, rPr(문단 기호), sectPr."""
    ppr = (f'<w:pStyle w:val="{style}"/><w:wordWrap/>{spacing}{ind}'
           + (f'<w:jc w:val="{jc}"/>' if jc else "") + f"<w:rPr>{mark}</w:rPr>{after_rpr}")
    return f"<w:p><w:pPr>{ppr}</w:pPr>{before_runs}{inner}</w:p>"


def blank(sz: int = 18, jc: str = None, b: bool = False) -> str:
    return cpara(jc=jc, mark=PFONT + ("<w:b/>" if b else "") + f'<w:sz w:val="{sz}"/>')


BORDER1 = "".join(f'<w:{s} w:val="single" w:sz="3" w:space="0" w:color="000000"/>' for s in ("top", "left", "bottom", "right"))
BORDER0 = "".join(f'<w:{s} w:val="none" w:sz="2" w:space="0" w:color="000000"/>' for s in ("top", "left", "bottom", "right"))
CELLMAR4 = "".join(f'<w:{s} w:w="28" w:type="dxa"/>' for s in ("top", "left", "bottom", "right"))
TBLLOOK = '<w:tblLook w:val="0000" w:firstRow="0" w:lastRow="0" w:firstColumn="0" w:lastColumn="0" w:noHBand="0" w:noVBand="0"/>'


def one_cell_table(width: int, height: int, paras: list) -> str:
    """작성예시의 제목/저자 표: 표 테두리는 있으나 셀 테두리 none → 보이지 않는 1칸 표, 세로 가운데."""
    return (f'<w:tbl><w:tblPr><w:tblOverlap w:val="never"/><w:tblW w:w="{width}" w:type="dxa"/><w:tblInd w:w="28" w:type="dxa"/>'
            f'<w:tblBorders>{BORDER1}</w:tblBorders><w:tblLayout w:type="fixed"/><w:tblCellMar>{CELLMAR4}</w:tblCellMar>{TBLLOOK}</w:tblPr>'
            f'<w:tblGrid><w:gridCol w:w="{width}"/></w:tblGrid>'
            f'<w:tr><w:trPr><w:trHeight w:val="{height}"/></w:trPr>'
            f'<w:tc><w:tcPr><w:tcW w:w="{width}" w:type="dxa"/><w:tcBorders>{BORDER0}</w:tcBorders><w:vAlign w:val="center"/></w:tcPr>'
            f'{"".join(paras)}</w:tc></w:tr></w:tbl>')


# ----------------------------------------------------------------------------- 분량 추정 (pt)
COL_PT = COL_W / 20.0          # 단 폭 226.75pt
PAGE_H_PT = (16838 - 1701 - 1701) / 20.0   # 671.8pt


def _weighted_len(text: str) -> float:
    return sum(1.0 if ord(c) > 0x2E80 else 0.55 for c in text)


def text_height(text: str, sz: int, width_pt: float = COL_PT, line_mult: float = 1.2) -> float:
    """HY신명조 9pt 기준 한 줄 ≈ 폭/글자크기 글자, 줄 높이 ≈ 크기×1.2(글꼴)×line_mult(288 auto)."""
    pt = sz / 2.0
    per_line = max(1.0, width_pt / pt)
    lines = max(1, math.ceil(_weighted_len(text) / per_line))
    return lines * pt * 1.2 * line_mult


class DocxConf:
    def __init__(self, nb: Numberer):
        self.nb = nb
        self.body: list[str] = []
        self.media: list[tuple] = []
        self.rels: list[tuple] = []
        self.next_rid = 20
        self.docpr_id = 100
        self.fig_no = self.tbl_no = self.eq_no = 0
        self.ch = 0
        self._last = None
        self.h_front = 0.0     # 1단 제목 블록 높이(pt) — 1쪽 두 단의 높이를 이만큼 깎는다
        self.h_body = 0.0      # 2단 본문 누적 높이(pt)

    # ---- front (1단) + Abstract
    def front(self, meta: dict, authors: dict):
        b = self.body
        b.append("<w:p/>")
        self.h_front += 19
        title_p = cpara(crun(meta["title_ko"], sz=36, b=True, font=TITLE_FONT, extra='<w:w w:val="95"/>'), jc="center",
                        before_runs='<w:bookmarkStart w:id="0" w:name="_top"/><w:bookmarkEnd w:id="0"/>')
        b.append(one_cell_table(9249, 936, [title_p]))
        self.h_front += max(936 / 20.0, text_height(meta["title_ko"], 36, PAGE_W / 20.0) + 3)
        b.append(cpara())
        self.h_front += 14.4
        ap = [blank(20, "center", b=True),
              cpara(crun(authors["author_ko"], b=True), jc="center"),
              cpara(crun(authors["affil_ko"], b=True), jc="center"),
              cpara(crun(authors["email"], b=True), jc="center"),
              blank(20, "center", b=True),
              cpara(crun(meta["title_en"], sz=24, b=True), jc="center"),
              blank(24, "center", b=True),
              cpara(crun(authors["author_en"], b=True), jc="center"),
              cpara(crun(authors["affil_en"], b=True), jc="center")]
        b.append(one_cell_table(9362, 3616, ap))
        self.h_front += max(3616 / 20.0, 7 * 14.4 + text_height(meta["title_en"], 24, PAGE_W / 20.0) + 3)
        b.append(cpara(after_rpr=SECT_1COL))       # 1단 섹션 끝
        self.h_front += 14.4
        # 2단 섹션 시작
        b.append(blank(24, "center", b=True)); self.h_body += 17.3
        b.append(cpara(crun("Abstract", sz=24, b=True), jc="center")); self.h_body += 17.3
        b.append(cpara(jc="center")); self.h_body += 14.4
        for p in meta.get("abstract_en", []):
            b.append(cpara(cruns(inline_runs("  " + p, self.nb, cite_sup=False), sz=18)))
            self.h_body += text_height("  " + p, 18)
        b.append(blank(18)); self.h_body += 13
        self._last = "front"

    # ---- headings / text
    def chapter(self, idx: int, title: str):
        self.ch = idx + 1
        if title.strip().upper() == "REFERENCES":
            self.body.append(blank(18)); self.h_body += 13
            self.body.append(cpara(crun("참고문헌 ", sz=24), jc="center")); self.h_body += 17.3
            self._last = "chapter"
            return
        if idx > 0:
            self.body.append(blank(18)); self.h_body += 13
        self.body.append(cpara(crun(f"{ROMAN[idx]}. {title.strip()}", sz=24) + crun(" ", sz=18), jc="center")); self.h_body += 17.3
        self.body.append(blank(18)); self.h_body += 13
        self._last = "chapter"

    def section(self, idx: int, title: str):
        if self._last not in ("chapter", "figure", "table"):
            self.body.append(blank(18)); self.h_body += 13
        self.body.append(cpara(cruns(inline_runs(f"{self.ch}.{idx} {title.strip()}", self.nb, cite_sup=False))))
        self.h_body += 14.4
        self._last = "section"

    def subsection(self, idx: int, title: str):
        self.body.append(cpara(cruns(inline_runs(f"({idx}) {title.strip()}", self.nb, cite_sup=False), sz=18)))
        self.h_body += 13
        self._last = "subsection"

    def paragraph(self, text: str):
        self.body.append(cpara(cruns(inline_runs("  " + text, self.nb, cite_sup=False), sz=18)))
        self.h_body += text_height("  " + text, 18)
        self._last = "para"

    def equation(self, latex: str) -> int:
        self.eq_no += 1
        # display 수식도 본문 9pt(sz 18) — 09-14 이전엔 docDefaults 10pt 로 본문보다 크게 렌더됐음(의도적 정정)
        inner = latex_to_omml(latex, rpr=math_rpr(18)) + crun(f"  ({self.eq_no})", sz=18)
        self.body.append(cpara(inner, jc="center", mark=PFONT + '<w:sz w:val="18"/>'))
        self.h_body += 26
        self._last = "eq"
        return self.eq_no

    # ---- figure / table
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
            '<w:r><w:rPr>' + FONT + '<w:noProof/></w:rPr><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0">'
            f'<wp:extent cx="{cx}" cy="{cy}"/><wp:effectExtent l="0" t="0" r="0" b="0"/>'
            f'<wp:docPr id="{self.docpr_id}" name="그림 {self.fig_no}"/>'
            '<wp:cNvGraphicFramePr><a:graphicFrameLocks xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" noChangeAspect="1"/></wp:cNvGraphicFramePr>'
            '<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
            '<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
            '<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
            f'<pic:nvPicPr><pic:cNvPr id="0" name="fig{self.fig_no}.png"/><pic:cNvPicPr/></pic:nvPicPr>'
            f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
            f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>'
            '</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r>')
        assert width != "page", "2단 본문에서 page 폭 그림은 지원하지 않음(단 폭 col 만)"
        self.body.append(cpara(drawing, jc="center", mark=PFONT + '<w:sz w:val="18"/>'))
        cap = f"Fig. {self.fig_no}. {en}" if en.strip() else f"그림 {self.fig_no}. {ko}"   # 09-14: 영문 캡션(@cap_en) 우선
        self.body.append(cpara(cruns(inline_runs(cap, self.nb, cite_sup=False), sz=18), jc="center"))
        self.body.append(blank(18))
        self.h_body += h_tw / 20.0 + 4 + text_height(cap, 18) + 13
        self._last = "figure"
        return self.fig_no

    def table(self, width: str, colw: list, rows: list, ko: str, en: str, note: str) -> int:
        self.tbl_no += 1
        assert width != "page", "2단 본문에서 page 폭 표는 지원하지 않음(단 폭 col 만)"
        ncol = max(len(r) for r in rows)
        if not colw:
            colw = [COL_W // ncol] * ncol
        assert len(colw) == ncol, f"표 {self.tbl_no}: 열 폭 {len(colw)}개 ≠ 열 {ncol}개"
        total = sum(colw)
        assert total <= COL_W, f"표 {self.tbl_no}: 열 폭 합 {total} > 단 폭 {COL_W}"
        sz = 16
        tblpr = (f'<w:tblPr><w:tblOverlap w:val="never"/><w:tblW w:w="{total}" w:type="dxa"/><w:tblInd w:w="28" w:type="dxa"/>'
                 f'<w:tblBorders>{BORDER1}<w:insideH w:val="single" w:sz="3" w:space="0" w:color="000000"/>'
                 '<w:insideV w:val="single" w:sz="3" w:space="0" w:color="000000"/></w:tblBorders>'
                 '<w:tblLayout w:type="fixed"/><w:tblCellMar><w:left w:w="28" w:type="dxa"/><w:right w:w="28" w:type="dxa"/></w:tblCellMar>'
                 f'{TBLLOOK}</w:tblPr>')
        grid = "<w:tblGrid>" + "".join(f'<w:gridCol w:w="{w}"/>' for w in colw) + "</w:tblGrid>"
        trs = []
        for ri, row in enumerate(rows):
            row = row + [""] * (ncol - len(row))
            tcs = []
            for ci, cell in enumerate(row):
                runs = inline_runs(cell.replace("<br>", "\n"), self.nb, cite_sup=False)
                pieces = []
                for t, f in runs:
                    for si, seg in enumerate(t.split("\n")):
                        if si > 0:
                            pieces.append("<w:r><w:br/></w:r>")
                        if seg:
                            pieces.append(cruns([(seg, f)], sz=sz, b=(ri == 0)))
                ppr = ('<w:pStyle w:val="a3"/><w:wordWrap/><w:spacing w:after="0" w:line="240" w:lineRule="auto"/>'
                       '<w:jc w:val="center"/><w:rPr>' + PFONT + f'<w:sz w:val="{sz}"/></w:rPr>')
                tcs.append(f'<w:tc><w:tcPr><w:tcW w:w="{colw[ci]}" w:type="dxa"/><w:tcBorders>{BORDER1}</w:tcBorders><w:vAlign w:val="center"/></w:tcPr>'
                           f'<w:p><w:pPr>{ppr}</w:pPr>{"".join(pieces)}</w:p></w:tc>')
            trpr = '<w:trPr><w:trHeight w:val="250"/>' + ('<w:tblHeader/>' if ri == 0 else '') + '</w:trPr>'
            trs.append(f"<w:tr>{trpr}{''.join(tcs)}</w:tr>")
        cap = f"Table {self.tbl_no}. {en}" if en.strip() else f"표 {self.tbl_no}. {ko}"   # 09-14: 영문 캡션(@cap_en) 우선
        self.body.append(cpara(cruns(inline_runs(cap, self.nb, cite_sup=False), sz=18), jc="center"))
        self.body.append(f"<w:tbl>{tblpr}{grid}{''.join(trs)}</w:tbl>")
        if note:
            # 표 주석은 본문 크기(9pt) — 09-14 교수님 코멘트("Fontsize=8인 이유?"); 표 셀만 8pt
            self.body.append(cpara(cruns(inline_runs(note, self.nb, cite_sup=False), sz=18)))
        self.body.append(blank(18))
        self.h_body += text_height(cap, 18) + len(rows) * 13.5 + (text_height(note, 18) if note else 0) + 13
        self._last = "table"
        return self.tbl_no

    # ---- references
    def references(self, bib: dict):
        for n, key in enumerate(self.nb.order, 1):
            typ, f = bib[key]
            runs = fmt_reference(typ, f)
            inner = crun(f"[{n}] ", sz=18) + "".join(crun(t, sz=18, i=it) for t, it in runs)
            self.body.append(cpara(inner, style="11", ind='<w:ind w:left="308" w:hanging="308"/>'))
            self.h_body += text_height(f"[{n}] " + "".join(t for t, _ in runs), 18, COL_PT - 308 / 20.0)
        self.body.append(cpara(style="11", ind='<w:ind w:left="308" w:hanging="308"/>', mark=PFONT + '<w:sz w:val="18"/>'))

    def estimate_pages(self) -> tuple:
        """1쪽 = 제목 블록 아래 두 단, 2쪽부터 = 온전한 두 단. (단 높이 합으로 본문 높이를 나눈 근사)"""
        page1 = 2 * max(0.0, PAGE_H_PT - self.h_front)
        rest = max(0.0, self.h_body - page1)
        pages = 1 + rest / (2 * PAGE_H_PT)
        return pages, self.h_front, self.h_body, page1


class MdConf(MdBuilder):
    def __init__(self, nb: Numberer, authors: dict):
        super().__init__(nb)
        self.authors = authors
        self.ch = 0

    def front(self, meta):
        o = self.out
        o.append(f"# {meta['title_ko']}\n")
        o.append(f"{self.authors['author_ko']} ({self.authors['affil_ko']}, {self.authors['email']})\n")
        o.append(f"**{meta['title_en']}**\n")
        o.append(f"{self.authors['author_en']} ({self.authors['affil_en']})\n")
        o.append("*(대한전자공학회 학술대회 2쪽 양식 — 작성예시 example_conference_2page.docx 기준; 저자·소속은 \"***\" 자리표시자)*\n")
        o.append("## Abstract\n")
        for p in meta.get("abstract_en", []):
            o.append(self._t(p) + "\n")
        o.append("---\n")

    def _t(self, text):
        return runs_text(inline_runs(text, self.nb, cite_sup=False))

    def chapter(self, idx, title):
        self.ch = idx + 1
        if title.strip().upper() == "REFERENCES":
            self.out.append("## 참고문헌\n")
        else:
            self.out.append(f"## {ROMAN[idx]}. {self._t(title)}\n")

    def section(self, idx, title):
        self.out.append(f"### {self.ch}.{idx} {self._t(title)}\n")

    def subsection(self, idx, title):
        self.out.append(f"#### ({idx}) {self._t(title)}\n")

    def figure(self, path, width, scale, ko, en):
        self.fig_no += 1
        rel = os.path.relpath(os.path.join(ROOT, path), IEIE)
        self.out.append(f"![Fig. {self.fig_no}]({rel})\n")
        self.out.append((f"Fig. {self.fig_no}. {self._t(en)}" if en.strip() else f"그림 {self.fig_no}. {self._t(ko)}") + "\n")

    def table(self, width, colw, rows, ko, en, note):
        self.tbl_no += 1
        self.out.append((f"Table {self.tbl_no}. {self._t(en)}" if en.strip() else f"표 {self.tbl_no}. {self._t(ko)}") + "\n")
        ncol = max(len(r) for r in rows)
        lines = []
        for ri, r in enumerate(rows):
            r = [runs_md(inline_runs(c, self.nb, cite_sup=False)) for c in r] + [""] * (ncol - len(r))
            lines.append("| " + " | ".join(r) + " |")
            if ri == 0:
                lines.append("|" + "---|" * ncol)
        self.out.append("\n".join(lines) + "\n")
        if note:
            self.out.append(self._t(note) + "\n")


def parse_authors(path: str) -> dict:
    """@author_ko/@affil_ko/@email/@author_en/@affil_en — parse_src 가 무시하는 앞머리 디렉티브."""
    out = {k: "***" for k in AUTHOR_KEYS}
    for ln in open(path, encoding="utf-8"):
        m = re.match(r"@(\w+):\s*(.*)$", ln.rstrip("\n"))
        if m and m.group(1) in AUTHOR_KEYS:
            out[m.group(1)] = m.group(2).strip()
    return out


MAX_PAGES = 5.0   # 2026 추계학술대회 프로시딩 게재용 규정: Double Column 1페이지 이상 5페이지 이내


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="IEIE 학술대회 양식 docx 빌더(프로시딩 게재용 — 저자·소속 포함)")
    ap.add_argument("--src", default=SRC, help="소스 .src.md (학술지판과 동일 소스를 공유 — 09-09 교수님: 두 파일 내용 동일, 저자만 차이)")
    ap.add_argument("--out", default=None, help="출력 stem (기본: 소스 이름에서 .src.md 를 뗀 것 + _conf) → <stem>.md / <stem>.docx")
    args = ap.parse_args(argv)
    src = args.src if os.path.isabs(args.src) else os.path.join(ROOT, args.src)
    stem = args.out or re.sub(r"\.src\.md$", "", os.path.basename(src)) + "_conf"
    out_md = os.path.join(os.path.dirname(src), stem + ".md")
    out_docx = os.path.join(os.path.dirname(src), stem + ".docx")

    bib = parse_bib(BIB)
    doc = parse_src(src)
    authors = parse_authors(src)
    nb = Numberer(bib)
    dx = DocxConf(nb)
    dx.front(doc.meta, authors)
    md = MdConf(nb, authors)
    md.front(doc.meta)
    walk(doc, [dx, md], bib)
    build_docx(dx, TEMPLATE, out_docx, doc.meta["title_ko"], sect_pr=SECT_2COL, subject="")
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write(md.text())
    info = validate(out_docx)
    pages, hf, hb, p1 = dx.estimate_pages()
    print(f"[md ] {os.path.relpath(out_md, ROOT)}  ({len(md.text())} chars)")
    print(f"[docx] {os.path.relpath(out_docx, ROOT)}  {info}")
    print(f"      figures={dx.fig_no} tables={dx.tbl_no} equations={dx.eq_no} references={len(nb.order)}")
    print(f"      분량 추정: 제목블록 {hf:.0f}pt, 2단 본문 {hb:.0f}pt (1쪽 수용 {p1:.0f}pt, 이후 쪽당 {2 * PAGE_H_PT:.0f}pt) → 약 {pages:.2f} 쪽"
          + (f"  ⚠ {MAX_PAGES:.0f}쪽 초과 가능 — 본문을 줄일 것" if pages > MAX_PAGES else f"  (규정 1~{MAX_PAGES:.0f}쪽)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
