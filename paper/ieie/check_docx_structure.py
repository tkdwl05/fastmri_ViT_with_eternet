#!/usr/bin/env python
"""IEIE 초안 docx 구조 점검기 (렌더러 없는 환경용).

Word/한글을 열 수 없는 컨테이너에서 빌더 산출 docx 의 OOXML 을 표준 라이브러리만으로 뜯어
렌더 없이도 잡을 수 있는 결함(깨진 XML·끊긴 관계 id·표 폭 초과·그림 해상도·캡션 순서·번호 불일치·
잔존 마크다운·자리표시자)을 PASS/WARN/FAIL 로 보고하고, 2단 레이아웃 기준의 대략 쪽수를 추정한다.
부동 표/그림의 실제 위치·줄바꿈·수식 렌더는 구조로 판정할 수 없으므로 Word 에서 직접 확인해야 한다
(체크리스트는 docs/paper_table_conventions.md §4).

사용:  python paper/ieie/check_docx_structure.py            # 두 초안 모두
       python paper/ieie/check_docx_structure.py <docx> [--template <docx>]
"""
import io, math, os, re, sys, zipfile
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IEIE = os.path.join(ROOT, "paper", "ieie")
NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "pic": "http://schemas.openxmlformats.org/drawingml/2006/picture",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
}
W = "{%s}" % NS["w"]; WP = "{%s}" % NS["wp"]; A = "{%s}" % NS["a"]; R = "{%s}" % NS["r"]; M = "{%s}" % NS["m"]
XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"
DEFAULT = [  # (docx, 대응 양식)
    ("draft_ieie_ko_v1.docx", "template_ieie_2021.docx"),
    ("draft_ieie_conf_ko_v1.docx", "example_conference_2page.docx"),
]


class Report:
    def __init__(self, name):
        self.name = name; self.rows = []
    def add(self, level, what, detail=""):
        self.rows.append((level, what, detail))
    def ok(self, what, detail=""): self.add("PASS", what, detail)
    def warn(self, what, detail=""): self.add("WARN", what, detail)
    def fail(self, what, detail=""): self.add("FAIL", what, detail)
    def dump(self):
        n = {k: sum(1 for r in self.rows if r[0] == k) for k in ("PASS", "WARN", "FAIL")}
        print(f"\n=== {self.name}   PASS {n['PASS']} / WARN {n['WARN']} / FAIL {n['FAIL']}")
        for lv, what, det in self.rows:
            print(f"  [{lv}] {what}" + (f" — {det}" if det else ""))
        return n["FAIL"]


def attr(el, name, default=None):
    return el.get(W + name, default) if el is not None else default


def text_of(el):
    return "".join(t.text or "" for t in el.iter(W + "t"))


def para_text(p):
    return text_of(p) + "".join("〔수식〕" for _ in p.iter(M + "oMath"))


def _sect_from(sp, prev=None):
    pg = sp.find("w:pgSz", NS); mar = sp.find("w:pgMar", NS); cols = sp.find("w:cols", NS)
    g = lambda el, k, dflt: int(attr(el, k, dflt)) if attr(el, k, dflt) is not None else dflt
    d = dict(w=g(pg, "w", 11906), h=g(pg, "h", 16838), left=g(mar, "left", 1134), right=g(mar, "right", 1134),
             top=g(mar, "top", 1701), bottom=g(mar, "bottom", 1701),
             ncol=g(cols, "num", 1), space=g(cols, "space", 0))
    d["text_w"] = d["w"] - d["left"] - d["right"]; d["text_h"] = d["h"] - d["top"] - d["bottom"]
    d["col_w"] = (d["text_w"] - d["space"] * (d["ncol"] - 1)) // d["ncol"]
    return d


def sect_info(root):
    """본문 마지막(문서 수준) sectPr."""
    return _sect_from(root.find("w:body/w:sectPr", NS))


def section_map(root):
    """최상위 요소 → 소속 섹션 dict. 문단 수준 sectPr 는 그 문단까지의 앞 내용에 적용된다."""
    body = root.find("w:body", NS)
    els = [el for el in body if el.tag != W + "sectPr"]
    secs = []; cur = []
    for el in els:
        cur.append(el)
        sp = el.find("w:pPr/w:sectPr", NS) if el.tag == W + "p" else None
        if sp is not None:
            secs.append((cur, _sect_from(sp))); cur = []
    secs.append((cur, sect_info(root)))
    return {id(el): sec for group, sec in secs for el in group}


def load_docx(path):
    z = zipfile.ZipFile(path)
    return z, {n: z.read(n) for n in z.namelist()}


def parent_map(root):
    return {c: p for p in root.iter() for c in p}


# ----------------------------------------------------------------------------- 개별 검사
def check_package(rep, z, parts):
    bad = z.testzip()
    rep.fail("zip 무결성", bad) if bad else rep.ok("zip 무결성", f"{len(parts)} parts")
    n_xml = 0
    for n, b in parts.items():
        if n.endswith(".xml") or n.endswith(".rels"):
            try:
                ET.fromstring(b); n_xml += 1
            except ET.ParseError as e:
                rep.fail("XML 파싱", f"{n}: {e}")
    rep.ok("XML well-formed", f"{n_xml} xml/rels parts")
    ct = parts.get("[Content_Types].xml", b"").decode("utf-8", "replace")
    exts = {m.lower() for m in re.findall(r'Extension="([^"]+)"', ct)}
    media = [n for n in parts if n.startswith("word/media/")]
    for mfile in media:
        ext = mfile.rsplit(".", 1)[-1].lower()
        if ext not in exts:
            rep.fail("[Content_Types] 확장자 누락", f"{ext} ({mfile})")
    for must in ("/word/document.xml", "/word/styles.xml"):
        if f'PartName="{must}"' not in ct:
            rep.fail("[Content_Types] Override 누락", must)
    return media


def check_rels(rep, parts, doc_root):
    rels = ET.fromstring(parts["word/_rels/document.xml.rels"])
    rid = {r.get("Id"): (r.get("Target"), r.get("Type").rsplit("/", 1)[-1], r.get("TargetMode")) for r in rels}
    used = set()
    for el in doc_root.iter():
        for k, v in el.attrib.items():
            if k.startswith(R):
                used.add(v)
                if v not in rid:
                    rep.fail("관계 id 미해결", f"{k.split('}')[-1]}={v}")
    missing = []
    for i, (tgt, typ, mode) in rid.items():
        if mode == "External":
            continue
        p = "word/" + tgt if not tgt.startswith("/") else tgt.lstrip("/")
        if p not in parts:
            missing.append(f"{i}->{tgt}")
    rep.fail("관계 target 부재", ", ".join(missing)) if missing else rep.ok("관계(rels) 해결", f"{len(used)} ids used / {len(rid)} defined")
    unused_media = [i for i, (t, ty, mo) in rid.items() if ty == "image" and i not in used]
    if unused_media:
        rep.warn("참조되지 않는 media rel", ", ".join(unused_media))
    return rid


def check_styles(rep, parts, doc_root):
    sty = ET.fromstring(parts["word/styles.xml"])
    ids = {s.get(W + "styleId") for s in sty.iter(W + "style")}
    names = {s.get(W + "styleId"): s.find("w:name", NS).get(W + "val") for s in sty.iter(W + "style")}
    missing = set()
    for tag in ("pStyle", "rStyle", "tblStyle"):
        for el in doc_root.iter(W + tag):
            if el.get(W + "val") not in ids:
                missing.add(el.get(W + "val"))
    rep.fail("정의되지 않은 style id", ", ".join(sorted(missing))) if missing else rep.ok("style id 참조", f"{len(ids)} styles")
    return names


def check_section(rep, sec, tpl_sec):
    same = all(sec[k] == tpl_sec[k] for k in ("w", "h", "left", "right", "top", "bottom", "ncol", "space"))
    msg = f"A4 {sec['w']}x{sec['h']} 여백 L{sec['left']} R{sec['right']} T{sec['top']} B{sec['bottom']} cols {sec['ncol']} gap {sec['space']} → 본문폭 {sec['text_w']} / 단폭 {sec['col_w']} twips"
    rep.ok("섹션 = 양식", msg) if same else rep.fail("섹션 ≠ 양식", msg + f" (양식: {tpl_sec})")


def table_grid(tbl):
    return [int(g.get(W + "w")) for g in tbl.findall("w:tblGrid/w:gridCol", NS)]


def is_float(tbl):
    return tbl.find("w:tblPr/w:tblpPr", NS)


def check_tables(rep, root, sec, smap):
    body = root.find("w:body", NS)
    n_real = 0; problems = 0
    def sec_of(tbl):
        return smap.get(id(tbl), sec)
    def walk(tbl, limit, depth, label):
        nonlocal n_real, problems
        grid = table_grid(tbl); gsum = sum(grid)
        tw = tbl.find("w:tblPr/w:tblW", NS)
        tw_val = int(tw.get(W + "w")) if tw is not None and tw.get(W + "type") == "dxa" else None
        inner_tbls = [t for t in tbl.iter(W + "tbl") if t is not tbl]
        is_wrapper = len(grid) == 1 and (inner_tbls or list(tbl.iter(W + "drawing")))
        is_block = len(grid) == 1 and not is_wrapper and len(tbl.findall("w:tr", NS)) <= 2   # 제목/저자 블록(양식 관례)
        kind = "wrapper" if is_wrapper else ("title-block" if is_block else "table")
        if is_block:
            limit = max(limit, sec_of(tbl)["text_w"])
        if kind == "table":
            n_real += 1
        if gsum > limit + 8:
            rep.fail(f"{label} 폭 초과", f"gridCol 합 {gsum} > 허용 {limit} twips ({kind}, depth {depth})"); problems += 1
        if tw_val is not None and abs(tw_val - gsum) > 8:
            rep.warn(f"{label} tblW≠gridCol 합", f"{tw_val} vs {gsum}")
        rows = tbl.findall("w:tr", NS)
        for ri, tr in enumerate(rows):
            n_cells = 0; tcw = 0
            for tc in tr.findall("w:tc", NS):
                span = tc.find("w:tcPr/w:gridSpan", NS)
                n_cells += int(span.get(W + "val")) if span is not None else 1
                cw = tc.find("w:tcPr/w:tcW", NS)
                if cw is not None and cw.get(W + "type") == "dxa":
                    tcw += int(cw.get(W + "w"))
            if n_cells != len(grid):
                rep.fail(f"{label} 행 {ri} 셀 수 불일치", f"{n_cells} cells vs {len(grid)} gridCols"); problems += 1
            elif tcw and abs(tcw - gsum) > 8 * len(grid):
                rep.warn(f"{label} 행 {ri} tcW 합≠gridCol 합", f"{tcw} vs {gsum}")
        if kind == "table":
            # 관례 검사: 머리행 굵게, 표 안 글자 8pt(sz 16), 테두리 존재
            hdr = rows[0] if rows else None
            hdr_runs = list(hdr.iter(W + "r")) if hdr is not None else []
            bold = all(r.find("w:rPr/w:b", NS) is not None for r in hdr_runs if text_of(r).strip())
            if hdr_runs and not bold:
                rep.warn(f"{label} 머리행 비굵게")
            szs = {s.get(W + "val") for s in tbl.iter(W + "sz")}
            has_border = tbl.find("w:tblPr/w:tblBorders", NS) is not None or any(tc.find("w:tcPr/w:tcBorders", NS) is not None for tc in tbl.iter(W + "tc"))
            rep.ok(f"{label} 구조", f"{len(rows)} 행 × {len(grid)} 열, 폭 {gsum}/{limit}, 글자 sz {sorted(szs) or '상속'}, 머리행 굵게 {bold}, 테두리 {'있음' if has_border else '없음'}")
        # 중첩 표: 셀 폭이 한계
        for tc in tbl.findall("w:tr/w:tc", NS):
            cw = tc.find("w:tcPr/w:tcW", NS)
            lim = int(cw.get(W + "w")) if cw is not None and cw.get(W + "type") == "dxa" else limit
            for t2 in tc.findall("w:tbl", NS):
                walk(t2, lim, depth + 1, label + "›내부표")
    k = 0
    for el in body:
        if el.tag == W + "tbl":
            k += 1
            s_el = smap[id(el)]
            fl = is_float(el)
            if fl is not None and fl.get(W + "horzAnchor") == "margin":
                limit, where = s_el["text_w"], "page-float"
            elif fl is not None:
                limit, where = s_el["col_w"], "col-float"
            else:
                limit, where = s_el["col_w"], f"inline/{s_el['ncol']}단"
            walk(el, limit, 0, f"표블록#{k}({where})")
    rep.ok("표 요약", f"최상위 블록 {k}, 실제 표 {n_real}, 폭/셀 문제 {problems}")
    return n_real


def check_drawings(rep, root, parts, rid, sec, pmap):
    n = 0; low = []
    try:
        from PIL import Image
    except Exception:
        Image = None
    for dr in root.iter(W + "drawing"):
        n += 1
        node = dr.find("wp:inline", NS) or dr.find("wp:anchor", NS)
        ext = node.find("wp:extent", NS)
        cx, cy = int(ext.get("cx")), int(ext.get("cy"))
        w_tw, h_tw = cx / 635.0, cy / 635.0
        # 컨테이너: 부동 wrapper 표 안이면 page 폭 허용
        limit = sec["col_w"]; where = "column"
        p = pmap.get(dr)
        while p is not None:
            if p.tag == W + "tbl":
                fl = is_float(p)
                if fl is not None and fl.get(W + "horzAnchor") == "margin":
                    limit, where = sec["text_w"], "page-float"
                break
            p = pmap.get(p)
        name = node.find("wp:docPr", NS).get("name")
        blip = node.find(".//a:blip", NS)
        emb = blip.get(R + "embed") if blip is not None else None
        tgt = rid.get(emb, (None,))[0]
        dpi = ""
        if Image and tgt and ("word/" + tgt) in parts:
            im = Image.open(io.BytesIO(parts["word/" + tgt])); px = im.size
            eff = px[0] / (cx / 914400.0)
            dpi = f"{px[0]}x{px[1]} px → {eff:.0f} dpi"
            if eff < 300:
                low.append(f"{name} {eff:.0f} dpi")
        flag = "FAIL" if w_tw > limit + 8 else "ok"
        msg = f"{name}: {cx/914400:.2f}×{cy/914400:.2f} in ({w_tw:.0f}×{h_tw:.0f} twips, {where} 허용 {limit}) {dpi}"
        rep.fail("그림 폭 초과", msg) if flag == "FAIL" else rep.ok("그림", msg)
        if h_tw > sec["text_h"]:
            rep.fail("그림 높이 > 본문 높이", msg)
    if low:
        rep.warn("그림 유효 해상도 < 300 dpi", "; ".join(low))
    ids = [d.get("id") for d in root.iter(WP + "docPr")]
    if len(ids) != len(set(ids)):
        rep.fail("docPr id 중복", str(ids))
    return n


def check_equations(rep, root, pmap):
    eqs = list(root.iter(M + "oMath"))
    bad = 0
    for e in eqs:
        p = pmap.get(e)
        while p is not None and p.tag != W + "p":
            p = pmap.get(p)
        if p is None:
            bad += 1; continue
        if not re.search(r"\(\d+\)\s*$", text_of(p)):
            rep.warn("수식 번호 없음", text_of(p)[:40])
    rep.fail("수식이 문단 밖", str(bad)) if bad else rep.ok("수식", f"{len(eqs)} oMath (문단 내, 우측 번호 확인)")
    return len(eqs)


def block_texts(root):
    """본문 순서대로 (kind, text) — 표/그림 wrapper 는 내부 문단까지 펼침."""
    out = []
    body = root.find("w:body", NS)
    def emit(el):
        if el.tag == W + "p":
            out.append(("fig" if el.find(".//w:drawing", NS) is not None else "p", para_text(el).strip()))
        elif el.tag == W + "tbl":
            grid = table_grid(el)
            inner = [t for t in el.iter(W + "tbl") if t is not el]
            if len(grid) == 1 and (inner or el.find(".//w:drawing", NS) is not None):
                for tc in el.findall("w:tr/w:tc", NS):
                    for ch in tc:
                        emit(ch)
            elif len(grid) == 1 and len(el.findall("w:tr", NS)) <= 2:
                out.append(("block", text_of(el).strip()))
            else:
                out.append(("tbl", " | ".join(text_of(tc).strip() for tc in el.findall("w:tr/w:tc", NS))[:80]))
    for el in body:
        emit(el)
    return out


def check_captions_and_refs(rep, blocks, n_tables, n_figs, n_eqs, journal):
    # 표: 캡션이 표 위(ko + en), 그림: 캡션이 그림 아래(ko + en)
    for i, (k, t) in enumerate(blocks):
        if k == "tbl":
            prev = [x for x in blocks[max(0, i - 3):i] if x[0] == "p" and x[1]]
            txt = " ".join(x[1] for x in prev)
            ko = re.search(r"(?<![가-힣])표\s*\d+", txt); en = re.search(r"Table\s*\d+", txt)
            if not ko:
                rep.fail("표 캡션(위) 없음", t[:50])
            elif journal and not en:
                rep.warn("표 영문 캡션 없음", txt[:50])
        if k == "fig":
            nxt = [x for x in blocks[i + 1:i + 4] if x[0] == "p" and x[1]]
            txt = " ".join(x[1] for x in nxt)
            ko = re.search(r"그림\s*\d+", txt); en = re.search(r"Fig\.\s*\d+", txt)
            if not ko:
                rep.fail("그림 캡션(아래) 없음", txt[:50])
            elif journal and not en:
                rep.warn("그림 영문 캡션 없음", txt[:50])
    alltext = "\n".join(t for k, t in blocks)
    refs = {"표": [int(x) for x in re.findall(r"(?<![가-힣])표\s*(\d+)", alltext)],
            "그림": [int(x) for x in re.findall(r"그림\s*(\d+)", alltext)],
            "식": [int(x) for x in re.findall(r"식\s*\((\d+)\)", alltext)]}
    for name, cnt in (("표", n_tables), ("그림", n_figs), ("식", n_eqs)):
        over = sorted({r for r in refs[name] if r > cnt})
        if over:
            rep.fail(f"{name} 번호 참조 > 개수", f"{over} (개수 {cnt})")
        elif cnt:
            rep.ok(f"{name} 참조 번호", f"최대 {max(refs[name]) if refs[name] else 0} ≤ {cnt}")
    left = {"'***' 자리표시자": alltext.count("***"), "[TBD]": alltext.count("TBD"),
            "잔존 마크다운 ** / __": len(re.findall(r"(?<!\*)\*\*(?!\*)|(?<!_)__(?!_)", alltext)),
            "잔존 @지시어": len(re.findall(r"(^|\s)@\w+:", alltext)),
            "잔존 [@cite]": len(re.findall(r"\[@", alltext)),
            "잔존 %% 주석": alltext.count("%%")}
    for k, v in left.items():
        if v:
            (rep.warn if k.startswith("'***'") or k == "[TBD]" else rep.fail)(k, f"{v}건")
    if not any(v for k, v in left.items() if not (k.startswith("'***'") or k == "[TBD]")):
        rep.ok("잔존 마크업 없음")
    return alltext


def check_text_runs(rep, root):
    n_bad = 0
    for t in root.iter(W + "t"):
        s = t.text or ""
        if s != s.strip() and t.get(XML_SPACE) != "preserve":
            n_bad += 1
        if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", s):
            rep.fail("제어문자", repr(s[:30]))
    rep.warn("공백 보존 누락 run", f"{n_bad}") if n_bad else rep.ok("run 공백 보존(xml:space)")
    body = root.find("w:body", NS)
    streak = best = 0
    for el in body:
        if el.tag == W + "p" and not para_text(el).strip() and el.find(".//w:drawing", NS) is None:
            streak += 1; best = max(best, streak)
        else:
            streak = 0
    rep.warn("연속 빈 문단", f"최대 {best}") if best >= 3 else rep.ok("빈 문단 연속 ≤ 2")


def check_fonts(rep, root, tpl_root):
    def fonts(rt):
        s = set()
        for f in rt.iter(W + "rFonts"):
            for k in ("ascii", "hAnsi", "eastAsia"):
                v = f.get(W + k)
                if v: s.add(v)
        return s
    mine, tpl = fonts(root), fonts(tpl_root)
    extra = mine - tpl
    rep.warn("양식에 없는 글꼴", ", ".join(sorted(extra))) if extra else rep.ok("글꼴 = 양식 범위", ", ".join(sorted(mine)))


# ----------------------------------------------------------------------------- 쪽수 추정 (2단)
def estimate_pages(rep, root, sec, styles_xml):
    sty = ET.fromstring(styles_xml)
    st_sz = {}; st_line = {}
    for s in sty.iter(W + "style"):
        sid = s.get(W + "styleId")
        sz = s.find("w:rPr/w:sz", NS); ln = s.find("w:pPr/w:spacing", NS)
        if sz is not None: st_sz[sid] = int(sz.get(W + "val"))
        if ln is not None and ln.get(W + "line"): st_line[sid] = int(ln.get(W + "line"))
    d_line = sty.find("w:docDefaults/w:pPrDefault/w:pPr/w:spacing", NS)
    d_line = int(d_line.get(W + "line")) if d_line is not None and d_line.get(W + "line") else 240
    col_pt = sec["col_w"] / 20.0; page_pt = sec["text_h"] / 20.0
    total = 0.0  # 단(column) 높이 누적 (pt)
    def wlen(s): return sum(1.0 if ord(c) > 0x2E80 else 0.55 for c in s)
    def p_height(p, width_pt):
        pst = p.find("w:pPr/w:pStyle", NS); sid = pst.get(W + "val") if pst is not None else None
        szs = [int(x.get(W + "val")) for x in p.iter(W + "sz")]
        sz = max(szs) if szs else st_sz.get(sid, 20)
        sp = p.find("w:pPr/w:spacing", NS)
        line = int(sp.get(W + "line")) if sp is not None and sp.get(W + "line") else st_line.get(sid, d_line)
        before = int(sp.get(W + "before", 0)) / 20.0 if sp is not None else 0
        after = int(sp.get(W + "after", 0)) / 20.0 if sp is not None else 0
        pt = sz / 2.0; per_line = max(1.0, width_pt / pt)
        txt = para_text(p)
        dr = p.find(".//wp:extent", NS)
        if dr is not None:
            return int(dr.get("cy")) / 12700.0 + before + after
        lines = max(1, math.ceil(wlen(txt) / per_line)) if txt.strip() else 1
        return lines * pt * 1.2 * (line / 240.0) + before + after
    def tbl_height(tbl, width_pt):
        h = 0.0
        for tr in tbl.findall("w:tr", NS):
            cells = tr.findall("w:tc", NS)
            hc = 0.0
            for tc in cells:
                cw = tc.find("w:tcPr/w:tcW", NS)
                wpt = int(cw.get(W + "w")) / 20.0 if cw is not None and cw.get(W + "type") == "dxa" else width_pt / max(1, len(cells))
                hcell = 0.0
                for ch in tc:
                    if ch.tag == W + "p": hcell += p_height(ch, wpt)
                    elif ch.tag == W + "tbl": hcell += tbl_height(ch, wpt)
                hc = max(hc, hcell)
            h += hc
        return h
    body = root.find("w:body", NS)
    for el in body:
        if el.tag == W + "p":
            total += p_height(el, col_pt)
        elif el.tag == W + "tbl":
            fl = is_float(el)
            if fl is not None and fl.get(W + "horzAnchor") == "margin":
                total += 2 * (tbl_height(el, sec["text_w"] / 20.0) + 12)   # 두 단 가로지름 → 단 높이 2배 소비
            else:
                total += tbl_height(el, col_pt) + 6
    pages = total / (sec["ncol"] * page_pt)
    rep.ok("쪽수 추정(휴리스틱 ±15%)", f"단 높이 합 {total:.0f} pt / (단 {sec['ncol']} × 쪽 {page_pt:.0f} pt) ≈ {pages:.1f} 쪽")
    return pages


# ----------------------------------------------------------------------------- main
def run(docx, template):
    rep = Report(os.path.relpath(docx, ROOT))
    z, parts = load_docx(docx)
    media = check_package(rep, z, parts)
    root = ET.fromstring(parts["word/document.xml"])
    pmap = parent_map(root)
    rid = check_rels(rep, parts, root)
    check_styles(rep, parts, root)
    sec = sect_info(root)
    tz, tparts = load_docx(template)
    troot = ET.fromstring(tparts["word/document.xml"])
    check_section(rep, sec, sect_info(troot))
    smap = section_map(root)
    n_secs = len({id(v) for v in smap.values()})
    rep.ok("섹션 수", f"{n_secs} (문단 수준 sectPr 포함)")
    n_tables = check_tables(rep, root, sec, smap)
    n_figs = check_drawings(rep, root, parts, rid, sec, pmap)
    n_eqs = check_equations(rep, root, pmap)
    blocks = block_texts(root)
    journal = "conf" not in os.path.basename(docx)
    check_captions_and_refs(rep, blocks, n_tables, n_figs, n_eqs, journal)
    check_text_runs(rep, root)
    check_fonts(rep, root, troot)
    estimate_pages(rep, root, sec, parts["word/styles.xml"])
    rep.ok("media", f"{len(media)} files, docx {os.path.getsize(docx)/1024:.0f} KB")
    return rep.dump()


if __name__ == "__main__":
    args = sys.argv[1:]
    if args:
        docx = os.path.abspath(args[0])
        tpl = os.path.abspath(args[args.index("--template") + 1]) if "--template" in args else os.path.join(IEIE, DEFAULT[0][1] if "conf" not in docx else DEFAULT[1][1])
        sys.exit(1 if run(docx, tpl) else 0)
    fails = 0
    for d, t in DEFAULT:
        fails += run(os.path.join(IEIE, d), os.path.join(IEIE, t))
    sys.exit(1 if fails else 0)
