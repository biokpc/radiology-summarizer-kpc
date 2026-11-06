import os
import sys
import re
import json
import base64
import types as _py_types
import streamlit as st
import streamlit.components.v1 as components
import pdfkit

# =====================================================================
# [핵심] PyMuPDF 부가 플러그인 충돌 원천 차단
# =====================================================================
_dummy_layout = _py_types.ModuleType("pymupdf.layout")
_dummy_layout.activate = lambda *args, **kwargs: None
_dummy_layout.__path__ = []
sys.modules["pymupdf.layout"] = _dummy_layout
sys.modules["pymupdf_layout"] = _dummy_layout
sys.modules["pymupdf4llm"] = _py_types.ModuleType("pymupdf4llm")

import fitz  # PyMuPDF
from google import genai
from google.genai import types
from pydantic import BaseModel, Field


# =====================================================================
# 1. 설정 및 API 키 관리
# =====================================================================
CONFIG_PATH = os.path.expanduser("~/.radiology_pdf_summarizer_cloud.json")

def load_saved_key() -> str:
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return str(st.secrets["GEMINI_API_KEY"]).strip()
    except Exception:
        pass
    if "key" in st.query_params and st.query_params["key"]:
        return str(st.query_params["key"]).strip()
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f).get("api_key", "").strip()
        except Exception:
            pass
    return ""

def save_api_key_local(api_key: str):
    clean_key = api_key.strip()
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump({"api_key": clean_key}, f)
    except Exception:
        pass
    try:
        st.query_params["key"] = clean_key
    except Exception:
        pass


# =====================================================================
# 2. Gemini API 출력 JSON 스키마
# =====================================================================
class TableData(BaseModel):
    title: str = Field(description="표 제목 (한글 및 영문 병기)")
    headers: list[str] = Field(description="표의 열 제목 리스트")
    rows: list[list[str]] = Field(description="표의 행 데이터 리스트")

class FigureSummary(BaseModel):
    figure_number: str = Field(description="Figure 번호만 깔끔하게 작성 (예: 'Figure 1', 'Figure 6', 'Figure 10')")
    case_diagnosis: str = Field(
        description="이 Figure 전체가 보여주는 핵심 증례 진단명 또는 핵심 주제 1개. 반드시 '한글명 (영문명)' 형태로 작성"
    )
    image_ids: list[str] = Field(description="해당 Figure에 매칭되는 추출 이미지 ID 리스트 (예: ['p3_fig1'])")
    descriptions: list[str] = Field(
        description="하위 항목별 설명. 반드시 '[Figure 번호] → [시퀀스/검사명/모식도] → [화살표·원·별표 지시 해부학 위치(영문 병기 필수) 및 핵심 소견] → [의미/진단]' 구조로 상세히 작성"
    )

class PaperSummaryResult(BaseModel):
    paper_title_ko: str = Field(description="논문 제목 (한글 번역 및 원문 병기)")
    definition_and_clinical: list[str] = Field(description="질환 정의 및 임상적 특징 요약 불릿 포인트 (해부학/질환명 영문 병기)")
    key_imaging_features: list[str] = Field(description="핵심 영상 소견 (Key Imaging Features) 불릿 포인트 (해부학/영상용어 영문 병기)")
    differential_diagnosis: list[str] = Field(description="감별진단 (DDx) 핵심 요약 불릿 포인트 (해부학/질환명 영문 병기)")
    pitfalls_and_management: list[str] = Field(description="판독 시 주의점 (Pitfalls) 및 치료 방침 불릿 포인트")
    tables: list[TableData] = Field(description="본문 내 주요 표 번역 또는 감별진단 비교 요약 표")
    figures: list[FigureSummary] = Field(description="본문 내 모든 Figure 캡션 및 소견 전수 정리")


def build_fallback_model_chain(client: genai.Client, preferred_tier: str) -> list[str]:
    try:
        all_models = [
            m.name.replace("models/", "")
            for m in client.models.list()
            if "gemini" in m.name and not any(x in m.name for x in ["vision", "tts", "image", "embedding", "aqa", "audio"])
        ]
    except Exception:
        all_models = []

    pro_list = [m for m in all_models if "pro" in m]
    flash_regular = [m for m in all_models if "flash" in m and "lite" not in m and "8b" not in m]
    flash_lite = [m for m in all_models if "flash" in m and ("lite" in m or "8b" in m)]

    chain = []
    if preferred_tier == "pro":
        chain.extend(["gemini-3.8-pro", "gemini-3.1-pro-preview"] + pro_list)
    else:
        chain.extend(["gemini-3.8-flash", "gemini-3.0-flash", "gemini-2.5-flash"] + flash_regular)

    chain.extend(["gemini-3.8-flash", "gemini-3.0-flash", "gemini-2.5-flash"] + flash_regular)
    chain.extend(["gemini-3.8-flash-lite", "gemini-2.5-flash-lite", "gemini-2.0-flash-lite", "gemini-2.0-flash"] + flash_lite)

    final_chain = []
    for m in chain:
        if m not in final_chain and (not all_models or m in all_models):
            final_chain.append(m)

    return final_chain if final_chain else ["gemini-3.8-flash", "gemini-2.5-flash-lite"]


# =====================================================================
# 3. 모식도·벡터·표·CT 통합 Figure 영역 정밀 캡처 엔진
# =====================================================================
FIG_CAP_PATTERN = re.compile(
    r"^\s*Fig(?:ure|\.)?\s*(?:(\d+)\s*[\.\:]|[A-Z][^\n]{5,}\([A-Z]\))",
    re.IGNORECASE | re.DOTALL,
)
TABLE_BARRIER_PATTERN = re.compile(
    r"^\s*(?:Table\s+\d+\s*[\.\:]|Note\.—|Note\.\s*-|Source\.—|Source\.\s*-|\*[A-Z])",
    re.IGNORECASE,
)
SECTION_HEADER_PATTERN = re.compile(
    r"^\s*(?:TEACHING POINTS|Introduction|History|Performance|Literature Review|Safety|Extracolonic Findings|Technique|Bowel Preparation|Colonic Distention|Interpretation and Reporting|Current Concepts|Polypectomy Strategy|Barriers to Implementation|Future Directions|Opportunistic Screening|Conclusion|References|Abstract|Materials and Methods|Results|Discussion)\s*$",
    re.IGNORECASE,
)

def extract_figures_from_page_render(page: fitz.Page, page_num: int, xref_counts: dict, next_auto_fig: int) -> tuple[list[dict], int]:
    pw, ph = page.rect.width, page.rect.height
    top_limit = ph * 0.055
    bot_limit = ph * 0.935

    raw_blocks = page.get_text("blocks")
    text_blocks = []
    fig_captions = []
    table_barriers = []
    body_blocks = []

    for b in raw_blocks:
        if b[6] != 0:
            continue
        r = fitz.Rect(b[:4])
        txt = b[4].strip()
        if not txt:
            continue

        if r.y1 <= ph * 0.062 or r.y0 >= ph * 0.932:
            continue
        if any(h in txt for h in ["radiographics.rsna.org", "October Special Issue", "Volume 46 Number", "This copy is for personal use"]):
            continue

        m_fig = FIG_CAP_PATTERN.match(txt)
        if m_fig:
            fnum = int(m_fig.group(1)) if m_fig.group(1) else next_auto_fig
            next_auto_fig = max(next_auto_fig, fnum + 1)
            fig_captions.append({"fig_num": fnum, "rect": r, "text": txt})
            continue

        if TABLE_BARRIER_PATTERN.match(txt):
            table_barriers.append(r)
            continue

        if len(txt) >= 135 or SECTION_HEADER_PATTERN.match(txt):
            body_blocks.append(r)
        else:
            text_blocks.append({"rect": r, "text": txt})

    if not fig_captions:
        return [], next_auto_fig

    fig_captions.sort(key=lambda c: c["rect"].y0)

    last_cap_y1 = max(c["rect"].y1 for c in fig_captions)
    for tb in list(text_blocks):
        r = tb["rect"]
        txt = tb["text"]
        if len(txt) >= 48 and (txt.endswith(".") or ")" in txt):
            if r.y0 >= last_cap_y1 - 4:
                body_blocks.append(r)
                text_blocks.remove(tb)
                continue
            for bb in body_blocks:
                if abs(r.x0 - bb.x0) < 25 and (abs(r.y0 - bb.y1) < 18 or abs(bb.y0 - r.y1) < 18):
                    body_blocks.append(r)
                    text_blocks.remove(tb)
                    break

    if table_barriers:
        changed = True
        while changed:
            changed = False
            tb_union = fitz.Rect(table_barriers[0])
            for tr in table_barriers[1:]:
                tb_union |= tr
            for tb in list(text_blocks):
                r = tb["rect"]
                if (r.y1 <= tb_union.y1 + 15 and r.y0 >= top_limit and tb_union.y0 < ph * 0.45 and r.y1 <= tb_union.y0 + 5) or (
                    r.y0 >= tb_union.y0 - 10 and r.y1 <= tb_union.y1 + 10
                ):
                    table_barriers.append(r)
                    text_blocks.remove(tb)
                    changed = True

    img_infos = page.get_image_info(xrefs=True)
    valid_img_rects = []
    for info in img_infos:
        xref = info.get("xref", 0)
        if xref and xref_counts.get(xref, 0) > 2:
            continue
        ir = fitz.Rect(info["bbox"]) & page.rect
        if ir.is_empty or ir.width < 22 or ir.height < 22:
            continue
        if ir.y1 <= top_limit or ir.y0 >= bot_limit:
            continue
        if ir.width > pw * 0.96 and ir.height > ph * 0.92:
            continue
        valid_img_rects.append(ir)

    valid_draw_rects = []
    for d in page.get_drawings():
        dr = fitz.Rect(d.get("rect", (0, 0, 0, 0))) & page.rect
        if dr.is_empty:
            continue
        if dr.width < 3 and dr.height < 3:
            continue
        if dr.y1 <= top_limit or dr.y0 >= bot_limit:
            continue
        if dr.width > pw * 0.85 and dr.height < 4:
            continue
        if dr.width > pw * 0.94 and dr.height > ph * 0.90:
            continue
        if any(dr.intersects(bb) for bb in body_blocks + table_barriers):
            continue
        valid_draw_rects.append(dr)

    K = len(fig_captions)
    assigned_imgs = {i: [] for i in range(K)}
    for ir in valid_img_rects:
        cy = (ir.y0 + ir.y1) / 2.0
        assigned = False
        for i in range(K - 1):
            if cy < fig_captions[i]["rect"].y1:
                assigned_imgs[i].append(ir)
                assigned = True
                break
        if not assigned:
            assigned_imgs[K - 1].append(ir)

    fig_zones = []
    for i, cap in enumerate(fig_captions):
        cap_r = cap["rect"]
        imgs = assigned_imgs[i]

        if imgs:
            core_rect = fitz.Rect(imgs[0])
            for ir in imgs[1:]:
                core_rect |= ir
        else:
            above_draws = [dr for dr in valid_draw_rects if dr.y1 <= cap_r.y0 + 5]
            if not above_draws:
                continue
            core_rect = fitz.Rect(above_draws[0])
            for dr in above_draws[1:]:
                core_rect |= dr

        y_min = top_limit
        if i > 0:
            y_min = max(y_min, fig_captions[i - 1]["rect"].y1 + 2)
        for br in body_blocks + table_barriers:
            if br.y1 <= core_rect.y0 + 6:
                y_min = max(y_min, br.y1 + 2)

        y_max = bot_limit
        if cap_r.y0 >= core_rect.y1 - 12:
            y_max = min(y_max, cap_r.y0 - 2)
        else:
            if i + 1 < K:
                y_max = min(y_max, fig_captions[i + 1]["rect"].y0 - 2)
        for br in body_blocks + table_barriers:
            if br.y0 >= core_rect.y1 - 6:
                y_max = min(y_max, br.y0 - 2)

        fig_rect = fitz.Rect(core_rect)

        candidate_rects = []
        for dr in valid_draw_rects:
            if dr.y0 >= y_min - 2 and dr.y1 <= y_max + 2:
                if not dr.intersects(cap_r):
                    candidate_rects.append(dr)
        for tb in text_blocks:
            tr = tb["rect"]
            if tr.y0 >= y_min - 2 and tr.y1 <= y_max + 2:
                if not tr.intersects(cap_r) and not any(tr.intersects(bb) for bb in body_blocks + table_barriers):
                    candidate_rects.append(tr)

        for cr in candidate_rects:
            fig_rect |= cr

        fig_rect = fitz.Rect(
            max(page.rect.x0 + 8, fig_rect.x0 - 4),
            max(top_limit, fig_rect.y0 - 4),
            min(page.rect.x1 - 8, fig_rect.x1 + 4),
            min(bot_limit, fig_rect.y1 + 4),
        )
        fig_zones.append({"fig_num": cap["fig_num"], "rect": fig_rect, "cap_rect": cap_r})

    for c in fig_captions:
        cr = c["rect"] + (-1, -1, 1, 1)
        page.draw_rect(cr, color=(1, 1, 1), fill=(1, 1, 1))
    for bb in body_blocks + table_barriers:
        br = bb + (-1, -1, 1, 1)
        page.draw_rect(br, color=(1, 1, 1), fill=(1, 1, 1))

    results = []
    for zone in fig_zones:
        fnum = zone["fig_num"]
        frect = zone["rect"]
        pix = page.get_pixmap(clip=frect, dpi=200, alpha=False)
        png_bytes = pix.tobytes("png")
        b64_str = base64.b64encode(png_bytes).decode("utf-8")
        img_id = f"p{page_num}_fig{fnum}"
        results.append({
            "id": img_id,
            "fig_num": fnum,
            "data_uri": f"data:image/png;base64,{b64_str}",
            "width": pix.width,
            "height": pix.height,
            "aspect": round(pix.width / max(pix.height, 1), 2),
        })

    return results, next_auto_fig

def extract_text_and_figures_from_bytes(pdf_bytes: bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    full_text_blocks = []
    extracted_images = {}
    xref_counts = {}
    for page in doc:
        for img in page.get_images(full=True):
            xref = img[0]
            xref_counts[xref] = xref_counts.get(xref, 0) + 1

    stop_text_extraction = False
    next_auto_fig = 1

    for page_idx, page in enumerate(doc):
        page_num = page_idx + 1
        page_text = page.get_text("text")

        ref_match = re.search(r"\nReferences\s*\n", page_text, re.IGNORECASE)
        if ref_match:
            page_text = page_text[: ref_match.start()]
            stop_text_extraction = True

        page_figs, next_auto_fig = extract_figures_from_page_render(page, page_num, xref_counts, next_auto_fig)
        page_image_ids = []
        for item in page_figs:
            img_id = item["id"]
            extracted_images[img_id] = item
            page_image_ids.append(f"{img_id}(Figure {item['fig_num']}: {item['width']}x{item['height']})")

        img_marker = f"\n[이 페이지에서 추출된 원본 Figure 이미지 ID 목록: {', '.join(page_image_ids)}]\n" if page_image_ids else ""
        full_text_blocks.append(f"--- [Page {page_num}] ---{img_marker}{page_text}")

        if stop_text_extraction:
            break

    doc.close()
    return "\n\n".join(full_text_blocks), extracted_images


# =====================================================================
# 4. Gemini API 요약 호출
# =====================================================================
def summarize_with_gemini(client: genai.Client, model_chain: list[str], extracted_text: str) -> tuple[PaperSummaryResult, str]:
    prompt = f"""
다음 영상의학 논문 텍스트를 분석하여 영상의학과 전문의가 빠르게 복습할 수 있도록 지정된 JSON 스키마에 맞춰 한글로 정리해 줘.

[절대 준수 작성 규칙]
1. 해부학(Anatomy) 및 의학용어 원칙 (가장 중요!):
   - 뼈, 관절, 혈관(동맥·정맥), 신경, 근육, 인대, 힘줄, 장기 등 모든 해부학 명칭은 절대로 한글만 단독으로 쓰지 말 것!
   - 반드시 영문 원어를 그대로 쓰거나, 한글(영문) 형태로 매번 병기할 것.
   - 영상 시퀀스(T1WI, T2WI, FS, PDWI, DWI, ADC, Doppler US, CTA, MRA, CTC, MIP 등)도 실제 판독문 스타일로 영문을 적극 병기할 것.
2. 파트 1 (전체 핵심 요약): 질환 정의, 임상적 특징, 핵심 영상 소견(Key Imaging Features), 감별진단(DDx), 판독 시 주의점(Pitfalls)만 개조식으로 압축할 것.
3. 파트 2 (핵심 표): 본문 주요 표 번역 또는 주요 감별진단(DDx) 비교 요약 표로 구성할 것.
4. 파트 3 (Figure 제목 진단명 표기 및 캡션 전수 정리):
   - 모든 Figure를 번호 순서대로 빠짐없이 정리할 것.
   - `figure_number`에는 'Figure 1', 'Figure 2'처럼 번호만 적고, `case_diagnosis`에는 해당 Figure의 핵심 진단명 또는 핵심 주제를 '한글명 (English Title)' 형태로 명확히 작성할 것!
   - '[이 페이지에서 추출된 원본 Figure 이미지 ID 목록: ...]'을 참고하여 순수 ID(예: 'p3_fig1')만 정확히 image_ids 리스트에 넣을 것.
   - 각 설명(`descriptions`)은 '[Figure 하위번호] → [시퀀스/단면/검사명 또는 모식도] → [화살표·원·별표 지시 해부학 위치(영문 병기 필수) 및 핵심 소견] → [의미/진단]' 구조로 상세히 작성할 것.

[논문 원문 텍스트]
{extracted_text}
"""
    last_exception = None
    for model_name in model_chain:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=PaperSummaryResult,
                    temperature=0.2,
                ),
            )
            return PaperSummaryResult.model_validate_json(response.text), model_name
        except Exception as e:
            last_exception = e
            continue
    raise last_exception


# =====================================================================
# 5. HTML 템플릿 렌더링
# =====================================================================
def render_figure_card_html(fig_num: str, diagnosis: str, img_objs: list[dict], descriptions: list[str], is_pdf: bool = False) -> str:
    clean_num = fig_num.strip().rstrip(":")
    clean_diag = diagnosis.strip()
    header_html = f"{clean_num} : <span class='diag-highlight'>{clean_diag}</span>" if clean_diag else clean_num
    desc_items_html = "".join(f"<li>{d}</li>" for d in descriptions)
    count = len(img_objs)
    
    # PDF 생성 시에는 너무 큰 여백을 줄이고 사진 크기를 살짝 축소하여 페이지 넘김 개선
    max_h_single = "580px" if is_pdf else "680px"
    max_h_grid = "450px" if is_pdf else "540px"
    margin_bottom = "10px" if is_pdf else "14px"

    if count == 0:
        return f"""
        <div class="figure-card" style="margin-bottom: {margin_bottom};">
            <div class="figure-title">{header_html}</div>
            <ul class="figure-desc">{desc_items_html}</ul>
        </div>
        """

    if count == 1 and img_objs[0]["aspect"] <= 0.85:
        tall_img = img_objs[0]
        return f"""
        <div class="figure-card" style="margin-bottom: {margin_bottom};">
            <div class="figure-title">{header_html}</div>
            <div class="side-by-side-table">
                <div class="side-img-col">
                    <img src="{tall_img['data_uri']}" class="tall-single-img" style="max-height: {max_h_single};" />
                </div>
                <div class="side-desc-col">
                    <ul class="figure-desc side-desc">{desc_items_html}</ul>
                </div>
            </div>
        </div>
        """

    if count == 1:
        single_img = img_objs[0]
        return f"""
        <div class="figure-card" style="margin-bottom: {margin_bottom};">
            <div class="figure-title">{header_html}</div>
            <div class="full-width-img-box">
                <img src="{single_img['data_uri']}" class="full-width-img" style="max-height: {max_h_single};" />
            </div>
            <ul class="figure-desc">{desc_items_html}</ul>
        </div>
        """

    cols = 2 if count in (2, 4) or any(i["aspect"] >= 1.3 for i in img_objs) else 3
    width_pct = {2: "49.4%", 3: "32.6%"}[cols]
    cells_html = "".join(
        f'<div class="fig-cell" style="width: {width_pct};"><img src="{item["data_uri"]}" class="grid-fill-img" style="max-height: {max_h_grid};" /></div>'
        for item in img_objs
    )
    return f"""
    <div class="figure-card" style="margin-bottom: {margin_bottom};">
        <div class="figure-title">{header_html}</div>
        <div class="figure-grid">{cells_html}</div>
        <ul class="figure-desc">{desc_items_html}</ul>
    </div>
    """


def build_full_report_html(summary: PaperSummaryResult, extracted_images: dict, is_pdf: bool = False) -> str:
    def render_list(items):
        return "".join(f"<li>{item}</li>" for item in items)

    tables_html = ""
    for tbl in summary.tables:
        headers_html = "".join(f"<th>{h}</th>" for h in tbl.headers)
        rows_html = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in tbl.rows)
        tables_html += f"""
        <div class="table-box">
            <h3>{tbl.title}</h3>
            <div class="table-scroll">
                <table>
                    <thead><tr>{headers_html}</tr></thead>
                    <tbody>{rows_html}</tbody>
                </table>
            </div>
        </div>
        """

    fig_num_map = {}
    for img_id, item in extracted_images.items():
        fnum = item.get("fig_num")
        if fnum is not None:
            fig_num_map.setdefault(fnum, []).append(img_id)

    figures_html = ""
    used_img_ids = set()
    for fig in summary.figures:
        valid_img_objs = []
        for raw_id in fig.image_ids:
            clean_id = raw_id.split("(")[0].strip()
            if clean_id in extracted_images and clean_id not in used_img_ids:
                used_img_ids.add(clean_id)
                valid_img_objs.append(extracted_images[clean_id])

        if not valid_img_objs:
            m_num = re.search(r"(\d+)", fig.figure_number)
            if m_num:
                fnum = int(m_num.group(1))
                for cid in fig_num_map.get(fnum, []):
                    if cid not in used_img_ids:
                        used_img_ids.add(cid)
                        valid_img_objs.append(extracted_images[cid])

        figures_html += render_figure_card_html(
            fig.figure_number, fig.case_diagnosis, valid_img_objs, fig.descriptions, is_pdf=is_pdf
        )

    # PDF일 때만 글자 크기와 여백을 촘촘하게 압축하여 페이지 수를 최소화
    page_css = ""
    if is_pdf:
        page_css = """
        @page { size: A4; margin: 10mm 9mm; }
        body { font-size: 9.2pt; line-height: 1.48; padding: 0; }
        .report-wrap { max-width: 100%; margin: 0; }
        h1 { font-size: 14pt; padding-bottom: 4px; }
        h2 { font-size: 11pt; padding: 4px 8px; margin-top: 14px; margin-bottom: 6px; }
        h3 { font-size: 10pt; margin-top: 10px; }
        .figure-card { padding: 8px; border-radius: 4px; }
        .figure-desc { padding: 6px 8px 6px 20px; font-size: 8.8pt; line-height: 1.48; }
        """
    else:
        page_css = """
        @page { size: A4; margin: 11mm 10mm; }
        body { font-size: 10pt; line-height: 1.55; padding: 12px; }
        .report-wrap { max-width: 840px; margin: 0 auto; }
        h1 { font-size: 15pt; padding-bottom: 6px; }
        h2 { font-size: 11.5pt; padding: 6px 10px; margin-top: 18px; margin-bottom: 8px; }
        h3 { font-size: 10.5pt; margin-top: 12px; }
        .figure-card { padding: 10px; border-radius: 6px; }
        .figure-desc { padding: 8px 10px 8px 22px; font-size: 9.5pt; line-height: 1.55; }
        """

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{summary.paper_title_ko}</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Nanum+Gothic:wght@400;700&display=swap');
        body {{
            font-family: 'Nanum Gothic', 'Malgun Gothic', '맑은 고딕', -apple-system, sans-serif;
            color: #1e293b; background-color: #ffffff;
            -webkit-print-color-adjust: exact; print-color-adjust: exact;
        }}
        {page_css}
        h1 {{ color: #0f2c59; border-bottom: 2px solid #0f2c59; margin-top: 0; }}
        h2 {{ color: #ffffff; background-color: #1e3a8a; }}
        h3 {{ color: #1e3a8a; margin-bottom: 4px; border-left: 4px solid #1e3a8a; padding-left: 6px; }}
        ul {{ margin-top: 4px; margin-bottom: 10px; padding-left: 18px; }}
        li {{ margin-bottom: 6px; }}
        .table-scroll {{ overflow-x: auto; -webkit-overflow-scrolling: touch; }}
        table {{ width: 100%; border-collapse: collapse; margin: 6px 0 12px 0; font-size: 9pt; }}
        th {{ background-color: #e2e8f0; color: #0f172a; font-weight: bold; border: 1px solid #cbd5e1; padding: 6px 8px; text-align: left; }}
        td {{ border: 1px solid #cbd5e1; padding: 6px 8px; vertical-align: top; }}
        .figure-card {{ border: 1px solid #cbd5e1; background-color: #f8fafc; break-inside: avoid; page-break-inside: avoid; }}
        .figure-title {{ font-size: 11pt; font-weight: bold; color: #0f2c59; background-color: #e2e8f0; border-left: 5px solid #1e3a8a; padding: 5px 10px; margin-bottom: 8px; border-radius: 2px; }}
        .diag-highlight {{ color: #1e3a8a; font-weight: bold; }}
        .full-width-img {{ width: 100%; height: auto; object-fit: contain; border: 1px solid #cbd5e1; border-radius: 4px; background: #ffffff; display: block; margin: 4px auto 8px auto; }}
        .side-by-side-table {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: flex-start; margin: 4px 0; }}
        .side-img-col {{ flex: 1 1 260px; text-align: center; }}
        .side-desc-col {{ flex: 1 1 260px; }}
        .tall-single-img {{ width: 100%; height: auto; object-fit: contain; border: 1px solid #cbd5e1; border-radius: 4px; background: #ffffff; display: block; }}
        .figure-grid {{ text-align: center; font-size: 0; margin: 4px 0 8px 0; width: 100%; }}
        .fig-cell {{ display: inline-block; vertical-align: top; box-sizing: border-box; padding: 2px; text-align: center; }}
        .grid-fill-img {{ width: 100%; height: auto; object-fit: contain; border: 1px solid #cbd5e1; border-radius: 4px; background: #ffffff; display: block; }}
        .figure-desc {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 4px; margin: 0; }}
        @media (max-width: 600px) {{
            .fig-cell {{ width: 100% !important; display: block; }}
        }}
    </style>
</head>
<body>
    <div class="report-wrap">
        <h1>{summary.paper_title_ko}</h1>

        <h2>파트 1. 전체 핵심 요약 (Core Summary)</h2>
        <h3>1) 질환 정의 및 임상적 특징</h3>
        <ul>{render_list(summary.definition_and_clinical)}</ul>

        <h3>2) 핵심 영상 소견 (Key Imaging Features)</h3>
        <ul>{render_list(summary.key_imaging_features)}</ul>

        <h3>3) 감별진단 (Differential Diagnosis)</h3>
        <ul>{render_list(summary.differential_diagnosis)}</ul>

        <h3>4) 판독 시 주의점 (Pitfalls) 및 치료 원칙</h3>
        <ul>{render_list(summary.pitfalls_and_management)}</ul>

        <h2>파트 2. 핵심 요약 표 (Summary Table)</h2>
        {tables_html}

        <h2>파트 3. 원본 Figure 이미지 및 캡션 전수 정리</h2>
        {figures_html}
    </div>
</body>
</html>"""


# =====================================================================
# 6. Streamlit 클라우드 전용 PDF 구워내기 (xvfb 가상 디스플레이 적용)
# =====================================================================
def generate_pdf_from_html(html_content: str) -> bytes:
    options = {
        'page-size': 'A4',
        'margin-top': '0mm',
        'margin-right': '0mm',
        'margin-bottom': '0mm',
        'margin-left': '0mm',
        'encoding': 'UTF-8',
        'enable-local-file-access': '',
        'no-outline': None,
        'print-media-type': '',
        'quiet': ''
    }
    try:
        # Streamlit Cloud 리눅스 환경에서 X 서버 없이 렌더링하기 위한 가상 래퍼 설정
        config = pdfkit.configuration(wkhtmltopdf='/usr/bin/wkhtmltopdf')
        pdf_bytes = pdfkit.from_string(html_content, False, options=options, configuration=config)
        return pdf_bytes
    except Exception as e:
        print(f"PDF 생성 오류: {e}")
        return None


# =====================================================================
# 7. 모바일 & PC 반응형 웹 인터페이스 UI
# =====================================================================
st.set_page_config(
    page_title="영상의학 논문 한글 요약기",
    page_icon="🏥",
    layout="centered",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 860px; }
    .main-banner {
        background: linear-gradient(135deg, #0f2c59, #1e3a8a);
        color: white; padding: 16px 20px; border-radius: 12px; margin-bottom: 16px;
    }
    .main-banner h2 { margin: 0; color: white; font-size: 1.3rem; }
    .main-banner p { margin: 4px 0 0 0; font-size: 0.85rem; opacity: 0.9; }
</style>
<div class="main-banner">
    <h2>🏥 영상의학 논문 한글 요약기</h2>
    <p>모식도·벡터·CT 무손실 캡처 · 모바일(HTML) & PC(PDF) 통합 저장</p>
</div>
""", unsafe_allow_html=True)

if "api_key" not in st.session_state:
    st.session_state.api_key = load_saved_key()

with st.expander("🔑 Gemini API 키 설정 (최초 1회 저장)", expanded=not bool(st.session_state.api_key)):
    key_input = st.text_input("Gemini API 키 입력", value=st.session_state.api_key, type="password")
    if st.button("💾 API 키 저장", use_container_width=True):
        if key_input.strip():
            st.session_state.api_key = key_input.strip()
            save_api_key_local(st.session_state.api_key)
            st.success("✅ API 키가 저장되었습니다!")
        else:
            st.warning("API 키를 입력해 주세요.")

tier_choice = st.radio(
    "🤖 AI 모델 선택",
    options=["flash", "pro"],
    format_func=lambda x: "⚡ Gemini Flash → [한글판] (빠른 요약 · 무료 API 추천)" if x == "flash" else "🧠 Gemini Pro → [한글판프로] (고정밀 판독 · 한도 초과 시 자동 우회)",
    horizontal=True,
)

uploaded_files = st.file_uploader(
    "📄 요약할 영문 영상의학 PDF 논문을 선택하세요 (여러 개 동시 선택 가능)",
    accept_multiple_files=True,
)

if uploaded_files:
    if st.button("🚀 선택한 PDF 논문 한글 요약 시작", type="primary", use_container_width=True):
        if not st.session_state.api_key:
            st.error("❌ 상단의 [Gemini API 키 설정]을 눌러 API 키를 먼저 입력해 주세요.")
        else:
            client = genai.Client(api_key=st.session_state.api_key)
            model_chain = build_fallback_model_chain(client, preferred_tier=tier_choice)
            st.session_state.results = []

            for idx, up_file in enumerate(uploaded_files, 1):
                with st.status(f"[{idx}/{len(uploaded_files)}] 📄 {up_file.name} 분석 및 요약 중...", expanded=True) as status:
                    try:
                        pdf_bytes = up_file.read()
                        st.write("🔍 원본 본문 및 Figure(모식도·표·CT) 고해상도 통합 캡처 중...")
                        extracted_text, extracted_images = extract_text_and_figures_from_bytes(pdf_bytes)
                        st.write(f"✅ 총 {len(extracted_images)}개 Figure 정밀 캡처 완료! AI 요약 생성 중...")

                        summary_result, used_model = summarize_with_gemini(client, model_chain, extracted_text)
                        prefix_tag = "[한글판프로]" if "pro" in used_model else "[한글판]"
                        base_name = os.path.splitext(up_file.name)[0]

                        # 모바일 전용 뷰어용 HTML (넓고 시원한 여백)
                        web_html = build_full_report_html(summary_result, extracted_images, is_pdf=False)
                        
                        # 인쇄·PC 보관용 압축 레이아웃 HTML을 만든 뒤 PDF로 굽기
                        st.write("🖨️ A4 판독 레이아웃 PDF 파일 생성 중...")
                        pdf_html = build_full_report_html(summary_result, extracted_images, is_pdf=True)
                        pdf_out_bytes = generate_pdf_from_html(pdf_html)

                        st.session_state.results.append({
                            "title": summary_result.paper_title_ko,
                            "used_model": used_model,
                            "img_count": len(extracted_images),
                            "web_html": web_html,
                            "html_name": f"{prefix_tag} {base_name}.html",
                            "pdf_bytes": pdf_out_bytes,
                            "pdf_name": f"{prefix_tag} {base_name}.pdf",
                        })
                        status.update(label=f"✅ [{used_model}] {up_file.name} 변환 완료!", state="complete", expanded=False)
                    except Exception as e:
                        status.update(label=f"❌ {up_file.name} 오류 발생", state="error")
                        st.error(f"오류 상세: {e}")

if st.session_state.get("results"):
    st.divider()
    for i, res in enumerate(st.session_state.results):
        st.subheader(f"📑 {res['title']}")
        st.caption(f"사용 모델: {res['used_model']} | 정밀 캡처된 Figure: {res['img_count']}개")

        # 다운로드 버튼을 2개로 분리 (PDF용, HTML용)
        col1, col2 = st.columns(2)
        with col1:
            if res.get("pdf_bytes"):
                st.download_button(
                    label=f"📥 A4 판독 레이아웃 PDF로 저장하기 (.pdf)",
                    data=res["pdf_bytes"],
                    file_name=res["pdf_name"],
                    mime="application/pdf",
                    use_container_width=True,
                    key=f"dl_pdf_{i}",
                )
            else:
                st.warning("⚠️ 서버 환경 문제로 PDF 변환에 실패했습니다. 우측 HTML을 다운로드해 주세요.")
        
        with col2:
            st.download_button(
                label=f"📱 스마트폰 뷰어 겸용 파일로 저장하기 (.html)",
                data=res["web_html"].encode("utf-8"),
                file_name=res["html_name"],
                mime="text/html",
                use_container_width=True,
                key=f"dl_html_{i}",
            )

        # 화면에는 모바일 뷰어용(넓은 레이아웃)을 띄워줌
        components.html(res["web_html"], height=850, scrolling=True)
