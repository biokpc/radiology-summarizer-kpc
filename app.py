import os
import sys
import re
import json
import time
import base64
import subprocess
import types as _py_types
import streamlit as st
import streamlit.components.v1 as components

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
# 1. 설정 및 API 키 관리 (클라우드 Secrets / 로컬 파일 / URL 파라미터 연동)
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
        description="이 Figure 전체가 보여주는 핵심 증례 진단명 1개. 반드시 '한글 진단명 (영문 진단명)' 형태로 작성"
    )
    image_ids: list[str] = Field(description="해당 Figure에 매칭되는 추출 이미지 ID 리스트 (예: ['p1_img1'])")
    descriptions: list[str] = Field(
        description="하위 항목별 설명. 반드시 '[Figure 번호] → [시퀀스/검사명] → [화살표·별표 등 지시 기호 소견 (해부학 명칭은 반드시 영문 병기)] → [의미/진단]' 구조로 작성"
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
# 3. PDF 바이트에서 텍스트 및 Figure 이미지 추출
# =====================================================================
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

    for page_idx, page in enumerate(doc):
        page_num = page_idx + 1
        page_text = page.get_text("text")

        ref_match = re.search(r"\nReferences\s*\n", page_text, re.IGNORECASE)
        if ref_match:
            page_text = page_text[: ref_match.start()]
            stop_text_extraction = True

        img_list = page.get_images(full=True)
        page_image_ids = []
        img_counter = 1

        for img_info in img_list:
            xref = img_info[0]
            if xref_counts.get(xref, 0) > 1:
                continue

            base_img = doc.extract_image(xref)
            width, height = base_img["width"], base_img["height"]
            if width < 160 or height < 120 or (width / height > 5.0):
                continue

            img_ext = base_img["ext"]
            img_id = f"p{page_num}_img{img_counter}"
            b64_str = base64.b64encode(base_img["image"]).decode("utf-8")

            extracted_images[img_id] = {
                "data_uri": f"data:image/{img_ext};base64,{b64_str}",
                "width": width,
                "height": height,
                "aspect": round(width / height, 2)
            }
            page_image_ids.append(f"{img_id}({width}x{height})")
            img_counter += 1

        img_marker = f"\n[이 페이지에서 추출된 원본 Figure 이미지 ID 목록: {', '.join(page_image_ids)}]\n" if page_image_ids else ""
        full_text_blocks.append(f"--- [Page {page_num}] ---{img_marker}{page_text}")

        if stop_text_extraction:
            break

    doc.close()
    return "\n\n".join(full_text_blocks), extracted_images


# =====================================================================
# 4. Gemini API 요약 및 판독 레이아웃 HTML/PDF 생성
# =====================================================================
def summarize_with_gemini(client: genai.Client, model_chain: list[str], extracted_text: str) -> tuple[PaperSummaryResult, str]:
    prompt = f"""
다음 영상의학 논문 텍스트를 분석하여 영상의학과 전문의가 빠르게 복습할 수 있도록 지정된 JSON 스키마에 맞춰 한글로 정리해 줘.

[절대 준수 작성 규칙]
1. 해부학(Anatomy) 및 의학용어 원칙 (가장 중요!):
   - 뼈, 관절, 혈관(동맥·정맥), 신경, 근육, 인대, 힘줄 등 모든 해부학 명칭은 절대로 한글만 단독으로 쓰지 말 것!
   - 반드시 영문 원어를 그대로 쓰거나, 한글(영문) 형태로 매번 병기(예: 원위 척골동맥(distal ulnar artery), 갈고리뼈 갈고리(hook of hamate))할 것.
   - 영상 시퀀스(T1WI, T2WI, FS, PDWI, DWI, ADC, Doppler US, CTA, MRA, DSA 등)도 실제 판독문 스타일로 영문을 적극 병기할 것.
2. 파트 1 (전체 핵심 요약): 질환 정의, 임상적 특징, 핵심 영상 소견(Key Imaging Features), 감별진단(DDx), 판독 시 주의점(Pitfalls)만 개조식으로 압축할 것.
3. 파트 2 (핵심 표): 본문 주요 표 번역 또는 주요 감별진단(DDx) 비교 요약 표 1개로 구성할 것.
4. 파트 3 (Figure 제목 진단명 표기 및 캡션 전수 정리):
   - 모든 Figure를 번호 순서대로 정리할 것.
   - `figure_number`에는 'Figure 1'처럼 번호만 적고, `case_diagnosis`에는 해당 Figure의 핵심 진단명 딱 1개를 '한글 진단명 (English Diagnosis)' 형태로 명확히 작성할 것!
   - '[이 페이지에서 추출된 원본 Figure 이미지 ID 목록: ...]'을 참고하여 순수 ID(예: 'p1_img1')만 빠짐없이 image_ids 리스트에 넣을 것.
   - 각 설명(`descriptions`)은 '[Figure 하위번호] → [시퀀스/단면/검사명] → [화살표·별표 지시 해부학 위치(영문 병기 필수) 및 영상 소견] → [의미/진단]' 구조로 작성할 것.

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


def render_figure_card_html(fig_num: str, diagnosis: str, img_objs: list[dict], descriptions: list[str]) -> str:
    clean_num = fig_num.strip().rstrip(":")
    clean_diag = diagnosis.strip()
    header_html = f"{clean_num} : <span class='diag-highlight'>{clean_diag}</span>" if clean_diag else clean_num
    desc_items_html = "".join(f"<li>{d}</li>" for d in descriptions)
    count = len(img_objs)

    if count == 0:
        return f"""
        <div class="figure-card">
            <div class="figure-title">{header_html}</div>
            <ul class="figure-desc">{desc_items_html}</ul>
        </div>
        """

    if count == 1 and img_objs[0]["aspect"] <= 0.88:
        tall_img = img_objs[0]
        return f"""
        <div class="figure-card">
            <div class="figure-title">{header_html}</div>
            <div class="side-by-side-table">
                <div class="side-img-col">
                    <img src="{tall_img['data_uri']}" class="tall-single-img" />
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
        <div class="figure-card">
            <div class="figure-title">{header_html}</div>
            <div class="full-width-img-box">
                <img src="{single_img['data_uri']}" class="full-width-img" />
            </div>
            <ul class="figure-desc">{desc_items_html}</ul>
        </div>
        """

    cols = 2 if count in (2, 4) or any(i["aspect"] >= 1.3 for i in img_objs) else 3
    width_pct = {2: "49.4%", 3: "32.6%"}[cols]
    cells_html = "".join(
        f'<div class="fig-cell" style="width: {width_pct};"><img src="{item["data_uri"]}" class="grid-fill-img" /></div>'
        for item in img_objs
    )
    return f"""
    <div class="figure-card">
        <div class="figure-title">{header_html}</div>
        <div class="figure-grid">{cells_html}</div>
        <ul class="figure-desc">{desc_items_html}</ul>
    </div>
    """


def build_full_report_html(summary: PaperSummaryResult, extracted_images: dict) -> str:
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

    figures_html = ""
    used_img_ids = set()
    for fig in summary.figures:
        valid_img_objs = []
        for raw_id in fig.image_ids:
            clean_id = raw_id.split("(")[0].strip()
            if clean_id in extracted_images:
                used_img_ids.add(clean_id)
                valid_img_objs.append(extracted_images[clean_id])

        figures_html += render_figure_card_html(
            fig.figure_number, fig.case_diagnosis, valid_img_objs, fig.descriptions
        )

    unmatched_objs = [extracted_images[k] for k in extracted_images.keys() if k not in used_img_ids]
    if unmatched_objs:
        figures_html += render_figure_card_html(
            "추가 추출 이미지", "Additional Figures", unmatched_objs, ["본문 자동 매칭 외 추가 이미지입니다."]
        )

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{summary.paper_title_ko}</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Nanum+Gothic:wght@400;700&display=swap');
        @page {{ size: A4; margin: 11mm 10mm; }}
        body {{
            font-family: 'Nanum Gothic', 'Malgun Gothic', '맑은 고딕', -apple-system, sans-serif;
            font-size: 10pt; line-height: 1.55; color: #1e293b;
            margin: 0; padding: 12px; background-color: #ffffff;
            -webkit-print-color-adjust: exact; print-color-adjust: exact;
        }}
        .report-wrap {{ max-width: 840px; margin: 0 auto; }}
        h1 {{ font-size: 15pt; color: #0f2c59; border-bottom: 2px solid #0f2c59; padding-bottom: 6px; margin-top: 0; }}
        h2 {{ font-size: 11.5pt; color: #ffffff; background-color: #1e3a8a; padding: 6px 10px; border-radius: 4px; margin-top: 18px; margin-bottom: 8px; }}
        h3 {{ font-size: 10.5pt; color: #1e3a8a; margin-top: 12px; margin-bottom: 4px; border-left: 4px solid #1e3a8a; padding-left: 6px; }}
        ul {{ margin-top: 4px; margin-bottom: 10px; padding-left: 18px; }}
        li {{ margin-bottom: 6px; }}
        .table-scroll {{ overflow-x: auto; -webkit-overflow-scrolling: touch; }}
        table {{ width: 100%; border-collapse: collapse; margin: 6px 0 12px 0; font-size: 9pt; }}
        th {{ background-color: #e2e8f0; color: #0f172a; font-weight: bold; border: 1px solid #cbd5e1; padding: 6px 8px; text-align: left; }}
        td {{ border: 1px solid #cbd5e1; padding: 6px 8px; vertical-align: top; }}
        .figure-card {{ border: 1px solid #cbd5e1; border-radius: 6px; padding: 10px; margin-bottom: 14px; background-color: #f8fafc; break-inside: avoid; page-break-inside: avoid; }}
        .figure-title {{ font-size: 11pt; font-weight: bold; color: #0f2c59; background-color: #e2e8f0; border-left: 5px solid #1e3a8a; padding: 5px 10px; margin-bottom: 8px; border-radius: 2px; }}
        .diag-highlight {{ color: #1e3a8a; font-weight: bold; }}
        .full-width-img {{ width: 100%; max-height: 600px; height: auto; object-fit: contain; border: 1px solid #94a3b8; border-radius: 4px; background: #000; display: block; margin: 4px auto 8px auto; }}
        .side-by-side-table {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: flex-start; margin: 4px 0; }}
        .side-img-col {{ flex: 1 1 260px; text-align: center; }}
        .side-desc-col {{ flex: 1 1 260px; }}
        .tall-single-img {{ width: 100%; max-height: 660px; height: auto; object-fit: contain; border: 1px solid #94a3b8; border-radius: 4px; background: #000; display: block; }}
        .figure-grid {{ text-align: center; font-size: 0; margin: 4px 0 8px 0; width: 100%; }}
        .fig-cell {{ display: inline-block; vertical-align: top; box-sizing: border-box; padding: 2px; text-align: center; }}
        .grid-fill-img {{ width: 100%; max-height: 520px; height: auto; object-fit: contain; border: 1px solid #94a3b8; border-radius: 4px; background: #000; display: block; }}
        .figure-desc {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 4px; padding: 8px 10px 8px 22px; margin: 0; font-size: 9.5pt; line-height: 1.55; }}
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


def try_generate_pdf_bytes(full_html: str):
    try:
        import weasyprint
        return weasyprint.HTML(string=full_html).write_pdf()
    except Exception:
        return None


# =====================================================================
# 5. Streamlit 모바일 & PC 반응형 웹 인터페이스
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
    <p>Figure 진단명 자동 표기 · 해부학 영문 병기 · 24시간 클라우드 웹앱</p>
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
            st.success("✅ API 키가 저장되었습니다! 현재 주소를 북마크하거나 [홈 화면에 추가]하면 계속 유지됩니다.")
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
    type=["pdf"],
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
                        st.write("🔍 원본 본문 텍스트 및 Figure 이미지 추출 중...")
                        extracted_text, extracted_images = extract_text_and_figures_from_bytes(pdf_bytes)
                        st.write(f"✅ Figure 이미지 {len(extracted_images)}개 추출 완료! AI 요약 생성 중...")

                        summary_result, used_model = summarize_with_gemini(client, model_chain, extracted_text)
                        prefix_tag = "[한글판프로]" if "pro" in used_model else "[한글판]"
                        base_name = os.path.splitext(up_file.name)[0]

                        full_html = build_full_report_html(summary_result, extracted_images)
                        pdf_out = try_generate_pdf_bytes(full_html)

                        st.session_state.results.append({
                            "title": summary_result.paper_title_ko,
                            "used_model": used_model,
                            "img_count": len(extracted_images),
                            "html": full_html,
                            "pdf_bytes": pdf_out,
                            "pdf_name": f"{prefix_tag} {base_name}.pdf",
                            "html_name": f"{prefix_tag} {base_name}.html",
                        })
                        status.update(label=f"✅ [{used_model}] {up_file.name} 변환 완료!", state="complete", expanded=False)
                    except Exception as e:
                        status.update(label=f"❌ {up_file.name} 오류 발생", state="error")
                        st.error(f"오류 상세: {e}")

if st.session_state.get("results"):
    st.divider()
    for i, res in enumerate(st.session_state.results):
        st.subheader(f"📑 {res['title']}")
        st.caption(f"사용 모델: {res['used_model']} | 추출된 Figure 이미지: {res['img_count']}개")

        col1, col2 = st.columns(2)
        with col1:
            if res["pdf_bytes"]:
                st.download_button(
                    label=f"📥 {res['pdf_name']} 저장",
                    data=res["pdf_bytes"],
                    file_name=res["pdf_name"],
                    mime="application/pdf",
                    use_container_width=True,
                    key=f"dl_pdf_{i}",
                )
            else:
                st.download_button(
                    label=f"📥 {res['html_name']} 저장 (단독 실행 파일)",
                    data=res["html"].encode("utf-8"),
                    file_name=res["html_name"],
                    mime="text/html",
                    use_container_width=True,
                    key=f"dl_html_{i}",
                )
        with col2:
            if res["pdf_bytes"]:
                st.download_button(
                    label=f"📱 모바일 뷰어 파일(.html) 저장",
                    data=res["html"].encode("utf-8"),
                    file_name=res["html_name"],
                    mime="text/html",
                    use_container_width=True,
                    key=f"dl_html_sub_{i}",
                )

        components.html(res["html"], height=850, scrolling=True)
