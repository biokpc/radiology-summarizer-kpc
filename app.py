import os
import sys
import re
import json
import time
import base64
import shutil
import socket
import pathlib
import tempfile
import subprocess
import webbrowser
import urllib.parse
import types as _py_types
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

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
# 1. 기존 데스크톱 앱과 설정(API 키) 100% 자동 공유
# =====================================================================
CONFIG_PATH = os.path.expanduser("~/.radiology_pdf_summarizer_win.json")
GENERATED_PDF_STORE = {}  # 다운로드용 PDF 메모리/경로 보관소

def load_config() -> dict:
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"api_key": "", "last_dir": os.path.expanduser("~/Downloads"), "model_tier": "flash"}

def save_config(cfg: dict):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def get_local_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


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
    if preferred_tier in ("pro", "argon"):
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
# 3. PDF 바이트에서 직접 텍스트 및 Figure 이미지 추출
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
                "bytes": base_img["image"],
                "ext": img_ext,
                "filename": f"{img_id}.{img_ext}",
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
# 4. Gemini 요약 호출 및 A4 PDF/모바일 HTML 생성
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
            err_str = str(e)
            if any(k in err_str for k in ["limit: 0", "429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "404", "NOT_FOUND"]):
                continue
            continue
    raise last_exception


def render_figure_card_html(fig_num: str, diagnosis: str, img_objs: list[dict], descriptions: list[str], use_data_uri: bool = False) -> str:
    clean_num = fig_num.strip().rstrip(":")
    clean_diag = diagnosis.strip()
    header_html = f"{clean_num} : <span class='diag-highlight'>{clean_diag}</span>" if clean_diag else clean_num
    desc_items_html = "".join(f"<li>{d}</li>" for d in descriptions)
    count = len(img_objs)

    def get_src(item):
        return item["data_uri"] if use_data_uri else item["filename"]

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
                    <img src="{get_src(tall_img)}" class="tall-single-img" />
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
                <img src="{get_src(single_img)}" class="full-width-img" />
            </div>
            <ul class="figure-desc">{desc_items_html}</ul>
        </div>
        """

    cols = 2 if count in (2, 4) or any(i["aspect"] >= 1.3 for i in img_objs) else 3
    width_pct = {2: "49.4%", 3: "32.6%"}[cols]
    cells_html = "".join(
        f'<div class="fig-cell" style="width: {width_pct};"><img src="{get_src(item)}" class="grid-fill-img" /></div>'
        for item in img_objs
    )
    return f"""
    <div class="figure-card">
        <div class="figure-title">{header_html}</div>
        <div class="figure-grid">{cells_html}</div>
        <ul class="figure-desc">{desc_items_html}</ul>
    </div>
    """


def build_report_body_html(summary: PaperSummaryResult, extracted_images: dict, use_data_uri: bool = True) -> str:
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
            fig.figure_number, fig.case_diagnosis, valid_img_objs, fig.descriptions, use_data_uri=use_data_uri
        )

    unmatched_objs = [extracted_images[k] for k in extracted_images.keys() if k not in used_img_ids]
    if unmatched_objs:
        figures_html += render_figure_card_html(
            "추가 추출 이미지", "Additional Figures", unmatched_objs, ["본문 자동 매칭 외 추가 이미지입니다."], use_data_uri=use_data_uri
        )

    return f"""
    <div class="report-content">
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
    """


def get_available_browsers() -> list[str]:
    local_app = os.environ.get("LOCALAPPDATA", "")
    prog_files = os.environ.get("PROGRAMFILES", r"C:\Program Files")
    prog_x86 = os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")
    candidates = [
        os.path.join(prog_files, r"Google\Chrome\Application\chrome.exe"),
        os.path.join(prog_x86, r"Google\Chrome\Application\chrome.exe"),
        os.path.join(local_app, r"Google\Chrome\Application\chrome.exe"),
        os.path.join(prog_x86, r"Microsoft\Edge\Application\msedge.exe"),
        os.path.join(prog_files, r"Microsoft\Edge\Application\msedge.exe"),
        os.path.join(local_app, r"Microsoft\Edge\Application\msedge.exe"),
    ]
    return [p for p in candidates if p and os.path.exists(p)]


def generate_pdf_file(summary: PaperSummaryResult, extracted_images: dict, output_pdf_path: str) -> bool:
    body_html = build_report_body_html(summary, extracted_images, use_data_uri=False)
    full_html = f"""
    <!DOCTYPE html><html lang="ko"><head><meta charset="UTF-8">
    <style>
        @page {{ size: A4; margin: 11mm 10mm; }}
        body {{ font-family: 'Malgun Gothic', '맑은 고딕', sans-serif; font-size: 9.8pt; line-height: 1.52; color: #1e293b; -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
        h1 {{ font-size: 15.5pt; color: #0f2c59; border-bottom: 2px solid #0f2c59; padding-bottom: 5px; margin-top: 0; }}
        h2 {{ font-size: 11.8pt; color: #ffffff; background-color: #1e3a8a; padding: 5px 10px; border-radius: 4px; margin-top: 16px; margin-bottom: 8px; }}
        h3 {{ font-size: 10.5pt; color: #1e3a8a; margin-top: 10px; margin-bottom: 4px; border-left: 4px solid #1e3a8a; padding-left: 6px; }}
        ul {{ margin-top: 4px; margin-bottom: 8px; padding-left: 18px; }}
        li {{ margin-bottom: 5px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 6px 0 12px 0; font-size: 9pt; }}
        th {{ background-color: #e2e8f0; color: #0f172a; font-weight: bold; border: 1px solid #cbd5e1; padding: 5px 7px; text-align: left; }}
        td {{ border: 1px solid #cbd5e1; padding: 5px 7px; vertical-align: top; }}
        .figure-card {{ border: 1px solid #cbd5e1; border-radius: 6px; padding: 9px 10px; margin-bottom: 14px; background-color: #f8fafc; break-inside: avoid; page-break-inside: avoid; }}
        .figure-title {{ font-size: 11.5pt; font-weight: bold; color: #0f2c59; background-color: #e2e8f0; border-left: 5px solid #1e3a8a; padding: 5px 10px; margin-bottom: 8px; border-radius: 2px; }}
        .diag-highlight {{ color: #1e3a8a; font-weight: bold; }}
        .full-width-img {{ width: 100%; max-height: 600px; height: auto; object-fit: contain; border: 1px solid #94a3b8; border-radius: 4px; background: #000; display: block; margin: 4px auto 8px auto; }}
        .side-by-side-table {{ display: table; width: 100%; table-layout: fixed; margin: 4px 0; }}
        .side-img-col {{ display: table-cell; width: 52%; vertical-align: top; padding-right: 8px; text-align: center; }}
        .side-desc-col {{ display: table-cell; width: 48%; vertical-align: top; }}
        .tall-single-img {{ width: 100%; max-height: 680px; height: auto; object-fit: contain; border: 1px solid #94a3b8; border-radius: 4px; background: #000; display: block; }}
        .figure-grid {{ text-align: center; font-size: 0; margin: 4px 0 8px 0; width: 100%; }}
        .fig-cell {{ display: inline-block; vertical-align: top; box-sizing: border-box; padding: 2px; text-align: center; }}
        .grid-fill-img {{ width: 100%; max-height: 520px; height: auto; object-fit: contain; border: 1px solid #94a3b8; border-radius: 4px; background: #000; display: block; }}
        .figure-desc {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 4px; padding: 8px 10px 8px 22px; margin: 0; font-size: 9.5pt; line-height: 1.52; }}
    </style></head><body>{body_html}</body></html>
    """
    tmp_dir = tempfile.mkdtemp(prefix="rad_mob_")
    tmp_html_path = os.path.join(tmp_dir, "report.html")
    tmp_pdf_path = os.path.join(tmp_dir, "report.pdf")

    try:
        for img_data in extracted_images.values():
            with open(os.path.join(tmp_dir, img_data["filename"]), "wb") as f:
                f.write(img_data["bytes"])
        with open(tmp_html_path, "w", encoding="utf-8") as f:
            f.write(full_html)

        pdf_fwd = tmp_pdf_path.replace("\\", "/")
        html_fwd = tmp_html_path.replace("\\", "/")
        creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

        for idx, browser_exe in enumerate(get_available_browsers()):
            for mode_idx, flag in enumerate(["--headless=new", "--headless=old", "--headless"]):
                cmd = [
                    browser_exe, flag,
                    f"--user-data-dir={tmp_dir.replace(chr(92), '/')}/p_{idx}_{mode_idx}",
                    "--disable-gpu", "--no-first-run", "--allow-file-access-from-files",
                    "--no-pdf-header-footer", f"--print-to-pdf={pdf_fwd}", html_fwd
                ]
                try:
                    subprocess.run(cmd, timeout=45, creationflags=creationflags, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except Exception:
                    pass
                for _ in range(8):
                    if os.path.exists(tmp_pdf_path) and os.path.getsize(tmp_pdf_path) > 1000:
                        shutil.copyfile(tmp_pdf_path, output_pdf_path)
                        return True
                    time.sleep(0.2)
        return False
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# =====================================================================
# 5. 모바일 웹앱 프론트엔드 (반응형 UI + QR 연결 + 홈 화면 앱 설치 지원)
# =====================================================================
def build_mobile_webapp_html(local_url: str, saved_api_key: str, saved_tier: str) -> str:
    qr_api_url = f"https://api.qrserver.com/v1/create-qr-code/?size=160x160&data={urllib.parse.quote(local_url)}"
    has_key = "true" if saved_api_key else "false"

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=5.0">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="mobile-web-app-capable" content="yes">
    <meta name="theme-color" content="#0f2c59">
    <link rel="manifest" href="/manifest.json">
    <title>영상의학 논문 요약기 Mobile</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0; padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, 'Malgun Gothic', '맑은 고딕', sans-serif;
            background-color: #f1f5f9; color: #1e293b;
        }}
        .app-header {{
            background: linear-gradient(135deg, #0f2c59, #1e3a8a);
            color: white; padding: 16px; text-align: center;
            position: sticky; top: 0; z-index: 100;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        }}
        .app-header h1 {{ margin: 0; font-size: 1.2rem; font-weight: 700; color: white; border: none; padding: 0; }}
        .app-header p {{ margin: 4px 0 0 0; font-size: 0.82rem; opacity: 0.9; }}
        .container {{ max-width: 820px; margin: 0 auto; padding: 14px; }}
        .card {{
            background: white; border-radius: 12px; padding: 16px;
            margin-bottom: 14px; box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        }}
        .card-title {{ font-weight: 700; font-size: 0.98rem; color: #0f2c59; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center; }}
        .qr-box {{
            display: flex; align-items: center; gap: 14px;
            background: #f8fafc; border: 1px dashed #94a3b8; border-radius: 10px; padding: 12px;
        }}
        .qr-box img {{ width: 115px; height: 115px; border-radius: 6px; border: 1px solid #cbd5e1; background: white; }}
        .qr-info {{ font-size: 0.88rem; line-height: 1.5; }}
        .url-badge {{
            display: inline-block; background: #e0e7ff; color: #1e3a8a;
            font-weight: 700; padding: 4px 8px; border-radius: 6px; font-family: monospace; font-size: 0.95rem; margin: 4px 0;
        }}
        .tier-group {{ display: flex; gap: 8px; margin-bottom: 12px; }}
        .tier-btn {{
            flex: 1; padding: 12px 8px; border: 2px solid #cbd5e1; border-radius: 10px;
            background: #f8fafc; font-weight: 700; font-size: 0.9rem; color: #475569; cursor: pointer; text-align: center;
        }}
        .tier-btn.active {{ border-color: #1e3a8a; background: #eff6ff; color: #1e3a8a; }}
        .upload-btn {{
            display: block; width: 100%; padding: 16px;
            background: #1e3a8a; color: white; border: none; border-radius: 12px;
            font-size: 1.05rem; font-weight: 700; text-align: center; cursor: pointer;
            box-shadow: 0 3px 6px rgba(30,58,138,0.25);
        }}
        .upload-btn:active {{ transform: scale(0.99); }}
        .api-row {{ display: flex; gap: 6px; margin-top: 8px; }}
        .api-row input {{ flex: 1; padding: 10px; border: 1px solid #cbd5e1; border-radius: 8px; font-size: 0.9rem; }}
        .api-row button {{ padding: 10px 14px; background: #0f766e; color: white; border: none; border-radius: 8px; font-weight: 700; cursor: pointer; }}
        .status-box {{
            display: none; background: #fffbeb; border: 1px solid #fde68a; color: #92400e;
            padding: 14px; border-radius: 10px; font-weight: 600; font-size: 0.92rem; margin-bottom: 14px; text-align: center;
        }}
        .dl-banner {{
            background: #ecfdf5; border: 2px solid #10b981; border-radius: 12px;
            padding: 14px; margin-bottom: 16px; text-align: center;
        }}
        .dl-btn {{
            display: inline-block; width: 100%; padding: 14px;
            background: #059669; color: white; text-decoration: none;
            font-weight: 700; font-size: 1rem; border-radius: 10px; margin-top: 6px;
        }}
        /* 모바일 리포트 전용 스타일 */
        .report-content {{ background: white; border-radius: 12px; padding: 16px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); line-height: 1.6; font-size: 0.96rem; }}
        .report-content h1 {{ font-size: 1.25rem; color: #0f2c59; border-bottom: 2px solid #0f2c59; padding-bottom: 8px; margin-top: 0; }}
        .report-content h2 {{ font-size: 1.05rem; color: white; background: #1e3a8a; padding: 7px 12px; border-radius: 6px; margin-top: 20px; }}
        .report-content h3 {{ font-size: 0.98rem; color: #1e3a8a; border-left: 4px solid #1e3a8a; padding-left: 8px; margin-top: 14px; }}
        .report-content ul {{ padding-left: 20px; margin: 6px 0 12px 0; }}
        .report-content li {{ margin-bottom: 7px; }}
        .table-scroll {{ overflow-x: auto; -webkit-overflow-scrolling: touch; margin: 8px 0 14px 0; }}
        .report-content table {{ width: 100%; border-collapse: collapse; font-size: 0.86rem; min-width: 480px; }}
        .report-content th {{ background: #e2e8f0; padding: 6px 8px; border: 1px solid #cbd5e1; text-align: left; }}
        .report-content td {{ padding: 6px 8px; border: 1px solid #cbd5e1; vertical-align: top; }}
        .figure-card {{ border: 1px solid #cbd5e1; border-radius: 8px; padding: 10px; margin-bottom: 16px; background: #f8fafc; }}
        .figure-title {{ font-size: 1.02rem; font-weight: 700; color: #0f2c59; background: #e2e8f0; border-left: 5px solid #1e3a8a; padding: 6px 10px; margin-bottom: 10px; border-radius: 4px; }}
        .diag-highlight {{ color: #1e3a8a; }}
        .full-width-img, .tall-single-img, .grid-fill-img {{ width: 100%; height: auto; border-radius: 6px; background: #000; display: block; margin-bottom: 8px; }}
        .side-by-side-table {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: flex-start; }}
        .side-img-col {{ flex: 1 1 260px; }}
        .side-desc-col {{ flex: 1 1 260px; }}
        .figure-grid {{ display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 8px; }}
        .fig-cell {{ flex: 1 1 46%; }}
        .figure-desc {{ background: white; border: 1px solid #e2e8f0; border-radius: 6px; padding: 10px 12px 10px 24px; margin: 0; font-size: 0.92rem; }}
        @media (max-width: 600px) {{
            .qr-box {{ flex-direction: column; text-align: center; }}
            .fig-cell {{ flex: 1 1 100%; }}
        }}
    </style>
</head>
<body>
    <div class="app-header">
        <h1>🏥 영상의학 논문 한글 요약기</h1>
        <p>Figure 진단명 자동 표기 · 해부학 영문 병기 · 모바일 웹앱</p>
    </div>

    <div class="container">
        <!-- 스마트폰 연결 안내 카드 -->
        <div class="card" id="qrCard">
            <div class="card-title">
                <span>📱 스마트폰·태블릿에서 바로 열기</span>
                <button onclick="document.getElementById('qrCard').style.display='none'" style="border:none;background:none;color:#64748b;cursor:pointer;font-size:0.82rem;">닫기 ✕</button>
            </div>
            <div class="qr-box">
                <img src="{qr_api_url}" alt="QR Code" onerror="this.style.display='none'">
                <div class="qr-info">
                    1. 스마트폰을 컴퓨터와 <b>같은 와이파이(Wi-Fi)</b>에 연결하세요.<br>
                    2. 스마트폰 카메라로 왼쪽 <b>QR코드</b>를 비추거나 브라우저 주소창에 아래 주소를 입력하세요:<br>
                    <span class="url-badge">{local_url}</span><br>
                    3. 접속 후 브라우저 메뉴에서 <b>[홈 화면에 추가]</b>를 누르면 스마트폰 바탕화면에 전용 앱 아이콘이 생성됩니다!
                </div>
            </div>
        </div>

        <!-- API 키 및 모델 선택 카드 -->
        <div class="card">
            <div class="card-title">
                <span>🤖 AI 모델 선택 및 논문 변환</span>
                <span id="keyStatus" style="font-size:0.8rem; color:#059669; cursor:pointer;" onclick="toggleApiBox()">
                    {"✅ API 키 저장됨 (변경)" if saved_api_key else "🔑 API 키 입력 필요"}
                </span>
            </div>

            <div id="apiBox" style="display:{"none" if saved_api_key else "block"}; margin-bottom:12px;">
                <div class="api-row">
                    <input type="password" id="apiKeyInput" placeholder="Gemini API 키를 입력하세요" value="{saved_api_key}">
                    <button onclick="saveApiKey()">저장</button>
                </div>
            </div>

            <div class="tier-group">
                <div id="tierFlash" class="tier-btn {"active" if saved_tier != "pro" else ""}" onclick="selectTier('flash')">
                    ⚡ Gemini Flash<br><span style="font-size:0.76rem;font-weight:normal;">빠른 요약 · [한글판]</span>
                </div>
                <div id="tierPro" class="tier-btn {"active" if saved_tier == "pro" else ""}" onclick="selectTier('pro')">
                    🧠 Gemini Pro<br><span style="font-size:0.76rem;font-weight:normal;">고정밀 판독 · [한글판프로]</span>
                </div>
            </div>

            <input type="file" id="pdfInput" accept="application/pdf" style="display:none" onchange="handleFileSelect(event)">
            <button class="upload-btn" onclick="document.getElementById('pdfInput').click()">
                📄 PDF 논문 선택 및 요약 시작
            </button>
        </div>

        <!-- 진행 상태 표시 -->
        <div id="statusBox" class="status-box"></div>

        <!-- 결과 출력 구역 -->
        <div id="resultArea"></div>
    </div>

    <script>
        let currentTier = "{saved_tier}";
        let hasSavedKey = {has_key};

        // 모바일 기기에서 접속한 경우 QR 안내 카드는 자동으로 숨김
        if (/Android|iPhone|iPad|iPod/i.test(navigator.userAgent)) {{
            document.getElementById('qrCard').style.display = 'none';
        }}

        function toggleApiBox() {{
            const box = document.getElementById('apiBox');
            box.style.display = box.style.display === 'none' ? 'block' : 'none';
        }}

        function selectTier(tier) {{
            currentTier = tier;
            document.getElementById('tierFlash').className = 'tier-btn' + (tier === 'flash' ? ' active' : '');
            document.getElementById('tierPro').className = 'tier-btn' + (tier === 'pro' ? ' active' : '');
        }}

        async function saveApiKey() {{
            const key = document.getElementById('apiKeyInput').value.trim();
            if (!key) return alert('API 키를 입력해 주세요.');
            const res = await fetch('/api/save_key', {{
                method: 'POST',
                headers: {{'Content-Type': 'application/json'}},
                body: JSON.stringify({{api_key: key, model_tier: currentTier}})
            }});
            if (res.ok) {{
                hasSavedKey = true;
                document.getElementById('keyStatus').innerText = '✅ API 키 저장됨 (변경)';
                document.getElementById('apiBox').style.display = 'none';
                alert('API 키가 영구 저장되었습니다!');
            }}
        }}

        function handleFileSelect(event) {{
            const file = event.target.files[0];
            if (!file) return;

            const key = document.getElementById('apiKeyInput').value.trim();
            if (!key && !hasSavedKey) {{
                alert('먼저 Gemini API 키를 입력하고 저장해 주세요.');
                document.getElementById('apiBox').style.display = 'block';
                return;
            }}

            const statusBox = document.getElementById('statusBox');
            statusBox.style.display = 'block';
            statusBox.innerHTML = '⏳ <b>' + file.name + '</b> 업로드 및 Figure 추출 중...<br><span style="font-size:0.82rem;font-weight:normal;">AI 요약 및 PDF 생성에 약 25~50초가 소요됩니다. 화면을 닫지 말고 잠시만 기다려 주세요.</span>';
            document.getElementById('resultArea').innerHTML = '';

            const reader = new FileReader();
            reader.onload = async function(e) {{
                const base64Data = e.target.result.split(',')[1];
                try {{
                    const response = await fetch('/api/summarize', {{
                        method: 'POST',
                        headers: {{'Content-Type': 'application/json'}},
                        body: JSON.stringify({{
                            filename: file.name,
                            pdf_base64: base64Data,
                            model_tier: currentTier,
                            api_key: key
                        }})
                    }});
                    const data = await response.json();
                    if (!response.ok || data.error) {{
                        throw new Error(data.error || '요약 처리 중 오류가 발생했습니다.');
                    }}
                    statusBox.style.display = 'none';

                    let dlHtml = '';
                    if (data.download_id) {{
                        dlHtml = `
                        <div class="dl-banner">
                            <div style="font-weight:700;color:#065f46;font-size:0.95rem;">
                                ✅ [${{data.used_model}}] 변환 완료! (추출 이미지: ${{data.img_count}}개)
                            </div>
                            <div style="font-size:0.8rem;color:#047857;margin:4px 0;">
                                PC 저장 위치: ${{data.pc_saved_path}}
                            </div>
                            <a class="dl-btn" href="/api/download/${{data.download_id}}" download="${{data.output_filename}}">
                                📥 스마트폰에 ${{data.output_filename}} 저장하기
                            </a>
                        </div>`;
                    }}
                    document.getElementById('resultArea').innerHTML = dlHtml + data.report_html;
                    window.scrollTo({{ top: document.getElementById('resultArea').offsetTop - 70, behavior: 'smooth' }});
                }} catch (err) {{
                    statusBox.style.display = 'block';
                    statusBox.innerHTML = '❌ 오류 발생: ' + err.message;
                }} finally {{
                    event.target.value = '';
                }}
            }};
            reader.readAsDataURL(file);
        }}
    </script>
</body>
</html>"""


# =====================================================================
# 6. 내장 웹 서버 핸들러
# =====================================================================
class MobileWebAppHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # 콘솔 로그 깔끔하게 유지

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/?"):
            cfg = load_config()
            local_ip = get_local_ip()
            port = self.server.server_port
            local_url = f"http://{local_ip}:{port}"
            html = build_mobile_webapp_html(local_url, cfg.get("api_key", ""), cfg.get("model_tier", "flash"))
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html.encode("utf-8"))

        elif self.path == "/manifest.json":
            manifest = {
                "name": "영상의학 논문 한글 요약기",
                "short_name": "논문요약기",
                "start_url": "/",
                "display": "standalone",
                "background_color": "#f1f5f9",
                "theme_color": "#0f2c59",
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(manifest, ensure_ascii=False).encode("utf-8"))

        elif self.path.startswith("/api/download/"):
            dl_id = self.path.split("/api/download/")[-1].split("?")[0]
            info = GENERATED_PDF_STORE.get(dl_id)
            if not info or not os.path.exists(info["path"]):
                self.send_error(404, "File Not Found")
                return
            with open(info["path"], "rb") as f:
                pdf_bytes = f.read()
            encoded_name = urllib.parse.quote(info["filename"])
            self.send_response(200)
            self.send_header("Content-Type", "application/pdf")
            self.send_header("Content-Disposition", f"attachment; filename*=UTF-8''{encoded_name}")
            self.send_header("Content-Length", str(len(pdf_bytes)))
            self.end_headers()
            self.wfile.write(pdf_bytes)
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(length)
        data = json.loads(raw_body.decode("utf-8"))

        if self.path == "/api/save_key":
            cfg = load_config()
            if data.get("api_key"):
                cfg["api_key"] = data["api_key"].strip()
            if data.get("model_tier"):
                cfg["model_tier"] = data["model_tier"]
            save_config(cfg)
            self.send_json({"ok": True})

        elif self.path == "/api/summarize":
            try:
                cfg = load_config()
                api_key = (data.get("api_key") or cfg.get("api_key") or "").strip()
                if not api_key:
                    self.send_json({"error": "API 키가 설정되지 않았습니다."}, status=400)
                    return

                tier = data.get("model_tier", cfg.get("model_tier", "flash"))
                cfg["api_key"] = api_key
                cfg["model_tier"] = tier
                save_config(cfg)

                filename = os.path.basename(data.get("filename", "paper.pdf"))
                pdf_bytes = base64.b64decode(data["pdf_base64"])

                print(f"\n📱 [모바일 요청 수신] {filename} 분석 시작...")
                extracted_text, extracted_images = extract_text_and_figures_from_bytes(pdf_bytes)
                print(f"   ├─ 본문 및 Figure {len(extracted_images)}개 추출 완료")

                client = genai.Client(api_key=api_key)
                model_chain = build_fallback_model_chain(client, preferred_tier=tier)
                summary_result, used_model = summarize_with_gemini(client, model_chain, extracted_text)
                print(f"   ├─ ✅ [{used_model}] AI 요약 완료!")

                prefix_tag = "[한글판프로]" if "pro" in used_model else "[한글판]"
                output_filename = f"{prefix_tag} {filename}"

                save_dir = cfg.get("last_dir", os.path.expanduser("~/Downloads"))
                if not os.path.exists(save_dir):
                    save_dir = os.path.expanduser("~/Downloads")
                output_pdf_path = os.path.normpath(os.path.join(save_dir, output_filename))

                pdf_ok = generate_pdf_file(summary_result, extracted_images, output_pdf_path)
                dl_id = str(int(time.time() * 1000))
                if pdf_ok:
                    GENERATED_PDF_STORE[dl_id] = {"path": output_pdf_path, "filename": output_filename}
                    print(f"   ✅ PDF 생성 및 저장 완료: {output_pdf_path}")

                report_html = build_report_body_html(summary_result, extracted_images, use_data_uri=True)
                self.send_json({
                    "ok": True,
                    "used_model": used_model,
                    "img_count": len(extracted_images),
                    "output_filename": output_filename,
                    "pc_saved_path": output_pdf_path if pdf_ok else "모바일 화면 전용 뷰",
                    "download_id": dl_id if pdf_ok else "",
                    "report_html": report_html,
                })
            except Exception as e:
                print(f"   ❌ 오류 발생: {e}")
                self.send_json({"error": str(e)}, status=500)

    def send_json(self, obj: dict, status: int = 200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_server():
    port = 8080
    server = ThreadingHTTPServer(("0.0.0.0", port), MobileWebAppHandler)
    local_ip = get_local_ip()
    url = f"http://{local_ip}:{port}"

    print("=" * 65)
    print("🏥 [영상의학 논문 한글 요약기 - 모바일 웹앱 서버 가동 완료!]")
    print(f"💻 내 컴퓨터 접속 주소 : http://localhost:{port}")
    print(f"📱 스마트폰 접속 주소  : {url}")
    print("   (스마트폰을 같은 와이파이에 연결한 뒤 위 주소나 QR코드로 접속하세요)")
    print("=" * 65)

    webbrowser.open(f"http://localhost:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run_server()