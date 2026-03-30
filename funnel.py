import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import re
import warnings
from openpyxl import Workbook
from openpyxl.styles import (Font, PatternFill, Alignment, Border, Side,
                              GradientFill)
from openpyxl.utils import get_column_letter
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="📊 Phân Tích TMĐT Online",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Sora:wght@700;800&display=swap');

html, body, [class*="css"], .stApp, .stMarkdown, .stText, button, input, select, textarea {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

.stApp { background: #f4f6fb; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] label {
    color: #64748b !important; font-size: 0.72rem !important;
    text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600 !important;
}
[data-testid="stSidebar"] h3 { color: #f1f5f9 !important; font-size: 0.95rem !important; font-weight: 700 !important; }
[data-testid="stSidebar"] .stSuccess { background: rgba(34,197,94,0.12) !important; border: 1px solid rgba(34,197,94,0.25) !important; }
[data-testid="stSidebar"] .stSuccess * { color: #86efac !important; }
[data-testid="stSidebar"] .stCaption * { color: #64748b !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.08) !important; }

/* ── Upload zone ── */
.upload-zone {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 55%, #0f3460 100%);
    border-radius: 24px; padding: 72px 40px; text-align: center;
    color: white; margin-bottom: 28px;
    border: 1.5px dashed rgba(255,255,255,0.15);
    position: relative; overflow: hidden;
}
.upload-zone::before {
    content: ''; position: absolute; inset: 0;
    background: radial-gradient(ellipse 60% 50% at 70% 30%, rgba(59,130,246,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.upload-zone h1 {
    font-family: 'Sora', sans-serif !important;
    font-size: 2.4rem; font-weight: 800; margin: 0 0 10px 0;
    letter-spacing: -0.02em;
}
.upload-zone p { opacity: 0.6; font-size: 0.93rem; margin: 0; }

/* ── Dashboard header ── */
.dash-header {
    background: linear-gradient(120deg, #0f172a 0%, #1e293b 50%, #0f3460 100%);
    border-radius: 18px; padding: 22px 30px; margin-bottom: 20px; color: white;
    border: 1px solid rgba(255,255,255,0.06);
    box-shadow: 0 4px 24px rgba(0,0,0,0.15);
}
.dash-header h2 {
    font-family: 'Sora', sans-serif !important;
    font-size: 1.4rem; font-weight: 800; margin: 0 0 5px 0; letter-spacing: -0.01em;
}
.dash-header p { opacity: 0.55; font-size: 0.82rem; margin: 0; line-height: 1.6; }

/* ── KPI cards ── */
.kpi-wrap {
    background: white; border-radius: 16px; padding: 18px 20px;
    border-top: 4px solid; box-shadow: 0 1px 12px rgba(0,0,0,0.06);
    height: 100%; transition: all 0.22s ease;
    position: relative; overflow: hidden;
}
.kpi-wrap::after {
    content: ''; position: absolute; right: -10px; bottom: -10px;
    width: 70px; height: 70px; border-radius: 50%;
    background: currentColor; opacity: 0.04;
    pointer-events: none;
}
.kpi-wrap:hover { box-shadow: 0 8px 28px rgba(0,0,0,0.12); transform: translateY(-3px); }
.kpi-icon { font-size: 1.4rem; margin-bottom: 10px; display: block; }
.kpi-label {
    font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.09em; color: #94a3b8; margin-bottom: 5px;
}
.kpi-value { font-size: 1.55rem; font-weight: 800; line-height: 1.1; letter-spacing: -0.02em; }
.kpi-sub { font-size: 0.75rem; color: #94a3b8; margin-top: 6px; }
.kpi-badge {
    display: inline-block; padding: 2px 9px; border-radius: 20px;
    font-size: 0.68rem; font-weight: 700; margin-top: 7px; letter-spacing: 0.02em;
}

/* ── Section titles ── */
.sec-title {
    font-size: 0.95rem; font-weight: 700; color: #0f172a;
    padding: 16px 0 10px 0; border-bottom: 2px solid #e2e8f0; margin-bottom: 16px;
    letter-spacing: -0.01em;
}

/* ── Insight boxes ── */
.insight {
    border-radius: 10px; padding: 12px 16px; margin: 7px 0;
    font-size: 0.85rem; line-height: 1.65; font-weight: 500;
}
.insight-warn  { background: #fffbeb; border-left: 4px solid #f59e0b; color: #78350f; }
.insight-bad   { background: #fff1f2; border-left: 4px solid #f43f5e; color: #881337; }
.insight-good  { background: #f0fdf4; border-left: 4px solid #22c55e; color: #14532d; }
.insight-info  { background: #eff6ff; border-left: 4px solid #3b82f6; color: #1e3a5f; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px; background: #e8ecf5; border-radius: 14px; padding: 5px;
    border: 1px solid #dde3f0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent; border-radius: 10px; padding: 8px 16px;
    font-weight: 600; color: #64748b; border: none !important; font-size: 0.82rem;
    transition: all 0.18s;
}
.stTabs [data-baseweb="tab"]:hover { color: #334155; background: rgba(255,255,255,0.6); }
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1e293b, #0f3460) !important;
    color: white !important; box-shadow: 0 2px 10px rgba(15,52,96,0.28) !important;
}

/* ── Risk items ── */
.risk-item {
    background: white; border-radius: 10px; padding: 10px 14px; margin: 5px 0;
    border-left: 4px solid; font-size: 0.82rem; box-shadow: 0 1px 5px rgba(0,0,0,0.06);
}

/* ── Tab filter bar ── */
.tab-filter-bar {
    background: #f8faff; border-radius: 12px; padding: 14px 18px; margin-bottom: 16px;
    border: 1.5px solid #dbeafe;
}

/* ── Download button styling ── */
.stDownloadButton button {
    background: linear-gradient(135deg, #1e293b, #0f3460) !important;
    color: white !important; border: none !important; border-radius: 10px !important;
    font-weight: 600 !important; font-size: 0.84rem !important;
    padding: 10px 20px !important; transition: all 0.2s !important;
    box-shadow: 0 2px 8px rgba(15,52,96,0.25) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
.stDownloadButton button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 14px rgba(15,52,96,0.35) !important;
}

/* ── Dataframe ── */
.stDataFrame { border-radius: 12px !important; overflow: hidden !important; }

/* ── Feature cards (landing) ── */
.feat-card {
    background: white; border-radius: 16px; padding: 28px 22px; text-align: center;
    box-shadow: 0 2px 16px rgba(0,0,0,0.07); min-height: 190px;
    border: 1px solid #e8ecf5; transition: all 0.22s;
}
.feat-card:hover { box-shadow: 0 8px 28px rgba(0,0,0,0.11); transform: translateY(-2px); }

/* scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #f1f5f9; border-radius: 3px; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #94a3b8; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# HELPERS – UI
# ══════════════════════════════════════════════════════════════════════
def kpi_card(icon, label, value, sub, color, badge_txt="", badge_color=""):
    badge = (f'<div class="kpi-badge" style="background:{badge_color}22;color:{badge_color}">'
             f'{badge_txt}</div>') if badge_txt else ""
    return f"""
    <div class="kpi-wrap" style="border-top-color:{color}">
        <span class="kpi-icon">{icon}</span>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value" style="color:{color}">{value}</div>
        <div class="kpi-sub">{sub}</div>
        {badge}
    </div>"""


# ══════════════════════════════════════════════════════════════════════
# EXCEL EXPORT
# ══════════════════════════════════════════════════════════════════════
def _thin_border():
    t = Side(style='thin', color='E2E8F0')
    return Border(left=t, right=t, top=t, bottom=t)


def _header_cell(ws, row, col, value,
                 bg="1E293B", fg="FFFFFF", size=10, bold=True, center=True, wrap=True):
    c = ws.cell(row=row, column=col, value=value)
    c.font = Font(name="Arial", bold=bold, color=fg, size=size)
    c.fill = PatternFill("solid", start_color=bg)
    c.alignment = Alignment(horizontal="center" if center else "left",
                            vertical="center", wrap_text=wrap)
    c.border = _thin_border()
    return c


def _data_cell(ws, row, col, value, fmt=None, bold=False, bg=None, fg="111827",
               center=False, wrap=False):
    c = ws.cell(row=row, column=col, value=value)
    c.font = Font(name="Arial", bold=bold, color=fg, size=9)
    if bg:
        c.fill = PatternFill("solid", start_color=bg)
    if fmt:
        c.number_format = fmt
    c.alignment = Alignment(horizontal="center" if center else "left",
                            vertical="center", wrap_text=wrap)
    c.border = _thin_border()
    return c


def _set_col_widths(ws, widths):
    for col_letter, w in widths.items():
        ws.column_dimensions[col_letter].width = w


def _freeze(ws, cell="A2"):
    ws.freeze_panes = cell


def _sheet_no_grid(ws):
    ws.sheet_view.showGridLines = False


def export_excel_bytes(filtered_df, sl_df, hoan_df, monthly_df, san_summary_df,
                       report_info=""):
    """Tạo file Excel báo cáo đẹp với nhiều sheet."""
    wb = Workbook()

    DARK   = "1E293B"
    BLUE   = "3B82F6"
    GREEN  = "22C55E"
    RED    = "EF4444"
    PURPLE = "8B5CF6"
    AMBER  = "F59E0B"
    PINK   = "EC4899"
    LGRAY  = "F8FAFF"
    MGRAY  = "EFF2F8"
    DKRED  = "881337"
    DKGRN  = "14532D"

    # ─────────────────────────────────────
    # SHEET 1 – TỔNG QUAN
    # ─────────────────────────────────────
    ws1 = wb.active
    ws1.title = "Tong quan"
    _sheet_no_grid(ws1)
    ws1.row_dimensions[1].height = 14
    ws1.row_dimensions[2].height = 38
    ws1.row_dimensions[3].height = 14

    # Title banner
    ws1.merge_cells("A2:I2")
    tc = ws1["A2"]
    tc.value = "📊  BÁO CÁO PHÂN TÍCH DOANH SỐ THƯƠNG MẠI ĐIỆN TỬ"
    tc.font = Font(name="Arial", bold=True, size=15, color="FFFFFF")
    tc.fill = PatternFill("solid", start_color=DARK)
    tc.alignment = Alignment(horizontal="center", vertical="center")

    # Sub-info row
    ws1.merge_cells("A4:I4")
    si = ws1["A4"]
    si.value = report_info or "Nguồn: File Excel tải lên  |  Xuất từ hệ thống Phân Tích TMĐT"
    si.font = Font(name="Arial", size=9, color="64748B")
    si.alignment = Alignment(horizontal="center", vertical="center")

    # KPI section label
    ws1.row_dimensions[6].height = 20
    ws1.merge_cells("A6:I6")
    kl = ws1["A6"]
    kl.value = "  KẾT QUẢ KINH DOANH TỔNG HỢP"
    kl.font = Font(name="Arial", bold=True, size=10, color=DARK)
    kl.fill = PatternFill("solid", start_color="E8ECF5")
    kl.alignment = Alignment(horizontal="left", vertical="center")

    # KPI data
    tv  = filtered_df['Tien_ve_TK'].sum() if 'Tien_ve_TK' in filtered_df else 0
    gv  = filtered_df['Tong_gia_von'].sum() if 'Tong_gia_von' in filtered_df else 0
    ln  = filtered_df['Loi_nhuan'].sum() if 'Loi_nhuan' in filtered_df else 0
    sl  = filtered_df['So_luong'].sum() if 'So_luong' in filtered_df else 0
    bien = round(ln / tv * 100, 1) if tv else 0.0
    hoan_total = hoan_df['Tong_gia_von'].sum() if len(hoan_df) > 0 else 0
    tang_df = filtered_df[filtered_df['La_hang_tang']] if 'La_hang_tang' in filtered_df.columns else pd.DataFrame()
    tang_gv = tang_df['Tong_gia_von'].sum() if len(tang_df) > 0 else 0

    kpi_rows = [
        ("💰 Tiền Về TK",      tv,        "#,##0₫",  BLUE,   "Dự kiến nhận về tài khoản"),
        ("📦 Tổng Giá Vốn",    gv,        "#,##0₫",  PURPLE, "Chi phí vốn hàng bán"),
        ("📊 Lợi Nhuận",       ln,        "#,##0₫",  GREEN if ln >= 0 else RED, "Lợi nhuận sau giá vốn"),
        ("🛒 Sản Lượng",       int(sl),   "#,##0",   AMBER,  "Tổng sản phẩm bán ra"),
        ("📈 Biên Lợi Nhuận",  bien / 100,"0.0%",    BLUE,   "Tỷ lệ lợi nhuận / tiền về"),
        ("🎁 Chi Phí Hàng Tặng",tang_gv,  "#,##0₫",  PINK,   "Giá vốn hàng tặng kèm"),
        ("↩️ Giá Vốn Hoàn",    hoan_total,"#,##0₫",  RED,    "Giá vốn đơn hàng bị hoàn"),
    ]

    ws1.row_dimensions[7].height = 15
    COLS_K = ['A','B','C','D','E','F','G','H','I']
    # Header row for KPIs
    for ci, h in enumerate(["Chỉ tiêu","Giá trị","Ghi chú"]):
        c = ws1.cell(row=8, column=ci*3+1 if ci == 0 else (5 if ci == 1 else 7))
        # simplified: just use rows 8-onward
    # Clear and rewrite KPIs cleanly
    ws1.row_dimensions[8].height = 18
    _header_cell(ws1, 8, 1, "CHỈ TIÊU", bg=DARK, size=9)
    _header_cell(ws1, 8, 2, "GIÁ TRỊ", bg=DARK, size=9)
    _header_cell(ws1, 8, 3, "GHI CHÚ", bg=DARK, size=9)
    ws1.merge_cells("C8:E8")

    for i, (label, val, fmt, color, note) in enumerate(kpi_rows):
        r = 9 + i
        ws1.row_dimensions[r].height = 20
        row_bg = LGRAY if i % 2 == 0 else "FFFFFF"
        _data_cell(ws1, r, 1, label, bold=True, bg=row_bg, fg=DARK)
        c2 = _data_cell(ws1, r, 2, val, fmt=fmt, bold=True, bg=row_bg, fg=color, center=True)
        ws1.merge_cells(f"C{r}:E{r}")
        _data_cell(ws1, r, 3, note, bg=row_bg, fg="64748B")

    # By-platform table
    start_r = 9 + len(kpi_rows) + 2
    ws1.merge_cells(f"A{start_r}:E{start_r}")
    lbl = ws1.cell(row=start_r, column=1, value="  HIỆU SUẤT THEO SÀN TMĐT")
    lbl.font = Font(name="Arial", bold=True, size=10, color=DARK)
    lbl.fill = PatternFill("solid", start_color="E8ECF5")
    lbl.alignment = Alignment(horizontal="left", vertical="center")
    ws1.row_dimensions[start_r].height = 20

    ph = start_r + 1
    for ci, h in enumerate(["Sàn TMĐT","Tiền Về TK","Tổng GV","Lợi Nhuận","Biên LN%"]):
        _header_cell(ws1, ph, ci + 1, h, bg=DARK, size=9)
    ws1.row_dimensions[ph].height = 18

    if len(san_summary_df) > 0:
        for i, row in san_summary_df.iterrows():
            r = ph + 1 + list(san_summary_df.index).index(i)
            row_bg = LGRAY if r % 2 == 0 else "FFFFFF"
            ws1.row_dimensions[r].height = 18
            _data_cell(ws1, r, 1, row.get('San', ''), bold=True, bg=row_bg, fg=DARK)
            _data_cell(ws1, r, 2, row.get('Tien_ve_TK', 0), fmt="#,##0₫", bg=row_bg, fg=BLUE, center=True)
            _data_cell(ws1, r, 3, row.get('Tong_gia_von', 0), fmt="#,##0₫", bg=row_bg, fg=PURPLE, center=True)
            ln_v = row.get('Loi_nhuan', 0)
            _data_cell(ws1, r, 4, ln_v, fmt="#,##0₫", bg=row_bg,
                       fg=GREEN if ln_v >= 0 else RED, center=True)
            bien_v = row.get('Bien_LN', 0) / 100 if 'Bien_LN' in row else 0
            _data_cell(ws1, r, 5, bien_v, fmt="0.0%", bg=row_bg, center=True,
                       fg=GREEN if bien_v >= 0 else RED)

    _set_col_widths(ws1, {'A': 28, 'B': 18, 'C': 18, 'D': 18, 'E': 14,
                          'F': 2, 'G': 2, 'H': 2, 'I': 2})
    _freeze(ws1, "A9")

    # ─────────────────────────────────────
    # SHEET 2 – CHI TIẾT GIAO DỊCH
    # ─────────────────────────────────────
    ws2 = wb.create_sheet("Chi tiet")
    _sheet_no_grid(ws2)
    _freeze(ws2, "A3")

    ws2.merge_cells("A1:K1")
    tc2 = ws2["A1"]
    tc2.value = "📋  CHI TIẾT GIAO DỊCH"
    tc2.font = Font(name="Arial", bold=True, size=12, color="FFFFFF")
    tc2.fill = PatternFill("solid", start_color=DARK)
    tc2.alignment = Alignment(horizontal="center", vertical="center")
    ws2.row_dimensions[1].height = 30

    col_defs = [
        ("Sàn",           'San',           None,        12),
        ("Tháng",         'Thang_label',   None,        14),
        ("Tên Hàng Hóa",  'Ten_hang_hoa',  None,        40),
        ("Khách Hàng",    'Khach_hang',    None,        22),
        ("Số Lượng",      'So_luong',      "#,##0",     12),
        ("Giá Vốn/SP",    'Gia_von',       "#,##0₫",    14),
        ("Tổng Giá Vốn",  'Tong_gia_von',  "#,##0₫",    16),
        ("Tiền Về TK",    'Tien_ve_TK',    "#,##0₫",    16),
        ("Lợi Nhuận",     'Loi_nhuan',     "#,##0₫",    16),
        ("Loại Hàng",     'La_hang_tang',  None,        12),
        ("Trạng Thái",    'Trang_thai',    None,        14),
    ]

    for ci, (h, _, _, _) in enumerate(col_defs):
        _header_cell(ws2, 2, ci + 1, h, bg=DARK, size=9)
    ws2.row_dimensions[2].height = 22

    for ri, (_, row) in enumerate(filtered_df.iterrows()):
        r = ri + 3
        ws2.row_dimensions[r].height = 16
        row_bg = LGRAY if ri % 2 == 0 else "FFFFFF"
        for ci, (_, field, fmt, _) in enumerate(col_defs):
            val = row.get(field, "")
            if field == 'La_hang_tang':
                val = "🎁 Tặng" if val else "✅ Bán"
            fg_clr = "111827"
            if field == 'Loi_nhuan':
                fg_clr = DKGRN if (val or 0) >= 0 else DKRED
            _data_cell(ws2, r, ci + 1, val, fmt=fmt, bg=row_bg, fg=fg_clr,
                       center=(field in ['So_luong','La_hang_tang','Trang_thai']))

    for ci, (_, _, _, w) in enumerate(col_defs):
        ws2.column_dimensions[get_column_letter(ci + 1)].width = w

    # ─────────────────────────────────────
    # SHEET 3 – THEO SÀN
    # ─────────────────────────────────────
    ws3 = wb.create_sheet("Theo san")
    _sheet_no_grid(ws3)

    ws3.merge_cells("A1:G1")
    tc3 = ws3["A1"]
    tc3.value = "🏪  HIỆU SUẤT THEO SÀN TMĐT"
    tc3.font = Font(name="Arial", bold=True, size=12, color="FFFFFF")
    tc3.fill = PatternFill("solid", start_color=DARK)
    tc3.alignment = Alignment(horizontal="center", vertical="center")
    ws3.row_dimensions[1].height = 30

    san_cols = [
        ("Sàn TMĐT",     'San',        None,     18),
        ("Tiền Về TK",   'Tien_ve_TK', "#,##0₫", 18),
        ("Tổng GV",      'Tong_gia_von',"#,##0₫", 18),
        ("Lợi Nhuận",    'Loi_nhuan',  "#,##0₫", 18),
        ("Biên LN%",     'Bien_LN',    "0.0%",   12),
        ("Sản Lượng",    'So_luong',   "#,##0",  14),
        ("Số Dòng SP",   'So_dong',    "#,##0",  12),
    ]
    for ci, (h, _, _, _) in enumerate(san_cols):
        _header_cell(ws3, 2, ci + 1, h, bg=DARK, size=9)
    ws3.row_dimensions[2].height = 22

    if len(san_summary_df) > 0:
        for ri, (_, row) in enumerate(san_summary_df.iterrows()):
            r = ri + 3
            ws3.row_dimensions[r].height = 20
            row_bg = LGRAY if ri % 2 == 0 else "FFFFFF"
            for ci, (_, field, fmt, _) in enumerate(san_cols):
                val = row.get(field, 0)
                if field == 'Bien_LN':
                    val = (val or 0) / 100
                fg_clr = "111827"
                if field in ('Loi_nhuan', 'Bien_LN'):
                    fg_clr = DKGRN if (val or 0) >= 0 else DKRED
                elif field == 'San':
                    fg_clr = DARK
                _data_cell(ws3, r, ci + 1, val, fmt=fmt, bg=row_bg, fg=fg_clr,
                           bold=(field == 'San'),
                           center=(field != 'San' and field != 'Ten_hang_hoa'))

    for ci, (_, _, _, w) in enumerate(san_cols):
        ws3.column_dimensions[get_column_letter(ci + 1)].width = w
    _freeze(ws3, "A3")

    # ─────────────────────────────────────
    # SHEET 4 – THEO THÁNG
    # ─────────────────────────────────────
    if monthly_df is not None and len(monthly_df) > 0:
        ws4 = wb.create_sheet("Theo thang")
        _sheet_no_grid(ws4)

        ws4.merge_cells("A1:H1")
        tc4 = ws4["A1"]
        tc4.value = "📅  PHÂN TÍCH THEO THÁNG"
        tc4.font = Font(name="Arial", bold=True, size=12, color="FFFFFF")
        tc4.fill = PatternFill("solid", start_color=DARK)
        tc4.alignment = Alignment(horizontal="center", vertical="center")
        ws4.row_dimensions[1].height = 30

        m_cols = [
            ("Tháng",       'Thang_label', None,     16),
            ("Tiền Về TK",  'Tien_ve_TK',  "#,##0₫", 18),
            ("Tổng GV",     'Tong_gia_von', "#,##0₫", 18),
            ("Lợi Nhuận",   'Loi_nhuan',   "#,##0₫", 18),
            ("Biên LN%",    'Bien_LN',     "0.0%",   12),
            ("Sản Lượng",   'So_luong',    "#,##0",  14),
            ("ΔDT vs Tháng Trước", None,   "0.0%",   16),
            ("ΔLN vs Tháng Trước", None,   "0.0%",   16),
        ]
        for ci, (h, _, _, _) in enumerate(m_cols):
            _header_cell(ws4, 2, ci + 1, h, bg=DARK, size=9)
        ws4.row_dimensions[2].height = 22

        df_m = monthly_df.sort_values('Thang_key').reset_index(drop=True)
        for ri, row in df_m.iterrows():
            r = ri + 3
            ws4.row_dimensions[r].height = 20
            row_bg = LGRAY if ri % 2 == 0 else "FFFFFF"

            prev_tv = df_m.iloc[ri - 1]['Tien_ve_TK'] if ri > 0 else None
            prev_ln = df_m.iloc[ri - 1]['Loi_nhuan']  if ri > 0 else None
            delta_tv = (row['Tien_ve_TK'] - prev_tv) / abs(prev_tv) if prev_tv else None
            delta_ln = (row['Loi_nhuan']  - prev_ln) / abs(prev_ln) if prev_ln else None

            row_vals = [
                row.get('Thang_label', ''),
                row.get('Tien_ve_TK', 0),
                row.get('Tong_gia_von', 0),
                row.get('Loi_nhuan', 0),
                (row.get('Bien_LN', 0) or 0) / 100,
                row.get('So_luong', 0),
                delta_tv,
                delta_ln,
            ]
            for ci, (_, field, fmt, _) in enumerate(m_cols):
                val = row_vals[ci]
                fg_clr = "111827"
                if field == 'Loi_nhuan' or ci in (4, 6, 7):
                    fg_clr = DKGRN if (val or 0) >= 0 else DKRED
                elif field == 'Thang_label':
                    fg_clr = DARK
                _data_cell(ws4, r, ci + 1, val, fmt=fmt, bg=row_bg, fg=fg_clr,
                           bold=(ci == 0), center=(ci > 0))

        for ci, (_, _, _, w) in enumerate(m_cols):
            ws4.column_dimensions[get_column_letter(ci + 1)].width = w
        _freeze(ws4, "A3")

    # ─────────────────────────────────────
    # SHEET 5 – HOÀN HÀNG
    # ─────────────────────────────────────
    if len(hoan_df) > 0:
        ws5 = wb.create_sheet("Hoan hang")
        _sheet_no_grid(ws5)

        ws5.merge_cells("A1:F1")
        tc5 = ws5["A1"]
        tc5.value = "↩️  CHI TIẾT ĐƠN HÀNG HOÀN"
        tc5.font = Font(name="Arial", bold=True, size=12, color="FFFFFF")
        tc5.fill = PatternFill("solid", start_color="7F1D1D")
        tc5.alignment = Alignment(horizontal="center", vertical="center")
        ws5.row_dimensions[1].height = 30

        h_cols = [
            ("Sàn",          'San',         None,     12),
            ("Tên Hàng",     'Ten_hang',    None,     36),
            ("Số Lượng",     'So_luong',    "#,##0",  12),
            ("Giá Vốn/SP",   'Gia_von',     "#,##0₫", 14),
            ("Tổng GV Hoàn", 'Tong_gia_von',"#,##0₫", 16),
            ("Tháng",        'Thang_label', None,     14),
        ]
        for ci, (h, _, _, _) in enumerate(h_cols):
            _header_cell(ws5, 2, ci + 1, h, bg="7F1D1D", size=9)
        ws5.row_dimensions[2].height = 22

        for ri, (_, row) in enumerate(hoan_df.iterrows()):
            r = ri + 3
            ws5.row_dimensions[r].height = 16
            row_bg = "FFF1F2" if ri % 2 == 0 else "FFFFFF"
            for ci, (_, field, fmt, _) in enumerate(h_cols):
                val = row.get(field, "")
                _data_cell(ws5, r, ci + 1, val, fmt=fmt, bg=row_bg,
                           fg=(DKRED if field == 'Tong_gia_von' else "111827"),
                           bold=(field == 'Tong_gia_von'),
                           center=(field in ['So_luong', 'San']))

        # Summary footer
        last_r = len(hoan_df) + 4
        ws5.merge_cells(f"A{last_r}:D{last_r}")
        _data_cell(ws5, last_r, 1, "TỔNG CỘNG", bold=True, bg="FEE2E2", fg=DKRED, center=True)
        _data_cell(ws5, last_r, 5, hoan_df['Tong_gia_von'].sum(),
                   fmt="#,##0₫", bold=True, bg="FEE2E2", fg=DKRED, center=True)

        for ci, (_, _, _, w) in enumerate(h_cols):
            ws5.column_dimensions[get_column_letter(ci + 1)].width = w
        _freeze(ws5, "A3")

    # ─────────────────────────────────────
    # SHEET 6 – SẢN PHẨM TOP
    # ─────────────────────────────────────
    if len(sl_df) > 0:
        ws6 = wb.create_sheet("Top san pham")
        _sheet_no_grid(ws6)

        ws6.merge_cells("A1:G1")
        tc6 = ws6["A1"]
        tc6.value = "🏆  TOP SẢN PHẨM THEO SẢN LƯỢNG"
        tc6.font = Font(name="Arial", bold=True, size=12, color="FFFFFF")
        tc6.fill = PatternFill("solid", start_color="1D4ED8")
        tc6.alignment = Alignment(horizontal="center", vertical="center")
        ws6.row_dimensions[1].height = 30

        sp_cols = [
            ("Thương Hiệu", 'Thuong_hieu', None,    18),
            ("Mã SP",       'Ma_sp',       None,    14),
            ("Tên Hàng",    'Ten_hang',    None,    40),
            ("TiktokShop",  'Tiktok',      "#,##0", 14),
            ("Shopee",      'Shopee',      "#,##0", 14),
            ("Lazada",      'Lazada',      "#,##0", 14),
            ("Tổng SL",     'Tong_SL',     "#,##0", 14),
        ]
        for ci, (h, _, _, _) in enumerate(sp_cols):
            _header_cell(ws6, 2, ci + 1, h, bg="1D4ED8", size=9)
        ws6.row_dimensions[2].height = 22

        top_sl = sl_df[~sl_df.get('La_hang_tang', False)].sort_values('Tong_SL', ascending=False).head(50)
        for ri, (_, row) in enumerate(top_sl.iterrows()):
            r = ri + 3
            ws6.row_dimensions[r].height = 16
            row_bg = LGRAY if ri % 2 == 0 else "FFFFFF"
            for ci, (_, field, fmt, _) in enumerate(sp_cols):
                val = row.get(field, "")
                is_num = field in ('Tiktok', 'Shopee', 'Lazada', 'Tong_SL')
                bold_col = field in ('Ten_hang', 'Tong_SL')
                fg_clr = BLUE if field == 'Tong_SL' else ("111827" if not is_num else "374151")
                _data_cell(ws6, r, ci + 1, val, fmt=fmt, bg=row_bg, fg=fg_clr,
                           bold=bold_col, center=is_num)

        for ci, (_, _, _, w) in enumerate(sp_cols):
            ws6.column_dimensions[get_column_letter(ci + 1)].width = w
        _freeze(ws6, "A3")

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════
# DATA PARSING HELPERS
# ══════════════════════════════════════════════════════════════════════
def extract_month_from_filename(filename):
    name = filename.lower()
    name = re.sub(r'\.(xlsx?|xls|csv)$', '', name)
    yr_default = "2024"

    def yr(s):
        m = re.search(r'(20\d{2})', s)
        return m.group(1) if m else yr_default

    def result(month_int, year_str):
        return f"{year_str}-{month_int:02d}", f"Tháng {month_int}/{year_str}"

    m = re.search(r'(20\d{2})[-_\.](0?[1-9]|1[0-2])\b', name)
    if m: return result(int(m.group(2)), m.group(1))
    m = re.search(r'\b(0?[1-9]|1[0-2])[-_](20\d{2})\b', name)
    if m: return result(int(m.group(1)), m.group(2))
    m = re.search(r'th[aá]ng\s*0?(\d{1,2})', name)
    if m:
        mo = int(m.group(1))
        if 1 <= mo <= 12: return result(mo, yr(name))
    m = re.search(r'(?<![a-z])t0?(\d{1,2})(?![a-z0-9])', name)
    if m:
        mo = int(m.group(1))
        if 1 <= mo <= 12: return result(mo, yr(name))
    month_map = {
        'january':1,'february':2,'march':3,'april':4,'may':5,'june':6,
        'july':7,'august':8,'september':9,'october':10,'november':11,'december':12,
        'jan':1,'feb':2,'mar':3,'apr':4,'jun':6,'jul':7,'aug':8,
        'sep':9,'oct':10,'nov':11,'dec':12
    }
    for word, num in sorted(month_map.items(), key=lambda x: -len(x[0])):
        if re.search(r'\b' + word + r'\b', name): return result(num, yr(name))
    m = re.search(r'\b(0?[1-9]|1[0-2])\b', name)
    if m:
        mo = int(m.group(1))
        return result(mo, yr(name))
    return None, filename


def parse_sales_sheet(df_raw, san_label):
    header_row = None
    for i, row in df_raw.iterrows():
        vals = str(row.values)
        if 'STT' in vals and 'Tên hàng hoá' in vals:
            header_row = i
            break
    if header_row is None: return pd.DataFrame()
    data = df_raw.iloc[header_row + 1:].copy()
    data.columns = range(len(data.columns))
    col_map = {0:'STT',1:'Khach_hang',2:'Ma_don_hang',3:'Ma_san_pham',
               4:'Ten_hang_hoa',5:'So_luong',6:'Gia_von',7:'Tong_gia_von',
               8:'Tien_ve_TK',9:'Doanh_thu_thuc',10:'Ngay',11:'Loi_nhuan',12:'Trang_thai'}
    cols = {k: v for k, v in col_map.items() if k < len(data.columns)}
    df = data[list(cols.keys())].copy()
    df.columns = list(cols.values())
    df = df[df['Ten_hang_hoa'].notna() &
            (df['Ten_hang_hoa'].astype(str).str.strip() != '') &
            (df['Ten_hang_hoa'].astype(str) != 'nan') &
            (df['Ten_hang_hoa'].astype(str).str.strip().str.len() > 2)]
    df['San'] = san_label
    df['La_hang_tang'] = df['Ma_san_pham'].astype(str).str.endswith('-T', na=False)
    for col in ['So_luong','Gia_von','Tong_gia_von','Tien_ve_TK','Loi_nhuan']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'Ngay' in df.columns:
        df['Ngay'] = pd.to_datetime(df['Ngay'], errors='coerce')
    df = df[~df['Ten_hang_hoa'].astype(str).str.upper().str.contains('TỔNG|TOTAL', na=False)]
    return df


def parse_sl_sheet(all_sheets):
    key = next((k for k in all_sheets if 'SL' in k.upper() or 'TỔNG' in k.upper()), None)
    if not key: return pd.DataFrame()
    df = all_sheets[key]
    sl = df.iloc[4:].copy()
    sl.columns = range(len(sl.columns))
    rename = {0:'STT',1:'Thuong_hieu',2:'Ma_sp',4:'Ten_hang',
              5:'Tiktok',6:'Shopee',7:'Lazada',8:'Tong_SL',9:'Gia_von'}
    sl = sl.rename(columns={k: v for k, v in rename.items() if k < len(sl.columns)})
    if 'Ten_hang' not in sl.columns: return pd.DataFrame()
    sl['Tong_SL'] = pd.to_numeric(sl.get('Tong_SL', 0), errors='coerce').fillna(0)
    sl['Gia_von'] = pd.to_numeric(sl.get('Gia_von', 0), errors='coerce').fillna(0)
    for c in ['Tiktok','Shopee','Lazada']:
        if c in sl.columns:
            sl[c] = pd.to_numeric(sl[c], errors='coerce').fillna(0)
    sl = sl[sl['Ten_hang'].notna() & (sl['Ten_hang'].astype(str).str.strip() != '') &
            (sl['Ten_hang'].astype(str) != 'nan')]
    sl = sl[sl['Tong_SL'] > 0]
    sl = sl[~sl['STT'].astype(str).str.upper().str.contains('TỔNG|TOTAL', na=False)]
    sl['La_hang_tang'] = sl['Ten_hang'].astype(str).str.contains('Hàng tặng|hàng tặng', na=False)
    return sl


def parse_hoan_hang(all_sheets):
    key = next((k for k in all_sheets if 'HOÀN' in k.upper() or 'HOAN' in k.upper()), None)
    if not key: return pd.DataFrame()
    df = all_sheets[key]
    header_row = None
    for i, row in df.iterrows():
        vals = str(row.values)
        if 'STT' in vals and 'Tên hàng hoá' in vals:
            header_row = i
            break
    if header_row is None: return pd.DataFrame()
    data = df.iloc[header_row + 1:].copy()
    data.columns = range(len(data.columns))
    col_map = {0:'San',1:'STT',2:'Khach_hang',3:'Ma_don',4:'Ma_sp',
               5:'Ten_hang',6:'So_luong',7:'Gia_von',8:'Tong_gia_von'}
    cols = {k: v for k, v in col_map.items() if k < len(data.columns)}
    df2 = data[list(cols.keys())].copy()
    df2.columns = list(cols.values())
    df2 = df2[df2['Ten_hang'].notna() & (df2['Ten_hang'].astype(str).str.strip() != '')]
    df2['Tong_gia_von'] = pd.to_numeric(df2['Tong_gia_von'], errors='coerce').fillna(0)
    df2['So_luong']     = pd.to_numeric(df2['So_luong'],     errors='coerce').fillna(0)
    df2['San'] = df2['San'].replace('', np.nan).ffill()
    df2 = df2[df2['Tong_gia_von'] > 0]
    return df2


@st.cache_data
def load_and_parse(file_bytes, filename):
    buf = io.BytesIO(file_bytes)
    all_sheets = pd.read_excel(buf, sheet_name=None, header=None)
    san_map = {}
    for s in all_sheets:
        su = s.upper()
        if 'TIKTOK' in su:   san_map['TiktokShop'] = s
        elif 'SHOPEE' in su: san_map['Shopee'] = s
        elif 'LAZADA' in su: san_map['Lazada'] = s
    month_key, month_label = extract_month_from_filename(filename)
    frames = []
    for label, sheet_name in san_map.items():
        df = parse_sales_sheet(all_sheets[sheet_name], label)
        if len(df) > 0:
            df['Ky_bao_cao']  = filename
            df['Thang_key']   = month_key
            df['Thang_label'] = month_label
            frames.append(df)
    all_data  = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    sl_data   = parse_sl_sheet(all_sheets)
    hoan_data = parse_hoan_hang(all_sheets)
    for d in [sl_data, hoan_data]:
        if len(d) > 0:
            d['Ky_bao_cao']  = filename
            d['Thang_key']   = month_key
            d['Thang_label'] = month_label
    return all_data, sl_data, hoan_data, list(san_map.keys()), list(all_sheets.keys()), month_key, month_label


def load_multiple_files(uploaded_files):
    all_data_list, sl_list, hoan_list = [], [], []
    all_sans  = set()
    errors    = []
    month_map = {}
    for uf in uploaded_files:
        try:
            fb = uf.read()
            d, s, h, sans, _, mk, ml = load_and_parse(fb, uf.name)
            if len(d) > 0:
                all_data_list.append(d)
                all_sans.update(sans)
            if len(s) > 0: sl_list.append(s)
            if len(h) > 0: hoan_list.append(h)
            if mk: month_map[mk] = ml
        except Exception as e:
            errors.append(f"❌ {uf.name}: {e}")
    all_data  = pd.concat(all_data_list, ignore_index=True) if all_data_list else pd.DataFrame()
    sl_data   = pd.concat(sl_list,       ignore_index=True) if sl_list       else pd.DataFrame()
    hoan_data = pd.concat(hoan_list,     ignore_index=True) if hoan_list     else pd.DataFrame()
    sorted_keys    = sorted(month_map.keys())
    ordered_labels = [month_map[k] for k in sorted_keys]
    label_to_key   = {v: k for k, v in month_map.items()}
    return all_data, sl_data, hoan_data, sorted(all_sans), errors, sorted_keys, ordered_labels, label_to_key


# ══════════════════════════════════════════════════════════════════════
# SIDEBAR – FILE UPLOAD
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 📂 Tải lên dữ liệu")
    uploaded_files = st.file_uploader(
        "Kéo thả file Excel vào đây",
        type=["xlsx","xls"],
        accept_multiple_files=True,
        help="Mỗi file = 1 tháng. Đặt tên file chứa tháng, vd: thang3.xlsx, T04-2024.xlsx"
    )
    st.markdown("---")


# ══════════════════════════════════════════════════════════════════════
# LANDING PAGE
# ══════════════════════════════════════════════════════════════════════
if not uploaded_files:
    st.markdown("""
    <div class="upload-zone">
        <div style="font-size:4rem;margin-bottom:20px;filter:drop-shadow(0 4px 12px rgba(59,130,246,0.4))">📊</div>
        <h1>Phân Tích Doanh Số TMĐT</h1>
        <p style="font-size:1.05rem;margin-top:10px">TiktokShop &nbsp;·&nbsp; Shopee &nbsp;·&nbsp; Lazada</p>
        <br>
        <p>👈 <strong>Kéo thả file Excel</strong> vào ô bên trái để bắt đầu</p>
        <p style="margin-top:8px;font-size:0.83rem;opacity:0.45">
            Hỗ trợ nhiều file cùng lúc · Mỗi file = 1 tháng · Xuất báo cáo Excel đẹp
        </p>
        <p style="margin-top:6px;font-size:0.78rem;opacity:0.35">
            Tên file gợi ý: <code>thang3.xlsx</code> &nbsp;·&nbsp; <code>T04-2024.xlsx</code>
            &nbsp;·&nbsp; <code>thang 5 2024.xlsx</code>
        </p>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, icon, title, desc in zip(
        [c1, c2, c3, c4],
        ["🏪", "📅", "💸", "📥"],
        ["Hiệu suất Sàn", "Phân tích Tháng", "Lợi nhuận & Rủi ro", "Xuất Excel Đẹp"],
        [
            "So sánh doanh thu, giá vốn, lợi nhuận giữa 3 sàn TMĐT",
            "Trend line, grouped bar tháng × sàn, pivot table, biên LN%",
            "Waterfall dòng tiền, ma trận rủi ro, đơn hoàn hàng",
            "Báo cáo Excel nhiều sheet, format chuyên nghiệp, có thể in",
        ]
    ):
        with col:
            st.markdown(f"""
            <div class="feat-card">
                <div style="font-size:2.2rem;margin-bottom:14px">{icon}</div>
                <div style="font-weight:700;font-size:0.93rem;margin-bottom:8px;color:#0f172a">{title}</div>
                <div style="font-size:0.79rem;color:#64748b;line-height:1.55">{desc}</div>
            </div>""", unsafe_allow_html=True)
    st.stop()


# ══════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════
with st.spinner("⏳ Đang đọc và phân tích dữ liệu..."):
    (all_data, sl_data, hoan_data,
     available_sans, load_errors,
     sorted_month_keys, ordered_month_labels, label_to_key) = load_multiple_files(uploaded_files)

for err in load_errors:
    st.warning(err)

if len(all_data) == 0:
    st.error("❌ Không tìm thấy dữ liệu giao dịch.")
    st.stop()

file_label = (f"{len(uploaded_files)} file" if len(uploaded_files) > 1
              else uploaded_files[0].name)
has_months = len(sorted_month_keys) > 0


# ══════════════════════════════════════════════════════════════════════
# SIDEBAR FILTERS
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🎛️ Bộ lọc toàn cục")
    san_filter = st.multiselect("🏪 Sàn TMĐT", options=available_sans, default=available_sans)

    if has_months:
        sb_month_sel = st.multiselect(
            "📅 Tháng", options=ordered_month_labels, default=ordered_month_labels,
            help="Lọc toàn bộ dashboard."
        )
        sb_month_keys = [label_to_key[l] for l in sb_month_sel if l in label_to_key]
    else:
        sb_month_sel  = []
        sb_month_keys = []
        st.caption("⚠️ Không nhận dạng được tháng.\nĐặt tên file như: thang3.xlsx")

    include_tang = st.checkbox("🎁 Bao gồm hàng tặng", value=False)
    show_profit  = st.checkbox("✅ Hiển thị hàng có lãi", value=True)
    show_loss    = st.checkbox("❌ Hiển thị hàng thua lỗ", value=True)

    if 'So_luong' in all_data.columns and not all_data['So_luong'].isna().all():
        max_sl   = int(all_data['So_luong'].max())
        sl_range = st.slider("📦 Sản lượng / dòng SP", 0, max(max_sl,1), (0, max(max_sl,1)))
    else:
        sl_range = (0, 99999)

    all_brands = []
    if len(sl_data) > 0 and 'Thuong_hieu' in sl_data.columns:
        brands = sl_data['Thuong_hieu'].ffill().dropna().unique().tolist()
        pn = {'tiktok shop','shopee','lazada','tiktokshop'}
        all_brands = [b for b in brands
                      if str(b).strip().lower() not in pn
                      and str(b).strip() not in ['','nan','Thương hiệu']]
    brand_filter = st.multiselect("🏷️ Thương hiệu", options=all_brands) if all_brands else []

    st.markdown("---")
    for fn in uploaded_files:
        st.success(f"✅ {fn.name}")
    st.caption(f"Sàn: {', '.join(available_sans)}")
    st.caption(f"Tổng: {len(all_data):,} dòng SP")
    if has_months:
        st.caption("📅 " + " · ".join(ordered_month_labels))


# ─── Apply filters ────────────────────────────────────────────────────
def apply_filters(df, month_keys=None, sans=None, tang=None, profit=None, loss=None):
    d = df.copy()
    mk = month_keys if month_keys is not None else sb_month_keys
    if mk and 'Thang_key' in d.columns:
        d = d[d['Thang_key'].isin(mk)]
    s = sans if sans is not None else san_filter
    if s: d = d[d['San'].isin(s)]
    t = tang if tang is not None else include_tang
    if not t: d = d[~d['La_hang_tang']]
    p  = profit if profit is not None else show_profit
    lo = loss   if loss   is not None else show_loss
    if not (p and lo):
        if p:   d = d[d['Loi_nhuan'] >= 0]
        elif lo: d = d[d['Loi_nhuan'] < 0]
    if 'So_luong' in d.columns:
        d = d[d['So_luong'].between(sl_range[0], sl_range[1], inclusive='both') |
              d['So_luong'].isna()]
    return d


filtered   = apply_filters(all_data)
tang_items = apply_filters(all_data, tang=True)[lambda x: x['La_hang_tang']]

total_gia_von   = filtered['Tong_gia_von'].sum()
total_tien_ve   = filtered['Tien_ve_TK'].sum()
total_loi_nhuan = filtered['Loi_nhuan'].sum()
total_sl        = filtered['So_luong'].sum()
total_tang_cost = tang_items['Tong_gia_von'].sum()
total_hoan      = hoan_data['Tong_gia_von'].sum() if len(hoan_data) > 0 else 0
profit_margin   = total_loi_nhuan / total_tien_ve * 100 if total_tien_ve > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════
months_active = " · ".join(sb_month_sel) if sb_month_sel else "Tất cả"
st.markdown(f"""
<div class="dash-header">
  <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px">
    <div>
      <h2>🛍️ Phân Tích Doanh Số Bán Hàng Online</h2>
      <p>📁 {file_label} &nbsp;|&nbsp; Sàn: {" · ".join(san_filter or ["—"])}
         &nbsp;|&nbsp; 📅 {months_active}
         &nbsp;|&nbsp; {"Có hàng tặng" if include_tang else "Không tính hàng tặng"}</p>
    </div>
    <div style="opacity:0.55;font-size:0.78rem;text-align:right;line-height:1.7">
        Hiển thị <b style="color:#93c5fd">{len(filtered):,}</b> dòng SP
    </div>
  </div>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# KPI ROW
# ══════════════════════════════════════════════════════════════════════
c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    st.markdown(kpi_card("💰","Tiền Về TK",
                         f"{total_tien_ve/1e6:.2f}M₫","Dự kiến nhận về","#3b82f6"), unsafe_allow_html=True)
with c2:
    st.markdown(kpi_card("📦","Tổng Giá Vốn",
                         f"{total_gia_von/1e6:.2f}M₫","Giá vốn hàng bán","#8b5cf6"), unsafe_allow_html=True)
with c3:
    clr = "#22c55e" if total_loi_nhuan >= 0 else "#ef4444"
    st.markdown(kpi_card("📊","Lợi Nhuận",f"{total_loi_nhuan/1e6:.2f}M₫","Sau giá vốn",clr,
                         f"Biên LN: {profit_margin:.1f}%","#22c55e" if profit_margin>=0 else "#ef4444"),
                unsafe_allow_html=True)
with c4:
    st.markdown(kpi_card("🛒","Sản Lượng",
                         f"{int(total_sl):,}","Tổng SP bán ra","#f59e0b"), unsafe_allow_html=True)
with c5:
    st.markdown(kpi_card("🎁","Hàng Tặng",
                         f"{total_tang_cost/1e6:.2f}M₫","Chi phí giá vốn tặng","#ec4899"), unsafe_allow_html=True)
with c6:
    st.markdown(kpi_card("↩️","Hàng Hoàn",
                         f"{total_hoan/1e6:.2f}M₫",f"{len(hoan_data)} dòng","#ef4444"), unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

CHART_THEME = dict(
    plot_bgcolor='white', paper_bgcolor='white',
    font=dict(family='Plus Jakarta Sans, sans-serif', size=11),
    margin=dict(t=50, b=30, l=10, r=10)
)
COLORS_SAN = {'TiktokShop':'#fe2c55','Shopee':'#f97316','Lazada':'#6366f1'}


# ══════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab_month, tab4, tab5 = st.tabs([
    "🏪 Hiệu suất Sàn", "📦 Sản phẩm", "💸 Lợi nhuận",
    "📅 Theo Tháng", "↩️ Hoàn hàng & Rủi ro", "🔍 Dữ liệu & Xuất"
])


def tab_month_filter(tab_key, label=""):
    if not has_months: return sb_month_keys
    with st.expander(f"🔽 Lọc tháng riêng cho tab này (mặc định theo Sidebar)", expanded=False):
        st.markdown('<div class="tab-filter-bar">', unsafe_allow_html=True)
        sel = st.multiselect(
            f"📅 Tháng{' — ' + label if label else ''}",
            options=ordered_month_labels, default=sb_month_sel, key=tab_key
        )
        st.caption("Để trống = dùng filter tháng từ Sidebar.")
        st.markdown('</div>', unsafe_allow_html=True)
    keys = [label_to_key[l] for l in sel if l in label_to_key]
    return keys if keys else sb_month_keys


# ════════════════════════════════════════
# TAB 1 – HIỆU SUẤT SÀN
# ════════════════════════════════════════
with tab1:
    t1_keys = tab_month_filter('t1_mf', 'Hiệu suất Sàn')
    f1 = apply_filters(all_data, month_keys=t1_keys)

    st.markdown('<div class="sec-title">🏪 So Sánh Hiệu Suất Các Sàn TMĐT</div>', unsafe_allow_html=True)
    ps = f1.groupby('San').agg(
        Tong_gia_von=('Tong_gia_von','sum'), Tien_ve_TK=('Tien_ve_TK','sum'),
        Loi_nhuan=('Loi_nhuan','sum'), So_luong=('So_luong','sum'),
        So_dong=('Ten_hang_hoa','count')
    ).reset_index()
    ps['Bien_LN'] = ps.apply(
        lambda r: round(r['Loi_nhuan']/r['Tien_ve_TK']*100, 1) if r['Tien_ve_TK']>0 else 0.0, axis=1)

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_bar(name='Giá vốn', x=ps['San'], y=ps['Tong_gia_von'], marker_color='#cbd5e1',
                    text=(ps['Tong_gia_von']/1e6).round(2), texttemplate='%{text:.2f}M₫', textposition='outside')
        fig.add_bar(name='Tiền về TK', x=ps['San'], y=ps['Tien_ve_TK'], marker_color='#3b82f6',
                    text=(ps['Tien_ve_TK']/1e6).round(2), texttemplate='%{text:.2f}M₫', textposition='outside')
        fig.update_layout(title='💰 Giá vốn vs Tiền về TK', barmode='group', height=370,
                          **CHART_THEME, legend=dict(orientation='h', y=-0.22))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        bar_colors = ['#22c55e' if v >= 0 else '#ef4444' for v in ps['Loi_nhuan']]
        fig2 = go.Figure()
        fig2.add_bar(x=ps['San'], y=ps['Loi_nhuan'], marker_color=bar_colors,
                     text=(ps['Loi_nhuan']/1e6).round(2), texttemplate='%{text:.2f}M₫', textposition='outside')
        fig2.add_hline(y=0, line_dash='dash', line_color='#94a3b8', line_width=1.5)
        fig2.update_layout(title='📊 Lợi nhuận theo sàn', height=370, **CHART_THEME)
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig3 = px.pie(ps, values='So_luong', names='San', title='🛒 Phân bổ sản lượng',
                      color='San', color_discrete_map=COLORS_SAN, hole=0.48)
        fig3.update_traces(textinfo='label+percent+value', pull=[0.04]*len(ps))
        fig3.update_layout(height=330, **CHART_THEME)
        st.plotly_chart(fig3, use_container_width=True)
    with col4:
        fig4 = px.pie(ps, values='Tong_gia_von', names='San', title='💸 Phân bổ giá vốn',
                      color='San', color_discrete_map=COLORS_SAN, hole=0.48)
        fig4.update_traces(textinfo='label+percent')
        fig4.update_layout(height=330, **CHART_THEME)
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown('<div class="sec-title">📋 Bảng tổng hợp</div>', unsafe_allow_html=True)
    ps_disp = ps.copy()
    for col in ['Tong_gia_von','Tien_ve_TK','Loi_nhuan']:
        ps_disp[col] = ps_disp[col].apply(lambda x: f"{x:,.0f}₫")
    ps_disp['Bien_LN']  = ps_disp['Bien_LN'].apply(lambda x: f"{x:.1f}%")
    ps_disp['So_luong'] = ps_disp['So_luong'].apply(lambda x: f"{x:.0f}")
    ps_disp.columns = ['Sàn','Tổng Giá Vốn','Tiền Về TK','Lợi Nhuận','Sản Lượng','Số Dòng SP','Biên LN%']
    st.dataframe(ps_disp, use_container_width=True, hide_index=True)

    if len(ps) > 0:
        best  = ps.loc[ps['Loi_nhuan'].idxmax(),'San']
        worst = ps.loc[ps['Loi_nhuan'].idxmin(),'San']
        st.markdown(f"""
        <div class="insight insight-good">✅ <b>Sàn tốt nhất:</b> <b>{best}</b> — lợi nhuận cao nhất.</div>
        <div class="insight insight-bad">⚠️ <b>Cần xem lại:</b> <b>{worst}</b> — lợi nhuận <b>{ps['Loi_nhuan'].min():,.0f}₫</b>.</div>
        """, unsafe_allow_html=True)

    # ── Excel export cho tab Sàn ──
    st.markdown("---")
    col_dl1, col_dl2 = st.columns([1, 3])
    with col_dl1:
        monthly_agg = None
        if has_months and len(all_data) > 0:
            monthly_agg = (all_data[~all_data['La_hang_tang']]
                .groupby(['Thang_key','Thang_label'])
                .agg(Tien_ve_TK=('Tien_ve_TK','sum'), Tong_gia_von=('Tong_gia_von','sum'),
                     Loi_nhuan=('Loi_nhuan','sum'), So_luong=('So_luong','sum'))
                .reset_index())
            monthly_agg['Bien_LN'] = monthly_agg.apply(
                lambda r: round(r['Loi_nhuan']/r['Tien_ve_TK']*100,1) if r['Tien_ve_TK']>0 else 0.0, axis=1)
        excel_bytes = export_excel_bytes(
            filtered_df=f1, sl_df=sl_data, hoan_df=hoan_data,
            monthly_df=monthly_agg, san_summary_df=ps,
            report_info=f"Sàn: {', '.join(san_filter)} | Tháng: {months_active}"
        )
        st.download_button(
            "📥 Xuất Excel báo cáo đầy đủ",
            data=excel_bytes,
            file_name="bao_cao_tmdt.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )


# ════════════════════════════════════════
# TAB 2 – SẢN PHẨM
# ════════════════════════════════════════
with tab2:
    t2_keys = tab_month_filter('t2_mf', 'Sản phẩm')
    f2 = apply_filters(all_data, month_keys=t2_keys)
    tang2 = f2[f2['La_hang_tang']].copy()

    st.markdown('<div class="sec-title">📦 Phân Tích Sản Phẩm Chi Tiết</div>', unsafe_allow_html=True)
    fc1, fc2, fc3 = st.columns([2,1,1])
    with fc1: top_n = st.slider("Top N sản phẩm", 5, 30, 15, key='topn')
    with fc2: hide_tang_sp = st.checkbox("Ẩn hàng tặng", value=True, key='hide_tang2')
    with fc3: sort_by = st.selectbox("Xếp theo", ['Sản lượng','Giá vốn'], key='sort_sp')

    if len(sl_data) > 0:
        sl_disp = sl_data.copy()
        if has_months and t2_keys:
            sl_disp = sl_disp[sl_disp['Thang_key'].isin(t2_keys)]
        if hide_tang_sp: sl_disp = sl_disp[~sl_disp['La_hang_tang']]
        if brand_filter and 'Thuong_hieu' in sl_disp.columns:
            sl_disp['Thuong_hieu_fill'] = sl_disp['Thuong_hieu'].ffill()
            sl_disp = sl_disp[sl_disp['Thuong_hieu_fill'].isin(brand_filter)]
        sort_col = 'Tong_SL' if sort_by == 'Sản lượng' else 'Gia_von'
        top_prod = sl_disp.nlargest(top_n, sort_col)

        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(top_prod.sort_values(sort_col), x=sort_col, y='Ten_hang', orientation='h',
                         title=f'🏆 Top {top_n} sản phẩm ({sort_by})',
                         color=sort_col, color_continuous_scale='Blues',
                         labels={sort_col:sort_by,'Ten_hang':''})
            fig.update_layout(height=520, **CHART_THEME, coloraxis_showscale=False,
                              yaxis={'categoryorder':'total ascending'})
            fig.update_traces(texttemplate='%{x:.0f}', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            top10 = top_prod.head(10)
            avail_cols = [c for c in ['Tiktok','Shopee','Lazada'] if c in top10.columns]
            if avail_cols:
                melt = top10[['Ten_hang']+avail_cols].melt(id_vars='Ten_hang', var_name='San', value_name='SL')
                melt = melt[melt['SL'] > 0]
                fig2 = px.bar(melt, x='SL', y='Ten_hang', color='San', orientation='h',
                              title='🛒 Phân bổ sản lượng theo sàn (Top 10)',
                              color_discrete_map={'Tiktok':'#fe2c55','Shopee':'#f97316','Lazada':'#6366f1'},
                              barmode='stack', labels={'SL':'Sản lượng','Ten_hang':''})
                fig2.update_layout(height=520, **CHART_THEME,
                                   yaxis={'categoryorder':'total ascending'},
                                   legend=dict(orientation='h',y=-0.15))
                st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="sec-title">🎁 Chi Phí Hàng Tặng Kèm</div>', unsafe_allow_html=True)
    tang_prod2 = (tang2.groupby('Ten_hang_hoa')
                  .agg(SL=('So_luong','sum'), Chi_phi=('Tong_gia_von','sum'))
                  .reset_index().sort_values('Chi_phi', ascending=False).head(15))
    tang_san2  = tang2.groupby('San').agg(SL=('So_luong','sum'), Chi_phi=('Tong_gia_von','sum')).reset_index()
    c1t, c2t = st.columns(2)
    with c1t:
        if len(tang_prod2) > 0:
            fig3 = px.bar(tang_prod2.sort_values('Chi_phi'), x='Chi_phi', y='Ten_hang_hoa', orientation='h',
                          title='💸 Chi phí hàng tặng theo loại', color='Chi_phi',
                          color_continuous_scale='Oranges',
                          labels={'Chi_phi':'Chi phí (₫)','Ten_hang_hoa':''})
            fig3.update_layout(height=400, **CHART_THEME, coloraxis_showscale=False)
            fig3.update_traces(texttemplate='%{x:,.0f}₫', textposition='outside')
            st.plotly_chart(fig3, use_container_width=True)
    with c2t:
        if len(tang_san2) > 0:
            fig4 = go.Figure()
            fig4.add_bar(name='SL hàng tặng', x=tang_san2['San'], y=tang_san2['SL'],
                         marker_color='#fbbf24', text=tang_san2['SL'].round(0),
                         texttemplate='%{text:.0f}', textposition='outside')
            fig4.add_scatter(name='Chi phí (₫)', x=tang_san2['San'], y=tang_san2['Chi_phi'],
                             mode='lines+markers+text', yaxis='y2',
                             line=dict(color='#ef4444',width=2.5), marker=dict(size=10,color='#ef4444'),
                             text=(tang_san2['Chi_phi']/1e6).round(2),
                             texttemplate='%{text:.2f}M₫', textposition='top center')
            fig4.update_layout(title='🎁 Hàng tặng: SL & Chi phí theo sàn', height=400, **CHART_THEME,
                               yaxis=dict(title='Sản lượng'),
                               yaxis2=dict(title='Chi phí', overlaying='y', side='right'),
                               legend=dict(orientation='h',y=-0.2))
            st.plotly_chart(fig4, use_container_width=True)

    tang2_gv  = tang2['Tong_gia_von'].sum()
    f2_gv     = f2['Tong_gia_von'].sum()
    tang_pct2 = tang2_gv / f2_gv * 100 if f2_gv > 0 else 0
    st.markdown(
        f'<div class="insight insight-warn">🎁 <b>Hàng tặng chiếm {tang_pct2:.1f}%</b> tổng giá vốn '
        f'(<b>{tang2_gv:,.0f}₫</b>). Cần theo dõi chặt!</div>', unsafe_allow_html=True)


# ════════════════════════════════════════
# TAB 3 – LỢI NHUẬN
# ════════════════════════════════════════
with tab3:
    t3_keys = tab_month_filter('t3_mf', 'Lợi nhuận')
    f3 = apply_filters(all_data, month_keys=t3_keys)
    tang3 = f3[f3['La_hang_tang']].copy()

    st.markdown('<div class="sec-title">💸 Phân Tích Lợi Nhuận & Cơ Cấu Chi Phí</div>', unsafe_allow_html=True)
    prod_pnl = f3.groupby('Ten_hang_hoa').agg(
        Tong_gia_von=('Tong_gia_von','sum'), Tien_ve_TK=('Tien_ve_TK','sum'),
        Loi_nhuan=('Loi_nhuan','sum'), So_luong=('So_luong','sum'),
        La_tang=('La_hang_tang','first')
    ).reset_index()
    prod_real = prod_pnl[~prod_pnl['La_tang']].copy()

    col1, col2 = st.columns(2)
    with col1:
        tp = prod_real.nlargest(12,'Loi_nhuan')
        fig = px.bar(tp.sort_values('Loi_nhuan'), x='Loi_nhuan', y='Ten_hang_hoa', orientation='h',
                     title='✅ Top sản phẩm lãi nhiều nhất', color='Loi_nhuan',
                     color_continuous_scale='Greens', labels={'Loi_nhuan':'Lợi nhuận (₫)','Ten_hang_hoa':''})
        fig.update_layout(height=460, **CHART_THEME, coloraxis_showscale=False)
        fig.update_traces(texttemplate='%{x:,.0f}₫', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        tl = prod_real.nsmallest(12,'Loi_nhuan')
        fig2 = px.bar(tl.sort_values('Loi_nhuan',ascending=False), x='Loi_nhuan', y='Ten_hang_hoa',
                      orientation='h', title='❌ Top sản phẩm lỗ nhiều nhất', color='Loi_nhuan',
                      color_continuous_scale='Reds_r', labels={'Loi_nhuan':'Lợi nhuận (₫)','Ten_hang_hoa':''})
        fig2.add_vline(x=0, line_dash='dash', line_color='gray')
        fig2.update_layout(height=460, **CHART_THEME, coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="sec-title">🪣 Waterfall Dòng Tiền</div>', unsafe_allow_html=True)
    tv3 = f3['Tien_ve_TK'].sum()
    gv3 = -f3[~f3['La_hang_tang']]['Tong_gia_von'].sum()
    tg3 = -tang3['Tong_gia_von'].sum()
    hv3 = -total_hoan
    net3 = tv3 + gv3 + tg3 + hv3
    fig3 = go.Figure(go.Waterfall(
        measure=["absolute","relative","relative","relative","total"],
        x=["📥 Tiền về TK","📦 (-) Giá vốn","🎁 (-) Hàng tặng","↩️ (-) Hoàn hàng","💰 Lợi nhuận ròng"],
        y=[tv3, gv3, tg3, hv3, 0],
        connector={"line":{"color":"#e2e8f0"}},
        increasing={"marker":{"color":"#22c55e"}},
        decreasing={"marker":{"color":"#ef4444"}},
        totals={"marker":{"color":"#3b82f6"}},
        text=[f"{tv3/1e6:.2f}M",f"{gv3/1e6:.2f}M",f"{tg3/1e6:.2f}M",f"{hv3/1e6:.2f}M",f"{net3/1e6:.2f}M"],
        textposition="outside"
    ))
    fig3.update_layout(title="💧 Từ Tiền Về TK → Lợi Nhuận Ròng (triệu ₫)", height=420, **CHART_THEME)
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown(
        f'<div class="insight insight-info">ℹ️ <b>Lợi nhuận ròng ({net3/1e6:.2f}M₫)</b> = '
        f'Tiền về TK − Giá vốn − Chi phí hàng tặng − Giá vốn hoàn hàng.</div>',
        unsafe_allow_html=True)

    st.markdown('<div class="sec-title">📉 Ma Trận: Sản Lượng vs Lợi Nhuận</div>', unsafe_allow_html=True)
    sc = prod_real[prod_real['So_luong'] > 0].copy()
    sc['Bien_LN'] = sc.apply(
        lambda r: round(r['Loi_nhuan']/r['Tong_gia_von']*100,1) if r['Tong_gia_von']!=0 else 0.0, axis=1)
    fig4 = px.scatter(sc, x='So_luong', y='Loi_nhuan',
                      size=sc['Tong_gia_von'].clip(lower=1),
                      color='Bien_LN', color_continuous_scale='RdYlGn',
                      hover_name='Ten_hang_hoa', title='🔵 Sản lượng vs Lợi nhuận',
                      labels={'So_luong':'Sản lượng','Loi_nhuan':'Lợi nhuận (₫)','Bien_LN':'Biên LN%'})
    fig4.add_hline(y=0, line_dash='dash', line_color='#94a3b8')
    if len(sc) > 0:
        fig4.add_vline(x=sc['So_luong'].median(), line_dash='dot', line_color='#3b82f6', opacity=0.5)
    fig4.update_layout(height=450, **CHART_THEME)
    st.plotly_chart(fig4, use_container_width=True)

    n_loss = (prod_real['Loi_nhuan'] < 0).sum()
    if n_loss > 0:
        wp = prod_real.nsmallest(1,'Loi_nhuan').iloc[0]
        st.markdown(
            f'<div class="insight insight-bad">⚠️ <b>{n_loss} sản phẩm</b> đang thua lỗ. '
            f'SP lỗ nặng nhất: <b>{wp["Ten_hang_hoa"]}</b> với mức lỗ <b>{wp["Loi_nhuan"]:,.0f}₫</b>.</div>',
            unsafe_allow_html=True)
    else:
        st.markdown('<div class="insight insight-good">✅ Tất cả sản phẩm hiện tại đều có lợi nhuận dương!</div>',
                    unsafe_allow_html=True)


# ════════════════════════════════════════
# TAB THÁNG
# ════════════════════════════════════════
with tab_month:
    st.markdown('<div class="sec-title">📅 Phân Tích Xu Hướng Theo Tháng</div>', unsafe_allow_html=True)

    if not has_months:
        st.markdown("""
        <div class="insight insight-info">
        ℹ️ <b>Để dùng tab này</b>, đặt tên file chứa thông tin tháng. Ví dụ:<br>
        <code>thang3.xlsx</code> &nbsp;·&nbsp; <code>T04-2024.xlsx</code> &nbsp;·&nbsp;
        <code>thang 5 2024.xlsx</code> &nbsp;·&nbsp; <code>March2024.xlsx</code>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown('<div class="tab-filter-bar">', unsafe_allow_html=True)
        col_mf1, col_mf2, col_mf3 = st.columns([2,1,1])
        with col_mf1:
            tm_sel  = st.multiselect("📅 Tháng để phân tích", options=ordered_month_labels,
                                     default=ordered_month_labels, key='tm_months')
            tm_keys = [label_to_key[l] for l in tm_sel if l in label_to_key]
        with col_mf2:
            tm_san  = st.multiselect("🏪 Sàn", options=available_sans, default=available_sans, key='tm_san')
        with col_mf3:
            tm_tang = st.checkbox("🎁 Gồm hàng tặng", value=False, key='tm_tang')
        st.markdown('</div>', unsafe_allow_html=True)

        if not tm_keys:
            st.warning("Vui lòng chọn ít nhất 1 tháng.")
            st.stop()

        df_tm = all_data.copy()
        if tm_keys: df_tm = df_tm[df_tm['Thang_key'].isin(tm_keys)]
        if tm_san:  df_tm = df_tm[df_tm['San'].isin(tm_san)]
        if not tm_tang: df_tm = df_tm[~df_tm['La_hang_tang']]

        monthly = (
            df_tm.groupby(['Thang_key','Thang_label'])
            .agg(Tien_ve_TK=('Tien_ve_TK','sum'), Tong_gia_von=('Tong_gia_von','sum'),
                 Loi_nhuan=('Loi_nhuan','sum'), So_luong=('So_luong','sum'))
            .reset_index().sort_values('Thang_key')
        )
        monthly['Bien_LN'] = monthly.apply(
            lambda r: round(r['Loi_nhuan']/r['Tien_ve_TK']*100,1) if r['Tien_ve_TK']>0 else 0.0, axis=1)
        monthly_san = (
            df_tm.groupby(['Thang_key','Thang_label','San'])
            .agg(Tien_ve_TK=('Tien_ve_TK','sum'), Loi_nhuan=('Loi_nhuan','sum'), So_luong=('So_luong','sum'))
            .reset_index().sort_values('Thang_key')
        )

        if len(monthly) >= 2:
            cur, prv = monthly.iloc[-1], monthly.iloc[-2]
            def delta(cv, pv):
                if pv == 0: return "", ""
                pct = (cv - pv) / abs(pv) * 100
                return (f"{'▲' if pct>=0 else '▼'} {abs(pct):.1f}% vs {prv['Thang_label']}",
                        "#22c55e" if pct >= 0 else "#ef4444")
            d1,c1_ = delta(cur['Tien_ve_TK'],prv['Tien_ve_TK'])
            d2,c2_ = delta(cur['Loi_nhuan'],prv['Loi_nhuan'])
            d3,c3_ = delta(cur['So_luong'],prv['So_luong'])
            st.markdown(f"#### 📌 Tháng gần nhất: **{cur['Thang_label']}**")
            km1, km2, km3 = st.columns(3)
            with km1: st.markdown(kpi_card("💰","Tiền về TK",f"{cur['Tien_ve_TK']/1e6:.2f}M₫","","#3b82f6",d1,c1_), unsafe_allow_html=True)
            with km2: st.markdown(kpi_card("📊","Lợi nhuận",f"{cur['Loi_nhuan']/1e6:.2f}M₫","",
                                           "#22c55e" if cur['Loi_nhuan']>=0 else "#ef4444",d2,c2_), unsafe_allow_html=True)
            with km3: st.markdown(kpi_card("🛒","Sản lượng",f"{int(cur['So_luong']):,}","","#f59e0b",d3,c3_), unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

        st.markdown('<div class="sec-title">📈 Xu Hướng Tiền Về TK & Lợi Nhuận</div>', unsafe_allow_html=True)
        c_l, c_r = st.columns(2)
        with c_l:
            fig_line = go.Figure()
            fig_line.add_scatter(x=monthly['Thang_label'], y=monthly['Tien_ve_TK'],
                                 mode='lines+markers+text', name='Tiền về TK',
                                 line=dict(color='#3b82f6',width=2.5), marker=dict(size=9),
                                 text=(monthly['Tien_ve_TK']/1e6).round(2),
                                 texttemplate='%{text:.2f}M', textposition='top center')
            fig_line.add_scatter(x=monthly['Thang_label'], y=monthly['Loi_nhuan'],
                                 mode='lines+markers+text', name='Lợi nhuận',
                                 line=dict(color='#22c55e',width=2.5,dash='dot'), marker=dict(size=9),
                                 text=(monthly['Loi_nhuan']/1e6).round(2),
                                 texttemplate='%{text:.2f}M', textposition='bottom center')
            fig_line.add_hline(y=0, line_dash='dash', line_color='#94a3b8', line_width=1)
            fig_line.update_layout(title='📈 Xu hướng Tiền về TK & Lợi nhuận',
                                   height=400, **CHART_THEME, legend=dict(orientation='h',y=-0.22))
            st.plotly_chart(fig_line, use_container_width=True)
        with c_r:
            fig_bar_m = px.bar(monthly, x='Thang_label', y='So_luong',
                               title='🛒 Sản lượng theo tháng (màu = biên LN%)',
                               color='Bien_LN', color_continuous_scale='RdYlGn', text='So_luong',
                               labels={'Thang_label':'Tháng','So_luong':'Sản lượng','Bien_LN':'Biên LN%'})
            fig_bar_m.update_traces(texttemplate='%{text:.0f}', textposition='outside')
            fig_bar_m.update_layout(height=400, **CHART_THEME)
            st.plotly_chart(fig_bar_m, use_container_width=True)

        st.markdown('<div class="sec-title">🏪 So Sánh Tháng × Sàn</div>', unsafe_allow_html=True)
        metric_opt = st.selectbox("Chỉ tiêu", ['Tiền về TK','Lợi nhuận','Sản lượng'], key='tm_metric')
        mc = {'Tiền về TK':'Tien_ve_TK','Lợi nhuận':'Loi_nhuan','Sản lượng':'So_luong'}[metric_opt]
        is_sl = mc == 'So_luong'
        text_v = monthly_san[mc].apply(lambda v: f"{v:.0f}" if is_sl else f"{v/1e6:.2f}M₫")
        fig_grp = px.bar(monthly_san, x='Thang_label', y=mc, color='San', barmode='group',
                         color_discrete_map=COLORS_SAN,
                         title=f'📊 {metric_opt} theo Tháng & Sàn', text=text_v,
                         labels={'Thang_label':'Tháng',mc:metric_opt,'San':'Sàn'})
        fig_grp.update_traces(textposition='outside')
        fig_grp.update_layout(height=420, **CHART_THEME, legend=dict(orientation='h',y=-0.2))
        st.plotly_chart(fig_grp, use_container_width=True)

        st.markdown('<div class="sec-title">📉 Biên Lợi Nhuận (%) Theo Tháng</div>', unsafe_allow_html=True)
        fig_bien = go.Figure()
        fig_bien.add_bar(x=monthly['Thang_label'], y=monthly['Bien_LN'],
                         marker_color=['#22c55e' if v >= 0 else '#ef4444' for v in monthly['Bien_LN']],
                         text=monthly['Bien_LN'].apply(lambda v: f"{v:.1f}%"), textposition='outside')
        fig_bien.add_hline(y=0, line_dash='dash', line_color='#94a3b8')
        if len(monthly) > 1:
            avg = monthly['Bien_LN'].mean()
            fig_bien.add_hline(y=avg, line_dash='dot', line_color='#3b82f6',
                               annotation_text=f"TB: {avg:.1f}%", annotation_position="top right")
        fig_bien.update_layout(title='📉 Biên lợi nhuận % từng tháng',
                               height=360, **CHART_THEME, yaxis_title='Biên LN (%)')
        st.plotly_chart(fig_bien, use_container_width=True)

        st.markdown('<div class="sec-title">📋 Bảng Pivot: Tháng × Sàn</div>', unsafe_allow_html=True)
        pv_metric = st.selectbox("Chỉ tiêu pivot",
                                 ['Tien_ve_TK','Loi_nhuan','So_luong'],
                                 format_func=lambda x: {'Tien_ve_TK':'Tiền về TK',
                                                        'Loi_nhuan':'Lợi nhuận',
                                                        'So_luong':'Sản lượng'}[x],
                                 key='pv_metric')
        pv_is_sl = pv_metric == 'So_luong'
        pivot = monthly_san.pivot_table(index='Thang_label', columns='San', values=pv_metric,
                                        aggfunc='sum', fill_value=0)
        order_map = {r['Thang_label']:r['Thang_key']
                     for _,r in monthly_san.drop_duplicates('Thang_label').iterrows()}
        pivot = pivot.loc[sorted(pivot.index, key=lambda x: order_map.get(x,x))]
        pivot['📦 Tổng'] = pivot.sum(axis=1)
        pivot_disp = pivot.copy()
        for c in pivot_disp.columns:
            pivot_disp[c] = pivot_disp[c].apply(
                lambda x: f"{x:.0f}" if pv_is_sl else f"{x/1e6:.3f}M₫")
        pivot_disp.index.name = 'Tháng'
        st.dataframe(pivot_disp, use_container_width=True)

        if len(monthly) >= 2:
            bm = monthly.loc[monthly['Loi_nhuan'].idxmax()]
            wm = monthly.loc[monthly['Loi_nhuan'].idxmin()]
            g  = ((monthly['Tien_ve_TK'].iloc[-1]-monthly['Tien_ve_TK'].iloc[-2]) /
                  abs(monthly['Tien_ve_TK'].iloc[-2])*100
                  if monthly['Tien_ve_TK'].iloc[-2] != 0 else 0)
            st.markdown(f"""
            <div class="insight insight-good">🏆 <b>Tháng tốt nhất:</b> {bm['Thang_label']} — Lợi nhuận
            <b>{bm['Loi_nhuan']/1e6:.2f}M₫</b>, biên LN <b>{bm['Bien_LN']:.1f}%</b>.</div>
            <div class="insight {'insight-good' if g>=0 else 'insight-bad'}">
            {'📈' if g>=0 else '📉'} <b>{monthly.iloc[-1]['Thang_label']}:</b>
            Tiền về TK {"tăng" if g>=0 else "giảm"} <b>{abs(g):.1f}%</b>
            so với {monthly.iloc[-2]['Thang_label']}.</div>
            <div class="insight insight-warn">⚠️ <b>Tháng thấp nhất:</b> {wm['Thang_label']} —
            Lợi nhuận <b>{wm['Loi_nhuan']/1e6:.2f}M₫</b>.</div>
            """, unsafe_allow_html=True)
        elif len(monthly) == 1:
            st.markdown('<div class="insight insight-info">ℹ️ Đang xem 1 tháng. '
                        'Tải thêm file từ các tháng khác để so sánh xu hướng.</div>',
                        unsafe_allow_html=True)

        # ── Excel export tab tháng ──
        st.markdown("---")
        excel_tm = export_excel_bytes(
            filtered_df=df_tm, sl_df=sl_data, hoan_df=hoan_data,
            monthly_df=monthly, san_summary_df=pd.DataFrame(),
            report_info=f"Phân tích theo tháng | {', '.join(tm_sel)}"
        )
        st.download_button(
            "📥 Xuất Excel báo cáo theo tháng",
            data=excel_tm,
            file_name="bao_cao_theo_thang.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


# ════════════════════════════════════════
# TAB 4 – HOÀN HÀNG & RỦI RO
# ════════════════════════════════════════
with tab4:
    t4_keys = tab_month_filter('t4_mf', 'Hoàn hàng & Rủi ro')
    f4 = apply_filters(all_data, month_keys=t4_keys)

    st.markdown('<div class="sec-title">↩️ Phân Tích Đơn Hoàn Hàng</div>', unsafe_allow_html=True)
    if len(hoan_data) > 0:
        hoan_pct = total_hoan / f4['Tong_gia_von'].sum() * 100 if f4['Tong_gia_von'].sum() > 0 else 0
        c1, c2, c3 = st.columns(3)
        with c1: st.markdown(kpi_card("↩️","Dòng SP Hoàn",f"{len(hoan_data)}","Tổng dòng ghi nhận","#ef4444"), unsafe_allow_html=True)
        with c2: st.markdown(kpi_card("💸","Giá Vốn Bị Hoàn",f"{total_hoan/1e6:.2f}M₫","Tiền vốn đang bị treo","#f59e0b"), unsafe_allow_html=True)
        with c3: st.markdown(kpi_card("📊","Tỷ Lệ / Giá Vốn",f"{hoan_pct:.1f}%","So với tổng giá vốn","#8b5cf6"), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        hoan_prod = (hoan_data.groupby('Ten_hang')['Tong_gia_von'].sum()
                     .reset_index().sort_values('Tong_gia_von', ascending=False))
        if len(hoan_prod) > 0:
            fig = px.bar(hoan_prod, x='Ten_hang', y='Tong_gia_von',
                         title='↩️ Giá vốn bị hoàn theo sản phẩm',
                         color='Tong_gia_von', color_continuous_scale='Reds',
                         labels={'Ten_hang':'Sản phẩm','Tong_gia_von':'Giá vốn (₫)'})
            fig.update_layout(height=360, **CHART_THEME, coloraxis_showscale=False, xaxis_tickangle=-25)
            fig.update_traces(texttemplate='%{y:,.0f}₫', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)

        hd = hoan_data[['Ten_hang','So_luong','Gia_von','Tong_gia_von']].copy()
        hd['Gia_von']      = hd['Gia_von'].apply(lambda x: f"{x:,.0f}₫")
        hd['Tong_gia_von'] = hd['Tong_gia_von'].apply(lambda x: f"{x:,.0f}₫")
        hd.columns = ['Tên hàng','SL hoàn','Giá vốn/SP','Tổng giá vốn']
        st.dataframe(hd, use_container_width=True, hide_index=True)
        st.markdown(
            f'<div class="insight insight-bad">↩️ Tỷ lệ hoàn <b>{hoan_pct:.1f}%</b> = '
            f'<b>{total_hoan:,.0f}₫</b> giá vốn đang bị treo.</div>',
            unsafe_allow_html=True)
    else:
        st.info("✅ Không tìm thấy dữ liệu hoàn hàng.")

    st.markdown('<div class="sec-title">⚠️ Ma Trận Rủi Ro Sản Phẩm</div>', unsafe_allow_html=True)
    prod_risk = f4[~f4['La_hang_tang']].groupby('Ten_hang_hoa').agg(
        GV=('Tong_gia_von','sum'), LN=('Loi_nhuan','sum'), SL=('So_luong','sum')
    ).reset_index()

    def classify_risk(row):
        if row['LN'] >= 0: return 'An toàn ✅'
        if row['GV'] > 500000: return 'Rủi ro cao 🔴'
        if row['GV'] > 100000: return 'Rủi ro TB 🟡'
        return 'Rủi ro thấp 🟢'

    prod_risk['Rui_ro'] = prod_risk.apply(classify_risk, axis=1)
    rc = prod_risk['Rui_ro'].value_counts().reset_index()
    rc.columns = ['Mức độ','Số SP']
    color_map_r = {'An toàn ✅':'#22c55e','Rủi ro thấp 🟢':'#84cc16',
                   'Rủi ro TB 🟡':'#f59e0b','Rủi ro cao 🔴':'#ef4444'}
    col_r1, col_r2 = st.columns([1,1.3])
    with col_r1:
        fig_r = px.pie(rc, values='Số SP', names='Mức độ', title='📊 Phân bổ rủi ro sản phẩm',
                       color='Mức độ', color_discrete_map=color_map_r, hole=0.42)
        fig_r.update_layout(height=360, **CHART_THEME)
        fig_r.update_traces(textinfo='label+percent+value')
        st.plotly_chart(fig_r, use_container_width=True)
    with col_r2:
        high = prod_risk[prod_risk['Rui_ro']=='Rủi ro cao 🔴'].sort_values('LN').head(10)
        st.markdown("**🔴 Sản phẩm rủi ro cao — cần xử lý ngay:**")
        if len(high) > 0:
            for _, row in high.iterrows():
                nm = str(row['Ten_hang_hoa'])
                nm = nm[:50] + '…' if len(nm) > 50 else nm
                st.markdown(f"""<div class="risk-item" style="border-left-color:#ef4444">
                    ❌ <b>{nm}</b><br>
                    <span style="color:#94a3b8;font-size:0.79rem">
                    Lỗ: <b style="color:#ef4444">{row['LN']:,.0f}₫</b>
                    &nbsp;|&nbsp; GV: {row['GV']:,.0f}₫
                    &nbsp;|&nbsp; SL: {row['SL']:.0f}</span>
                </div>""", unsafe_allow_html=True)
        else:
            st.success("✅ Không có sản phẩm ở mức rủi ro cao!")


# ════════════════════════════════════════
# TAB 5 – DỮ LIỆU & XUẤT
# ════════════════════════════════════════
with tab5:
    st.markdown('<div class="sec-title">🔍 Tra Cứu & Xuất Dữ Liệu</div>', unsafe_allow_html=True)

    fc1, fc2, fc3, fc4, fc5 = st.columns([2,1,1,1,1])
    with fc1: search = st.text_input("🔎 Tìm sản phẩm / khách hàng", placeholder="Nhập từ khoá...")
    with fc2: san_det = st.multiselect("Sàn", available_sans, default=available_sans, key='det_san')
    with fc3:
        if has_months:
            month_det = st.multiselect("📅 Tháng", options=ordered_month_labels,
                                       default=sb_month_sel, key='det_month')
            month_det_keys = [label_to_key[l] for l in month_det if l in label_to_key]
        else:
            month_det_keys = []
    with fc4: type_det = st.selectbox("Loại hàng", ['Tất cả','Hàng bán','Hàng tặng'])
    with fc5: pnl_det  = st.selectbox("Lợi nhuận",  ['Tất cả','Có lãi ✅','Thua lỗ ❌'])

    det = all_data.copy()
    if san_det: det = det[det['San'].isin(san_det)]
    if has_months and month_det_keys: det = det[det['Thang_key'].isin(month_det_keys)]
    if type_det == 'Hàng bán':    det = det[~det['La_hang_tang']]
    elif type_det == 'Hàng tặng': det = det[det['La_hang_tang']]
    if pnl_det == 'Có lãi ✅':    det = det[det['Loi_nhuan'] >= 0]
    elif pnl_det == 'Thua lỗ ❌': det = det[det['Loi_nhuan'] < 0]
    if search:
        mask = (det['Ten_hang_hoa'].astype(str).str.contains(search,case=False,na=False) |
                det.get('Khach_hang', pd.Series(dtype=str)).astype(str).str.contains(search,case=False,na=False))
        det = det[mask]

    total_det_ln = det['Loi_nhuan'].sum()
    st.markdown(f"""
    <div style="background:white;border-radius:12px;padding:12px 18px;margin-bottom:14px;
                box-shadow:0 1px 8px rgba(0,0,0,0.06);font-size:0.84rem;color:#475569;
                border:1px solid #e8ecf5">
    📊 <b>{len(det):,}</b> dòng &nbsp;|&nbsp; 🛒 SL: <b>{det['So_luong'].sum():.0f}</b>
    &nbsp;|&nbsp; 💸 GV: <b>{det['Tong_gia_von'].sum():,.0f}₫</b>
    &nbsp;|&nbsp; 📥 TK: <b>{det['Tien_ve_TK'].sum():,.0f}₫</b>
    &nbsp;|&nbsp; 📊 LN:
    <b style="color:{'#22c55e' if total_det_ln>=0 else '#ef4444'}">{total_det_ln:,.0f}₫</b>
    </div>""", unsafe_allow_html=True)

    show_cols = [c for c in ['San','Thang_label','Ten_hang_hoa','Khach_hang','So_luong',
                              'Gia_von','Tong_gia_von','Tien_ve_TK','Loi_nhuan','La_hang_tang']
                 if c in det.columns]
    det_disp = det[show_cols].copy()
    det_disp['La_hang_tang'] = det_disp['La_hang_tang'].map({True:'🎁 Tặng',False:'✅ Bán'})
    rename_det = {'San':'Sàn','Thang_label':'Tháng','Ten_hang_hoa':'Tên hàng','Khach_hang':'Khách',
                  'So_luong':'SL','Gia_von':'GV/SP','Tong_gia_von':'T.GV',
                  'Tien_ve_TK':'Tiền về TK','Loi_nhuan':'Lợi nhuận','La_hang_tang':'Loại'}
    det_disp.rename(columns=rename_det, inplace=True)
    st.dataframe(det_disp, use_container_width=True, height=420, hide_index=True)

    # ── Download buttons ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📥 Xuất dữ liệu")
    dl1, dl2, dl3 = st.columns(3)

    # CSV export
    with dl1:
        csv_buf = det.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
        st.download_button(
            "📄 Tải CSV (bộ lọc hiện tại)",
            data=csv_buf,
            file_name="tmdt_export_filtered.csv",
            mime='text/csv',
            use_container_width=True
        )

    # Excel full report
    with dl2:
        ps_for_excel = all_data.groupby('San').agg(
            Tong_gia_von=('Tong_gia_von','sum'), Tien_ve_TK=('Tien_ve_TK','sum'),
            Loi_nhuan=('Loi_nhuan','sum'), So_luong=('So_luong','sum'),
            So_dong=('Ten_hang_hoa','count')
        ).reset_index()
        ps_for_excel['Bien_LN'] = ps_for_excel.apply(
            lambda r: round(r['Loi_nhuan']/r['Tien_ve_TK']*100,1) if r['Tien_ve_TK']>0 else 0.0, axis=1)

        monthly_for_excel = None
        if has_months and len(all_data) > 0:
            monthly_for_excel = (
                all_data[~all_data['La_hang_tang']]
                .groupby(['Thang_key','Thang_label'])
                .agg(Tien_ve_TK=('Tien_ve_TK','sum'), Tong_gia_von=('Tong_gia_von','sum'),
                     Loi_nhuan=('Loi_nhuan','sum'), So_luong=('So_luong','sum'))
                .reset_index()
            )
            monthly_for_excel['Bien_LN'] = monthly_for_excel.apply(
                lambda r: round(r['Loi_nhuan']/r['Tien_ve_TK']*100,1) if r['Tien_ve_TK']>0 else 0.0, axis=1)

        excel_full = export_excel_bytes(
            filtered_df=det, sl_df=sl_data, hoan_df=hoan_data,
            monthly_df=monthly_for_excel, san_summary_df=ps_for_excel,
            report_info=f"Sàn: {', '.join(san_det)} | Loại: {type_det} | Lọc: {search or 'tất cả'}"
        )
        st.download_button(
            "📊 Xuất Excel đẹp (bộ lọc hiện tại)",
            data=excel_full,
            file_name="bao_cao_tmdt_chitiet.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

    # Excel all data
    with dl3:
        excel_all = export_excel_bytes(
            filtered_df=all_data[~all_data['La_hang_tang']],
            sl_df=sl_data, hoan_df=hoan_data,
            monthly_df=monthly_for_excel, san_summary_df=ps_for_excel,
            report_info=f"Toàn bộ dữ liệu | {len(uploaded_files)} file"
        )
        st.download_button(
            "📦 Xuất Excel toàn bộ dữ liệu",
            data=excel_all,
            file_name="bao_cao_tmdt_toanbo.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

    st.markdown("""
    <div class="insight insight-info" style="margin-top:8px">
    ℹ️ File Excel xuất ra gồm nhiều sheet: <b>Tổng quan · Chi tiết · Theo sàn · Theo tháng · Hoàn hàng · Top sản phẩm</b>
    &nbsp;— Mở bằng Microsoft Excel hoặc Google Sheets.
    </div>""", unsafe_allow_html=True)

    if len(sl_data) > 0 and 'Thuong_hieu' in sl_data.columns:
        st.markdown('<div class="sec-title">🏷️ Tổng Hợp Theo Thương Hiệu</div>', unsafe_allow_html=True)
        sl_b = sl_data.copy()
        sl_b['TH'] = sl_b['Thuong_hieu'].ffill()
        pn = {'tiktok shop','shopee','lazada','tiktokshop'}
        sl_b = sl_b[~sl_b['TH'].astype(str).str.strip().str.lower().isin(pn)]
        bs = (sl_b[~sl_b['La_hang_tang']].groupby('TH')
              .agg(So_SP=('Ten_hang','count'), Tong_SL=('Tong_SL','sum'))
              .reset_index().sort_values('Tong_SL', ascending=False).head(15))
        bs.columns = ['Thương hiệu','Số mã SP','Tổng SL']
        fig_b = px.bar(bs, x='Thương hiệu', y='Tổng SL',
                       title='🏷️ Sản lượng bán theo thương hiệu',
                       color='Tổng SL', color_continuous_scale='Purples', text='Tổng SL')
        fig_b.update_layout(height=380, **CHART_THEME, coloraxis_showscale=False, xaxis_tickangle=-30)
        fig_b.update_traces(texttemplate='%{text:.0f}', textposition='outside')
        st.plotly_chart(fig_b, use_container_width=True)


# ─── FOOTER ──────────────────────────────────────────────────────────
st.markdown("---")
months_footer = " · ".join(ordered_month_labels) if ordered_month_labels else "—"
st.markdown(f"""
<div style="text-align:center;color:#94a3b8;font-size:0.76rem;padding:12px 0;
            font-family:'Plus Jakarta Sans',sans-serif">
    📊 Phân tích TMĐT Online &nbsp;|&nbsp; <code>{file_label}</code>
    &nbsp;|&nbsp; Sàn: {" · ".join(available_sans)}
    &nbsp;|&nbsp; 📅 {months_footer}
    &nbsp;|&nbsp; {len(all_data):,} giao dịch
    &nbsp;|&nbsp; <span style="opacity:0.5">Phông: Plus Jakarta Sans · Xuất Excel: openpyxl</span>
</div>""", unsafe_allow_html=True)
