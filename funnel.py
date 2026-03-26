import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import warnings
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
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Be Vietnam Pro', sans-serif !important; }
.stApp { background: #f0f2f8; }
[data-testid="stSidebar"] { background: #1a1a2e !important; }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] label { color: #94a3b8 !important; font-size: 0.78rem !important; text-transform: uppercase; letter-spacing: 0.05em; }
[data-testid="stSidebar"] h3 { color: #ffffff !important; font-size: 1rem !important; }

.upload-zone {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
    border-radius: 20px; padding: 64px 36px; text-align: center;
    color: white; margin-bottom: 24px; border: 2px dashed rgba(255,255,255,0.18);
}
.upload-zone h1 { font-size: 2.2rem; font-weight: 800; margin: 0 0 8px 0; }
.upload-zone p { opacity: 0.65; font-size: 0.95rem; margin: 0; }

.dash-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px; padding: 22px 30px; margin-bottom: 22px; color: white;
}
.dash-header h2 { font-size: 1.5rem; font-weight: 700; margin: 0 0 4px 0; }
.dash-header p { opacity: 0.65; font-size: 0.85rem; margin: 0; }

.kpi-wrap {
    background: white; border-radius: 14px; padding: 18px 20px;
    border-top: 4px solid; box-shadow: 0 2px 16px rgba(0,0,0,0.07);
    height: 100%; transition: all 0.25s ease;
}
.kpi-wrap:hover { box-shadow: 0 6px 24px rgba(0,0,0,0.13); transform: translateY(-2px); }
.kpi-icon { font-size: 1.5rem; margin-bottom: 8px; }
.kpi-label { font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.07em; color: #94a3b8; margin-bottom: 4px; }
.kpi-value { font-size: 1.6rem; font-weight: 800; line-height: 1.1; }
.kpi-sub { font-size: 0.78rem; color: #94a3b8; margin-top: 5px; }
.kpi-badge { display: inline-block; padding: 2px 8px; border-radius: 20px; font-size: 0.7rem; font-weight: 700; margin-top: 6px; }

.sec-title {
    font-size: 1.05rem; font-weight: 700; color: #1a1a2e;
    padding: 14px 0 10px 0; border-bottom: 2px solid #e2e8f0; margin-bottom: 14px;
}

.insight { border-radius: 10px; padding: 13px 16px; margin: 8px 0; font-size: 0.87rem; line-height: 1.6; }
.insight-warn  { background: #fffbeb; border-left: 4px solid #f59e0b; }
.insight-bad   { background: #fff1f2; border-left: 4px solid #f43f5e; }
.insight-good  { background: #f0fdf4; border-left: 4px solid #22c55e; }
.insight-info  { background: #eff6ff; border-left: 4px solid #3b82f6; }

.stTabs [data-baseweb="tab-list"] { gap: 6px; background: #e2e8f0; border-radius: 12px; padding: 4px; }
.stTabs [data-baseweb="tab"] {
    background: transparent; border-radius: 9px; padding: 8px 18px;
    font-weight: 600; color: #64748b; border: none !important; font-size: 0.85rem;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1a1a2e, #0f3460) !important;
    color: white !important; box-shadow: 0 2px 8px rgba(15,52,96,0.3) !important;
}

.risk-item {
    background: white; border-radius: 10px; padding: 10px 14px; margin: 5px 0;
    border-left: 4px solid; font-size: 0.83rem; box-shadow: 0 1px 6px rgba(0,0,0,0.06);
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════
def kpi_card(icon, label, value, sub, color, badge_txt="", badge_color=""):
    badge = f'<div class="kpi-badge" style="background:{badge_color}22; color:{badge_color}">{badge_txt}</div>' if badge_txt else ""
    return f"""
    <div class="kpi-wrap" style="border-top-color:{color}">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value" style="color:{color}">{value}</div>
        <div class="kpi-sub">{sub}</div>
        {badge}
    </div>"""


def parse_sales_sheet(df_raw, san_label):
    header_row = None
    for i, row in df_raw.iterrows():
        vals = str(row.values)
        if 'STT' in vals and 'Tên hàng hoá' in vals:
            header_row = i
            break
    if header_row is None:
        return pd.DataFrame()
    data = df_raw.iloc[header_row + 1:].copy()
    data.columns = range(len(data.columns))
    col_map = {0: 'STT', 1: 'Khach_hang', 2: 'Ma_don_hang', 3: 'Ma_san_pham',
               4: 'Ten_hang_hoa', 5: 'So_luong', 6: 'Gia_von', 7: 'Tong_gia_von',
               8: 'Tien_ve_TK', 9: 'Doanh_thu_thuc', 10: 'Ngay', 11: 'Loi_nhuan', 12: 'Trang_thai'}
    cols = {k: v for k, v in col_map.items() if k < len(data.columns)}
    df = data[list(cols.keys())].copy()
    df.columns = list(cols.values())
    # FIX BUG 3: lọc bỏ junk rows (tên hàng quá ngắn như backtick, khoảng trắng)
    df = df[
        df['Ten_hang_hoa'].notna() &
        (df['Ten_hang_hoa'].astype(str).str.strip() != '') &
        (df['Ten_hang_hoa'].astype(str) != 'nan') &
        (df['Ten_hang_hoa'].astype(str).str.strip().str.len() > 2)
    ]
    df['San'] = san_label
    df['La_hang_tang'] = df['Ma_san_pham'].astype(str).str.endswith('-T', na=False)
    for col in ['So_luong', 'Gia_von', 'Tong_gia_von', 'Tien_ve_TK', 'Loi_nhuan']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'Ngay' in df.columns:
        df['Ngay'] = pd.to_datetime(df['Ngay'], errors='coerce')
    df = df[~df['Ten_hang_hoa'].astype(str).str.upper().str.contains('TỔNG|TOTAL', na=False)]
    return df


def parse_sl_sheet(all_sheets):
    key = next((k for k in all_sheets if 'SL' in k.upper() or 'TỔNG' in k.upper()), None)
    if not key:
        return pd.DataFrame()
    df = all_sheets[key]
    sl = df.iloc[4:].copy()
    sl.columns = range(len(sl.columns))
    rename = {0: 'STT', 1: 'Thuong_hieu', 2: 'Ma_sp', 4: 'Ten_hang',
              5: 'Tiktok', 6: 'Shopee', 7: 'Lazada', 8: 'Tong_SL', 9: 'Gia_von'}
    sl = sl.rename(columns={k: v for k, v in rename.items() if k < len(sl.columns)})
    if 'Ten_hang' not in sl.columns:
        return pd.DataFrame()
    sl['Tong_SL'] = pd.to_numeric(sl.get('Tong_SL', 0), errors='coerce').fillna(0)
    sl['Gia_von'] = pd.to_numeric(sl.get('Gia_von', 0), errors='coerce').fillna(0)
    for c in ['Tiktok', 'Shopee', 'Lazada']:
        if c in sl.columns:
            sl[c] = pd.to_numeric(sl[c], errors='coerce').fillna(0)
    # FIX BUG 2: phải filter Ten_hang có giá trị TRƯỚC khi lọc Tong_SL > 0
    # để loại bỏ các footer row tổng cuối sheet (có Tong_SL lớn nhưng Ten_hang = NaN)
    sl = sl[
        sl['Ten_hang'].notna() &
        (sl['Ten_hang'].astype(str).str.strip() != '') &
        (sl['Ten_hang'].astype(str) != 'nan')
    ]
    sl = sl[sl['Tong_SL'] > 0]
    sl = sl[~sl['STT'].astype(str).str.upper().str.contains('TỔNG|TOTAL', na=False)]
    sl['La_hang_tang'] = sl['Ten_hang'].astype(str).str.contains('Hàng tặng|hàng tặng', na=False)
    return sl


def parse_hoan_hang(all_sheets):
    key = next((k for k in all_sheets if 'HOÀN' in k.upper() or 'HOAN' in k.upper()), None)
    if not key:
        return pd.DataFrame()
    df = all_sheets[key]
    header_row = None
    for i, row in df.iterrows():
        vals = str(row.values)
        if 'STT' in vals and 'Tên hàng hoá' in vals:
            header_row = i
            break
    if header_row is None:
        return pd.DataFrame()
    data = df.iloc[header_row + 1:].copy()
    data.columns = range(len(data.columns))
    col_map = {0: 'San', 1: 'STT', 2: 'Khach_hang', 3: 'Ma_don', 4: 'Ma_sp',
               5: 'Ten_hang', 6: 'So_luong', 7: 'Gia_von', 8: 'Tong_gia_von'}
    cols = {k: v for k, v in col_map.items() if k < len(data.columns)}
    df2 = data[list(cols.keys())].copy()
    df2.columns = list(cols.values())
    df2 = df2[df2['Ten_hang'].notna() & (df2['Ten_hang'].astype(str).str.strip() != '')]
    df2['Tong_gia_von'] = pd.to_numeric(df2['Tong_gia_von'], errors='coerce').fillna(0)
    df2['So_luong'] = pd.to_numeric(df2['So_luong'], errors='coerce').fillna(0)
    # Ffill cột Sàn vì sheet gộp nhiều sàn
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
        if 'TIKTOK' in su:
            san_map['TiktokShop'] = s
        elif 'SHOPEE' in su:
            san_map['Shopee'] = s
        elif 'LAZADA' in su:
            san_map['Lazada'] = s

    frames = []
    for label, sheet_name in san_map.items():
        df = parse_sales_sheet(all_sheets[sheet_name], label)
        if len(df) > 0:
            frames.append(df)

    all_data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    sl_data = parse_sl_sheet(all_sheets)
    hoan_data = parse_hoan_hang(all_sheets)

    return all_data, sl_data, hoan_data, list(san_map.keys()), list(all_sheets.keys())


# ══════════════════════════════════════════════════════════════════════
# SIDEBAR – FILE UPLOAD
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 📂 Tải lên dữ liệu")
    uploaded = st.file_uploader(
        "Kéo thả file Excel vào đây",
        type=["xlsx", "xls"],
        help="File Excel báo cáo TMĐT có các sheet TiktokShop, Shopee, Lazada..."
    )
    st.markdown("---")


# ══════════════════════════════════════════════════════════════════════
# LANDING PAGE
# ══════════════════════════════════════════════════════════════════════
if not uploaded:
    st.markdown("""
    <div class="upload-zone">
        <div style="font-size:4rem; margin-bottom:20px">📊</div>
        <h1>Phân Tích Doanh Số TMĐT</h1>
        <p style="font-size:1.1rem; margin-top:10px">TiktokShop &nbsp;·&nbsp; Shopee &nbsp;·&nbsp; Lazada</p>
        <br>
        <p>👈 <strong>Kéo thả file Excel</strong> vào ô bên trái để bắt đầu</p>
        <p style="margin-top:10px; font-size:0.85rem; opacity:0.5">
            Hỗ trợ định dạng .xlsx — Có thể tải lên nhiều kỳ báo cáo khác nhau
        </p>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    for col, icon, title, desc in zip(
        [c1, c2, c3],
        ["🏪", "📦", "💸"],
        ["Hiệu suất theo Sàn", "Phân tích Sản phẩm", "Lợi nhuận & Rủi ro"],
        [
            "So sánh doanh thu, giá vốn, lợi nhuận giữa 3 sàn TMĐT với bộ lọc linh hoạt",
            "Top sản phẩm bán chạy, phân bổ sàn, chi phí hàng tặng kèm theo",
            "Waterfall dòng tiền, ma trận rủi ro, đơn hoàn hàng, SP lãi/lỗ"
        ]
    ):
        with col:
            st.markdown(f"""
            <div style="background:white;border-radius:14px;padding:28px 22px;text-align:center;
                        box-shadow:0 2px 14px rgba(0,0,0,0.08);min-height:180px">
                <div style="font-size:2.4rem">{icon}</div>
                <div style="font-weight:700;font-size:1rem;margin:12px 0 8px;color:#1a1a2e">{title}</div>
                <div style="font-size:0.82rem;color:#64748b;line-height:1.55">{desc}</div>
            </div>""", unsafe_allow_html=True)
    st.stop()


# ══════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════
file_bytes = uploaded.read()
with st.spinner("⏳ Đang đọc và phân tích dữ liệu..."):
    all_data, sl_data, hoan_data, available_sans, all_sheet_names = load_and_parse(file_bytes, uploaded.name)

if len(all_data) == 0:
    st.error("❌ Không tìm thấy dữ liệu giao dịch. Kiểm tra xem file có sheet TiktokShop/Shopee/Lazada không.")
    with st.expander("📋 Sheets tìm thấy trong file"):
        st.write(all_sheet_names)
    st.stop()


# ══════════════════════════════════════════════════════════════════════
# SIDEBAR FILTERS
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🎛️ Bộ lọc")

    san_filter = st.multiselect(
        "🏪 Sàn TMĐT", options=available_sans, default=available_sans
    )
    include_tang = st.checkbox("🎁 Bao gồm hàng tặng", value=False)

    show_profit = st.checkbox("✅ Hiển thị hàng có lãi", value=True)
    show_loss   = st.checkbox("❌ Hiển thị hàng thua lỗ", value=True)

    if 'So_luong' in all_data.columns and not all_data['So_luong'].isna().all():
        max_sl = int(all_data['So_luong'].max())
        sl_range = st.slider("📦 Sản lượng / dòng SP", 0, max(max_sl, 1), (0, max(max_sl, 1)))
    else:
        sl_range = (0, 99999)

    all_brands = []
    if len(sl_data) > 0 and 'Thuong_hieu' in sl_data.columns:
        # FIX: ffill thương hiệu trong SL sheet trước khi lấy unique
        brands_series = sl_data['Thuong_hieu'].ffill()
        brands = brands_series.dropna().unique().tolist()
        # Loại bỏ tên sàn bị lọt vào cuối sheet
        platform_names = {'tiktok shop', 'shopee', 'lazada', 'tiktokshop'}
        all_brands = [b for b in brands if str(b).strip().lower() not in platform_names
                      and str(b).strip() not in ['', 'nan', 'Thương hiệu']]
    brand_filter = st.multiselect("🏷️ Thương hiệu", options=all_brands) if all_brands else []

    st.markdown("---")
    st.success(f"✅ {uploaded.name}")
    st.caption(f"Sàn: {', '.join(available_sans)}")
    st.caption(f"Tổng: {len(all_data):,} dòng SP")


# ─── Apply Filters ─────────────────────────────────────────────────────
filtered = all_data[all_data['San'].isin(san_filter)] if san_filter else all_data.copy()
if not include_tang:
    filtered = filtered[~filtered['La_hang_tang']]
if not (show_profit and show_loss):
    if show_profit:   filtered = filtered[filtered['Loi_nhuan'] >= 0]
    elif show_loss:   filtered = filtered[filtered['Loi_nhuan'] < 0]
if 'So_luong' in filtered.columns:
    filtered = filtered[
        filtered['So_luong'].between(sl_range[0], sl_range[1], inclusive='both') |
        filtered['So_luong'].isna()
    ]
tang_items = all_data[all_data['La_hang_tang'] & all_data['San'].isin(san_filter if san_filter else available_sans)]


# ══════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="dash-header">
  <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px">
    <div>
      <h2>🛍️ Phân Tích Doanh Số Bán Hàng Online</h2>
      <p>📁 {uploaded.name} &nbsp;|&nbsp; Sàn: {" · ".join(san_filter if san_filter else ["—"])}
         &nbsp;|&nbsp; {"Có hàng tặng" if include_tang else "Không tính hàng tặng"}</p>
    </div>
    <div style="opacity:0.6;font-size:0.8rem;text-align:right">
      Hiển thị <b>{len(filtered):,}</b> dòng SP
    </div>
  </div>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# KPI ROW
# ══════════════════════════════════════════════════════════════════════
total_gia_von   = filtered['Tong_gia_von'].sum()
total_tien_ve   = filtered['Tien_ve_TK'].sum()
total_loi_nhuan = filtered['Loi_nhuan'].sum()
total_sl        = filtered['So_luong'].sum()
total_tang_cost = tang_items['Tong_gia_von'].sum()
total_hoan      = hoan_data['Tong_gia_von'].sum() if len(hoan_data) > 0 else 0

# FIX: tránh chia cho 0 hoặc Tien_ve_TK âm làm biên LN vô nghĩa
if total_tien_ve > 0:
    profit_margin = total_loi_nhuan / total_tien_ve * 100
else:
    profit_margin = 0.0

c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1: st.markdown(kpi_card("💰","Tiền Về TK",f"{total_tien_ve/1e6:.2f}M₫","Dự kiến nhận về","#3b82f6"), unsafe_allow_html=True)
with c2: st.markdown(kpi_card("📦","Tổng Giá Vốn",f"{total_gia_von/1e6:.2f}M₫","Giá vốn hàng bán","#8b5cf6"), unsafe_allow_html=True)
with c3:
    color_ln = "#22c55e" if total_loi_nhuan >= 0 else "#ef4444"
    st.markdown(kpi_card("📊","Lợi Nhuận",f"{total_loi_nhuan/1e6:.2f}M₫","Sau giá vốn",color_ln,
                f"Biên LN: {profit_margin:.1f}%","#22c55e" if profit_margin >= 0 else "#ef4444"), unsafe_allow_html=True)
with c4: st.markdown(kpi_card("🛒","Sản Lượng",f"{int(total_sl):,}","Tổng SP bán ra","#f59e0b"), unsafe_allow_html=True)
with c5: st.markdown(kpi_card("🎁","Hàng Tặng",f"{total_tang_cost/1e6:.2f}M₫","Chi phí giá vốn tặng","#ec4899"), unsafe_allow_html=True)
with c6: st.markdown(kpi_card("↩️","Hàng Hoàn",f"{total_hoan/1e6:.2f}M₫",f"{len(hoan_data)} dòng hoàn trả","#ef4444"), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

CHART_THEME = dict(plot_bgcolor='white', paper_bgcolor='white', font=dict(family='Be Vietnam Pro', size=12))
COLORS_SAN  = {'TiktokShop': '#fe2c55', 'Shopee': '#f97316', 'Lazada': '#6366f1'}


# ══════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏪 Hiệu suất Sàn",
    "📦 Sản phẩm",
    "💸 Lợi nhuận",
    "↩️ Hoàn hàng & Rủi ro",
    "🔍 Dữ liệu chi tiết"
])


# ── TAB 1 ──────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="sec-title">🏪 So Sánh Hiệu Suất Các Sàn TMĐT</div>', unsafe_allow_html=True)

    ps = filtered.groupby('San').agg(
        Tong_gia_von=('Tong_gia_von', 'sum'),
        Tien_ve_TK=('Tien_ve_TK', 'sum'),
        Loi_nhuan=('Loi_nhuan', 'sum'),
        So_luong=('So_luong', 'sum'),
        So_dong=('Ten_hang_hoa', 'count')
    ).reset_index()
    ps['Bien_LN'] = ps.apply(
        lambda r: round(r['Loi_nhuan'] / r['Tien_ve_TK'] * 100, 1) if r['Tien_ve_TK'] > 0 else 0.0,
        axis=1
    )

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_bar(name='Giá vốn', x=ps['San'], y=ps['Tong_gia_von'],
                    marker_color='#cbd5e1', text=(ps['Tong_gia_von']/1e6).round(2),
                    texttemplate='%{text:.2f}M₫', textposition='outside')
        fig.add_bar(name='Tiền về TK', x=ps['San'], y=ps['Tien_ve_TK'],
                    marker_color='#3b82f6', text=(ps['Tien_ve_TK']/1e6).round(2),
                    texttemplate='%{text:.2f}M₫', textposition='outside')
        fig.update_layout(title='💰 Giá vốn vs Tiền về TK', barmode='group',
                          height=370, **CHART_THEME, legend=dict(orientation='h', y=-0.2))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        bar_colors = ['#22c55e' if v >= 0 else '#ef4444' for v in ps['Loi_nhuan']]
        fig2 = go.Figure()
        fig2.add_bar(x=ps['San'], y=ps['Loi_nhuan'], marker_color=bar_colors,
                     text=(ps['Loi_nhuan']/1e6).round(2),
                     texttemplate='%{text:.2f}M₫', textposition='outside')
        fig2.add_hline(y=0, line_dash='dash', line_color='#94a3b8', line_width=1.5)
        fig2.update_layout(title='📊 Lợi nhuận theo sàn', height=370, **CHART_THEME)
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig3 = px.pie(ps, values='So_luong', names='San',
                      title='🛒 Phân bổ sản lượng theo sàn',
                      color='San', color_discrete_map=COLORS_SAN, hole=0.48)
        fig3.update_traces(textinfo='label+percent+value', pull=[0.04]*len(ps))
        fig3.update_layout(height=330, **CHART_THEME)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        fig4 = px.pie(ps, values='Tong_gia_von', names='San',
                      title='💸 Phân bổ giá vốn theo sàn',
                      color='San', color_discrete_map=COLORS_SAN, hole=0.48)
        fig4.update_traces(textinfo='label+percent')
        fig4.update_layout(height=330, **CHART_THEME)
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown('<div class="sec-title">📋 Bảng tổng hợp</div>', unsafe_allow_html=True)
    ps_disp = ps.copy()
    for col in ['Tong_gia_von', 'Tien_ve_TK', 'Loi_nhuan']:
        ps_disp[col] = ps_disp[col].apply(lambda x: f"{x:,.0f}₫")
    ps_disp['Bien_LN'] = ps_disp['Bien_LN'].apply(lambda x: f"{x:.1f}%")
    ps_disp['So_luong'] = ps_disp['So_luong'].apply(lambda x: f"{x:.0f}")
    ps_disp.columns = ['Sàn', 'Tổng Giá Vốn', 'Tiền Về TK', 'Lợi Nhuận', 'Sản Lượng', 'Số Dòng SP', 'Biên LN%']
    st.dataframe(ps_disp, use_container_width=True, hide_index=True)

    if len(ps) > 0:
        best  = ps.loc[ps['Loi_nhuan'].idxmax(), 'San']
        worst = ps.loc[ps['Loi_nhuan'].idxmin(), 'San']
        worst_ln = ps['Loi_nhuan'].min()
        st.markdown(f"""
        <div class="insight insight-good">✅ <b>Sàn tốt nhất:</b> <b>{best}</b> — có lợi nhuận cao nhất trong kỳ.</div>
        <div class="insight insight-bad">⚠️ <b>Cần xem lại:</b> <b>{worst}</b> lợi nhuận <b>{worst_ln:,.0f}₫</b>.
        Kiểm tra chiến lược giá, hàng tặng và flash sale.</div>""", unsafe_allow_html=True)


# ── TAB 2 ──────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="sec-title">📦 Phân Tích Sản Phẩm Chi Tiết</div>', unsafe_allow_html=True)

    fc1, fc2, fc3 = st.columns([2, 1, 1])
    with fc1: top_n = st.slider("Top N sản phẩm", 5, 30, 15, key='topn')
    with fc2: hide_tang_sp = st.checkbox("Ẩn hàng tặng", value=True, key='hide_tang2')
    with fc3: sort_by = st.selectbox("Xếp theo", ['Sản lượng', 'Giá vốn'], key='sort_sp')

    if len(sl_data) > 0:
        sl_disp = sl_data.copy()
        if hide_tang_sp:
            sl_disp = sl_disp[~sl_disp['La_hang_tang']]
        if brand_filter and 'Thuong_hieu' in sl_disp.columns:
            sl_disp['Thuong_hieu_fill'] = sl_disp['Thuong_hieu'].ffill()
            sl_disp = sl_disp[sl_disp['Thuong_hieu_fill'].isin(brand_filter)]
        sort_col = 'Tong_SL' if sort_by == 'Sản lượng' else 'Gia_von'
        top_prod = sl_disp.nlargest(top_n, sort_col)

        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(top_prod.sort_values(sort_col),
                         x=sort_col, y='Ten_hang', orientation='h',
                         title=f'🏆 Top {top_n} sản phẩm ({sort_by})',
                         color=sort_col, color_continuous_scale='Blues',
                         labels={sort_col: sort_by, 'Ten_hang': ''})
            fig.update_layout(height=520, **CHART_THEME, coloraxis_showscale=False,
                              yaxis={'categoryorder': 'total ascending'})
            fig.update_traces(texttemplate='%{x:.0f}', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            top10 = top_prod.head(10)
            avail_cols = [c for c in ['Tiktok', 'Shopee', 'Lazada'] if c in top10.columns]
            if avail_cols:
                melt = top10[['Ten_hang'] + avail_cols].melt(
                    id_vars='Ten_hang', var_name='San', value_name='SL')
                melt = melt[melt['SL'] > 0]
                fig2 = px.bar(melt, x='SL', y='Ten_hang', color='San', orientation='h',
                              title='🛒 Phân bổ sản lượng theo sàn (Top 10)',
                              color_discrete_map={'Tiktok': '#fe2c55', 'Shopee': '#f97316', 'Lazada': '#6366f1'},
                              barmode='stack', labels={'SL': 'Sản lượng', 'Ten_hang': ''})
                fig2.update_layout(height=520, **CHART_THEME,
                                   yaxis={'categoryorder': 'total ascending'},
                                   legend=dict(orientation='h', y=-0.15))
                st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="sec-title">🎁 Chi Phí Hàng Tặng Kèm</div>', unsafe_allow_html=True)
    tang_prod = tang_items.groupby('Ten_hang_hoa').agg(
        SL=('So_luong', 'sum'), Chi_phi=('Tong_gia_von', 'sum')
    ).reset_index().sort_values('Chi_phi', ascending=False).head(15)
    tang_san = tang_items.groupby('San').agg(
        SL=('So_luong', 'sum'), Chi_phi=('Tong_gia_von', 'sum')
    ).reset_index()

    c1t, c2t = st.columns(2)
    with c1t:
        if len(tang_prod) > 0:
            fig3 = px.bar(tang_prod.sort_values('Chi_phi'),
                          x='Chi_phi', y='Ten_hang_hoa', orientation='h',
                          title='💸 Chi phí hàng tặng theo loại',
                          color='Chi_phi', color_continuous_scale='Oranges',
                          labels={'Chi_phi': 'Chi phí (₫)', 'Ten_hang_hoa': ''})
            fig3.update_layout(height=400, **CHART_THEME, coloraxis_showscale=False)
            fig3.update_traces(texttemplate='%{x:,.0f}₫', textposition='outside')
            st.plotly_chart(fig3, use_container_width=True)

    with c2t:
        if len(tang_san) > 0:
            fig4 = go.Figure()
            fig4.add_bar(name='SL hàng tặng', x=tang_san['San'], y=tang_san['SL'],
                         marker_color='#fbbf24', text=tang_san['SL'].round(0),
                         texttemplate='%{text:.0f}', textposition='outside')
            fig4.add_scatter(name='Chi phí (₫)', x=tang_san['San'], y=tang_san['Chi_phi'],
                             mode='lines+markers+text', yaxis='y2',
                             line=dict(color='#ef4444', width=2.5),
                             marker=dict(size=10, color='#ef4444'),
                             text=(tang_san['Chi_phi']/1e6).round(2),
                             texttemplate='%{text:.2f}M₫', textposition='top center')
            fig4.update_layout(
                title='🎁 Hàng tặng: Sản lượng & Chi phí theo sàn',
                height=400, **CHART_THEME,
                yaxis=dict(title='Sản lượng'),
                yaxis2=dict(title='Chi phí (₫)', overlaying='y', side='right'),
                legend=dict(orientation='h', y=-0.2))
            st.plotly_chart(fig4, use_container_width=True)

    tang_pct = total_tang_cost / total_gia_von * 100 if total_gia_von > 0 else 0
    st.markdown(f"""
    <div class="insight insight-warn">
    🎁 <b>Hàng tặng chiếm {tang_pct:.1f}%</b> tổng giá vốn hàng bán (<b>{total_tang_cost:,.0f}₫</b>).
    Chi phí ẩn này thường bị bỏ qua trong báo cáo — cần theo dõi chặt!
    </div>""", unsafe_allow_html=True)


# ── TAB 3 ──────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="sec-title">💸 Phân Tích Lợi Nhuận & Cơ Cấu Chi Phí</div>', unsafe_allow_html=True)

    prod_pnl = filtered.groupby('Ten_hang_hoa').agg(
        Tong_gia_von=('Tong_gia_von', 'sum'),
        Tien_ve_TK=('Tien_ve_TK', 'sum'),
        Loi_nhuan=('Loi_nhuan', 'sum'),
        So_luong=('So_luong', 'sum'),
        La_tang=('La_hang_tang', 'first')
    ).reset_index()
    prod_real = prod_pnl[~prod_pnl['La_tang']].copy()

    col1, col2 = st.columns(2)
    with col1:
        tp = prod_real.nlargest(12, 'Loi_nhuan')
        fig = px.bar(tp.sort_values('Loi_nhuan'), x='Loi_nhuan', y='Ten_hang_hoa',
                     orientation='h', title='✅ Top sản phẩm lãi nhiều nhất',
                     color='Loi_nhuan', color_continuous_scale='Greens',
                     labels={'Loi_nhuan': 'Lợi nhuận (₫)', 'Ten_hang_hoa': ''})
        fig.update_layout(height=460, **CHART_THEME, coloraxis_showscale=False)
        fig.update_traces(texttemplate='%{x:,.0f}₫', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        tl = prod_real.nsmallest(12, 'Loi_nhuan')
        fig2 = px.bar(tl.sort_values('Loi_nhuan', ascending=False),
                      x='Loi_nhuan', y='Ten_hang_hoa', orientation='h',
                      title='❌ Top sản phẩm lỗ nhiều nhất',
                      color='Loi_nhuan', color_continuous_scale='Reds_r',
                      labels={'Loi_nhuan': 'Lợi nhuận (₫)', 'Ten_hang_hoa': ''})
        fig2.add_vline(x=0, line_dash='dash', line_color='gray')
        fig2.update_layout(height=460, **CHART_THEME, coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    # ── WATERFALL ──────────────────────────────────────────────────────
    st.markdown('<div class="sec-title">🪣 Waterfall Dòng Tiền</div>', unsafe_allow_html=True)
    tv  = filtered['Tien_ve_TK'].sum()
    gv  = -filtered[~filtered['La_hang_tang']]['Tong_gia_von'].sum()
    tg  = -tang_items['Tong_gia_von'].sum()
    hv  = -total_hoan
    # FIX BUG 1: net phải cộng đủ 4 thành phần, bao gồm cả hàng hoàn (hv)
    net = tv + gv + tg + hv

    fig3 = go.Figure(go.Waterfall(
        measure=["absolute", "relative", "relative", "relative", "total"],
        x=["📥 Tiền về TK", "📦 (-) Giá vốn", "🎁 (-) Hàng tặng", "↩️ (-) Hoàn hàng", "💰 Lợi nhuận ròng"],
        y=[tv, gv, tg, hv, 0],
        connector={"line": {"color": "#e2e8f0"}},
        increasing={"marker": {"color": "#22c55e"}},
        decreasing={"marker": {"color": "#ef4444"}},
        totals={"marker": {"color": "#3b82f6"}},
        text=[f"{tv/1e6:.2f}M", f"{gv/1e6:.2f}M", f"{tg/1e6:.2f}M",
              f"{hv/1e6:.2f}M", f"{net/1e6:.2f}M"],
        textposition="outside"
    ))
    fig3.update_layout(title="💧 Từ Tiền Về TK → Lợi Nhuận Ròng (triệu ₫)",
                       height=420, **CHART_THEME)
    st.plotly_chart(fig3, use_container_width=True)

    # Ghi chú giải thích Lợi nhuận ròng vs Lợi nhuận KPI
    st.markdown(f"""
    <div class="insight insight-info">
    ℹ️ <b>Lợi nhuận ròng ({net/1e6:.2f}M₫)</b> = Tiền về TK − Giá vốn hàng bán − Chi phí hàng tặng − Giá vốn hàng hoàn.
    Khác với KPI "Lợi nhuận" ({total_loi_nhuan/1e6:.2f}M₫) vốn chỉ tính từ dữ liệu giao dịch, chưa trừ hoàn hàng.
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-title">📉 Ma Trận: Sản Lượng vs Lợi Nhuận</div>', unsafe_allow_html=True)
    sc = prod_real[prod_real['So_luong'] > 0].copy()
    sc['Bien_LN'] = sc.apply(
        lambda r: round(r['Loi_nhuan'] / r['Tong_gia_von'] * 100, 1) if r['Tong_gia_von'] != 0 else 0.0,
        axis=1
    )

    fig4 = px.scatter(sc, x='So_luong', y='Loi_nhuan',
                      size=sc['Tong_gia_von'].clip(lower=1),
                      color='Bien_LN', color_continuous_scale='RdYlGn',
                      hover_name='Ten_hang_hoa',
                      hover_data={'So_luong': True, 'Loi_nhuan': ':,.0f', 'Bien_LN': ':.1f'},
                      title='🔵 Sản lượng vs Lợi nhuận (kích cỡ bong = Giá vốn)',
                      labels={'So_luong': 'Sản lượng', 'Loi_nhuan': 'Lợi nhuận (₫)', 'Bien_LN': 'Biên LN%'})
    fig4.add_hline(y=0, line_dash='dash', line_color='#94a3b8')
    if len(sc) > 0:
        fig4.add_vline(x=sc['So_luong'].median(), line_dash='dot', line_color='#3b82f6', opacity=0.5)
    fig4.update_layout(height=450, **CHART_THEME)
    st.plotly_chart(fig4, use_container_width=True)

    n_loss = (prod_real['Loi_nhuan'] < 0).sum()
    if n_loss > 0:
        worst_p = prod_real.nsmallest(1, 'Loi_nhuan').iloc[0]
        st.markdown(f"""
        <div class="insight insight-bad">
        ⚠️ <b>{n_loss} sản phẩm</b> đang thua lỗ. SP lỗ nặng nhất: <b>{worst_p['Ten_hang_hoa']}</b>
        với mức lỗ <b>{worst_p['Loi_nhuan']:,.0f}₫</b>.
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown('<div class="insight insight-good">✅ Tất cả sản phẩm hiện tại đều có lợi nhuận dương!</div>',
                    unsafe_allow_html=True)


# ── TAB 4 ──────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="sec-title">↩️ Phân Tích Đơn Hoàn Hàng</div>', unsafe_allow_html=True)

    if len(hoan_data) > 0:
        hoan_pct = total_hoan / total_gia_von * 100 if total_gia_von > 0 else 0
        c1, c2, c3 = st.columns(3)
        with c1: st.markdown(kpi_card("↩️","Dòng SP Hoàn",f"{len(hoan_data)}","Tổng dòng ghi nhận","#ef4444"), unsafe_allow_html=True)
        with c2: st.markdown(kpi_card("💸","Giá Vốn Bị Hoàn",f"{total_hoan/1e6:.2f}M₫","Tiền vốn đang bị treo","#f59e0b"), unsafe_allow_html=True)
        with c3: st.markdown(kpi_card("📊","Tỷ Lệ / Giá Vốn",f"{hoan_pct:.1f}%","So với tổng giá vốn hàng bán","#8b5cf6"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        hoan_prod = hoan_data.groupby('Ten_hang')['Tong_gia_von'].sum().reset_index().sort_values('Tong_gia_von', ascending=False)
        if len(hoan_prod) > 0:
            fig = px.bar(hoan_prod, x='Ten_hang', y='Tong_gia_von',
                         title='↩️ Giá vốn bị hoàn theo sản phẩm',
                         color='Tong_gia_von', color_continuous_scale='Reds',
                         labels={'Ten_hang': 'Sản phẩm', 'Tong_gia_von': 'Giá vốn (₫)'})
            fig.update_layout(height=360, **CHART_THEME, coloraxis_showscale=False, xaxis_tickangle=-25)
            fig.update_traces(texttemplate='%{y:,.0f}₫', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)

        hd = hoan_data[['Ten_hang', 'So_luong', 'Gia_von', 'Tong_gia_von']].copy()
        hd['Gia_von']      = hd['Gia_von'].apply(lambda x: f"{x:,.0f}₫")
        hd['Tong_gia_von'] = hd['Tong_gia_von'].apply(lambda x: f"{x:,.0f}₫")
        hd.columns = ['Tên hàng', 'SL hoàn', 'Giá vốn/SP', 'Tổng giá vốn']
        st.dataframe(hd, use_container_width=True, hide_index=True)

        st.markdown(f"""
        <div class="insight insight-bad">
        ↩️ Tỷ lệ hoàn <b>{hoan_pct:.1f}%</b> = <b>{total_hoan:,.0f}₫</b> giá vốn đang bị treo.
        Cần phân tích lý do hoàn để cải thiện — mỗi đơn hoàn là chi phí cơ hội bị mất.
        </div>""", unsafe_allow_html=True)
    else:
        st.info("✅ Không tìm thấy dữ liệu hoàn hàng trong file này.")

    st.markdown('<div class="sec-title">⚠️ Ma Trận Rủi Ro Sản Phẩm</div>', unsafe_allow_html=True)
    prod_risk = filtered[~filtered['La_hang_tang']].groupby('Ten_hang_hoa').agg(
        GV=('Tong_gia_von', 'sum'), LN=('Loi_nhuan', 'sum'), SL=('So_luong', 'sum')
    ).reset_index()

    def classify_risk(row):
        if row['LN'] >= 0: return 'An toàn ✅'
        if row['GV'] > 500000: return 'Rủi ro cao 🔴'
        if row['GV'] > 100000: return 'Rủi ro TB 🟡'
        return 'Rủi ro thấp 🟢'

    prod_risk['Rui_ro'] = prod_risk.apply(classify_risk, axis=1)
    rc = prod_risk['Rui_ro'].value_counts().reset_index()
    rc.columns = ['Mức độ', 'Số SP']
    color_map_r = {'An toàn ✅': '#22c55e', 'Rủi ro thấp 🟢': '#84cc16',
                   'Rủi ro TB 🟡': '#f59e0b', 'Rủi ro cao 🔴': '#ef4444'}

    col_r1, col_r2 = st.columns([1, 1.3])
    with col_r1:
        fig_r = px.pie(rc, values='Số SP', names='Mức độ',
                       title='📊 Phân bổ rủi ro sản phẩm',
                       color='Mức độ', color_discrete_map=color_map_r, hole=0.42)
        fig_r.update_layout(height=360, **CHART_THEME)
        fig_r.update_traces(textinfo='label+percent+value')
        st.plotly_chart(fig_r, use_container_width=True)

    with col_r2:
        high = prod_risk[prod_risk['Rui_ro'] == 'Rủi ro cao 🔴'].sort_values('LN').head(10)
        st.markdown("**🔴 Sản phẩm rủi ro cao — cần xử lý ngay:**")
        if len(high) > 0:
            for _, row in high.iterrows():
                nm = str(row['Ten_hang_hoa'])
                nm = nm[:48] + '…' if len(nm) > 48 else nm
                st.markdown(f"""
                <div class="risk-item" style="border-left-color:#ef4444">
                    ❌ <b>{nm}</b><br>
                    <span style="color:#94a3b8;font-size:0.8rem">
                    Lỗ: <b style="color:#ef4444">{row['LN']:,.0f}₫</b> &nbsp;|&nbsp;
                    GV: {row['GV']:,.0f}₫ &nbsp;|&nbsp; SL: {row['SL']:.0f}
                    </span>
                </div>""", unsafe_allow_html=True)
        else:
            st.success("✅ Không có sản phẩm ở mức rủi ro cao!")


# ── TAB 5 ──────────────────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="sec-title">🔍 Tra Cứu & Xuất Dữ Liệu</div>', unsafe_allow_html=True)

    fc1, fc2, fc3, fc4 = st.columns([2, 1, 1, 1])
    with fc1: search = st.text_input("🔎 Tìm sản phẩm / khách hàng", placeholder="Nhập từ khoá...")
    with fc2: san_det = st.multiselect("Sàn", available_sans, default=available_sans, key='det_san')
    with fc3: type_det = st.selectbox("Loại hàng", ['Tất cả', 'Hàng bán', 'Hàng tặng'])
    with fc4: pnl_det  = st.selectbox("Lợi nhuận", ['Tất cả', 'Có lãi ✅', 'Thua lỗ ❌'])

    det = all_data[all_data['San'].isin(san_det)].copy() if san_det else all_data.copy()
    if type_det == 'Hàng bán':   det = det[~det['La_hang_tang']]
    elif type_det == 'Hàng tặng': det = det[det['La_hang_tang']]
    if pnl_det == 'Có lãi ✅':    det = det[det['Loi_nhuan'] >= 0]
    elif pnl_det == 'Thua lỗ ❌': det = det[det['Loi_nhuan'] < 0]
    if search:
        mask = (
            det['Ten_hang_hoa'].astype(str).str.contains(search, case=False, na=False) |
            det.get('Khach_hang', pd.Series(dtype=str)).astype(str).str.contains(search, case=False, na=False)
        )
        det = det[mask]

    total_det_ln = det['Loi_nhuan'].sum()
    st.markdown(f"""
    <div style="background:white;border-radius:10px;padding:12px 18px;margin-bottom:12px;
                box-shadow:0 1px 6px rgba(0,0,0,0.06);font-size:0.85rem;color:#475569">
    📊 <b>{len(det):,}</b> dòng &nbsp;|&nbsp;
    🛒 SL: <b>{det['So_luong'].sum():.0f}</b> &nbsp;|&nbsp;
    💸 GV: <b>{det['Tong_gia_von'].sum():,.0f}₫</b> &nbsp;|&nbsp;
    📥 TK: <b>{det['Tien_ve_TK'].sum():,.0f}₫</b> &nbsp;|&nbsp;
    📊 LN: <b style="color:{'#22c55e' if total_det_ln>=0 else '#ef4444'}">{total_det_ln:,.0f}₫</b>
    </div>""", unsafe_allow_html=True)

    show_cols = [c for c in ['San', 'Ten_hang_hoa', 'Khach_hang', 'So_luong', 'Gia_von',
                              'Tong_gia_von', 'Tien_ve_TK', 'Loi_nhuan', 'La_hang_tang'] if c in det.columns]
    det_disp = det[show_cols].copy()
    det_disp['La_hang_tang'] = det_disp['La_hang_tang'].map({True: '🎁 Tặng', False: '✅ Bán'})
    rename_det = {'San':'Sàn','Ten_hang_hoa':'Tên hàng','Khach_hang':'Khách',
                  'So_luong':'SL','Gia_von':'GV/SP','Tong_gia_von':'T.GV',
                  'Tien_ve_TK':'Tiền về TK','Loi_nhuan':'Lợi nhuận','La_hang_tang':'Loại'}
    det_disp.rename(columns=rename_det, inplace=True)
    st.dataframe(det_disp, use_container_width=True, height=480, hide_index=True)

    csv_buf = det.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
    st.download_button(
        "⬇️ Tải CSV (theo bộ lọc hiện tại)", data=csv_buf,
        file_name=f"tmdt_{uploaded.name.replace('.xlsx','')}_filtered.csv",
        mime='text/csv'
    )

    if len(sl_data) > 0 and 'Thuong_hieu' in sl_data.columns:
        st.markdown('<div class="sec-title">🏷️ Tổng Hợp Theo Thương Hiệu</div>', unsafe_allow_html=True)
        sl_b = sl_data.copy()
        sl_b['TH'] = sl_b['Thuong_hieu'].ffill()
        # FIX: loại bỏ tên sàn bị lọt vào cột thương hiệu ở footer
        platform_names = {'tiktok shop', 'shopee', 'lazada', 'tiktokshop'}
        sl_b = sl_b[~sl_b['TH'].astype(str).str.strip().str.lower().isin(platform_names)]
        bs = sl_b[~sl_b['La_hang_tang']].groupby('TH').agg(
            So_SP=('Ten_hang', 'count'), Tong_SL=('Tong_SL', 'sum')
        ).reset_index().sort_values('Tong_SL', ascending=False).head(15)
        bs.columns = ['Thương hiệu', 'Số mã SP', 'Tổng SL']
        fig_b = px.bar(bs, x='Thương hiệu', y='Tổng SL',
                       title='🏷️ Sản lượng bán theo thương hiệu',
                       color='Tổng SL', color_continuous_scale='Purples',
                       text='Tổng SL')
        fig_b.update_layout(height=380, **CHART_THEME, coloraxis_showscale=False, xaxis_tickangle=-30)
        fig_b.update_traces(texttemplate='%{text:.0f}', textposition='outside')
        st.plotly_chart(fig_b, use_container_width=True)


# ─── FOOTER ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"""
<div style="text-align:center;color:#94a3b8;font-size:0.78rem;padding:10px 0">
    📊 Phân tích TMĐT Online &nbsp;|&nbsp;
    <code>{uploaded.name}</code> &nbsp;|&nbsp;
    Sàn: {" · ".join(available_sans)} &nbsp;|&nbsp;
    {len(all_data):,} giao dịch được phân tích
</div>""", unsafe_allow_html=True)
