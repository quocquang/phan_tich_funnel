import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="📊 Báo Cáo TMĐT Tháng 02/2026",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Be Vietnam Pro', sans-serif; }

    .main { background: #f8f9fc; }

    .header-banner {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 16px;
        padding: 28px 36px;
        margin-bottom: 28px;
        color: white;
        display: flex;
        align-items: center;
        gap: 20px;
    }
    .header-banner h1 { font-size: 1.9rem; font-weight: 700; margin: 0; }
    .header-banner p { font-size: 0.95rem; opacity: 0.75; margin: 4px 0 0 0; }

    .kpi-card {
        background: white;
        border-radius: 14px;
        padding: 20px 22px;
        border-left: 5px solid;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07);
        margin-bottom: 4px;
        transition: transform 0.2s;
    }
    .kpi-card:hover { transform: translateY(-2px); }
    .kpi-label { font-size: 0.78rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: #888; margin-bottom: 6px; }
    .kpi-value { font-size: 1.75rem; font-weight: 700; line-height: 1.1; }
    .kpi-sub { font-size: 0.8rem; color: #aaa; margin-top: 4px; }

    .section-title {
        font-size: 1.15rem; font-weight: 700; color: #1a1a2e;
        margin: 28px 0 14px 0; padding-bottom: 8px;
        border-bottom: 2px solid #e8ecf5;
    }

    .insight-box {
        background: linear-gradient(135deg, #fff7e6, #fff3d6);
        border-left: 4px solid #f59e0b;
        border-radius: 10px;
        padding: 14px 18px;
        margin: 10px 0;
        font-size: 0.88rem;
    }
    .insight-negative {
        background: linear-gradient(135deg, #fff1f2, #ffe8ea);
        border-left: 4px solid #ef4444;
    }
    .insight-positive {
        background: linear-gradient(135deg, #f0fff4, #e6ffed);
        border-left: 4px solid #22c55e;
    }

    .platform-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 2px;
    }
    .badge-tiktok { background: #1a1a2e; color: #fe2c55; border: 1px solid #fe2c55; }
    .badge-shopee { background: #ff5722; color: white; }
    .badge-lazada { background: #0f146b; color: #ff6900; border: 1px solid #ff6900; }

    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #f0f2f8; border-radius: 10px; padding: 8px 20px;
        font-weight: 600; color: #555; border: none !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1a1a2e, #0f3460) !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)


# ─── DATA LOADING ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    file = "Doanh_số_bán_hàng_ONLINE_Tháng_02_2026.xlsx"
    try:
        all_sheets = pd.read_excel(file, sheet_name=None, header=None)
    except:
        # fallback path
        all_sheets = pd.read_excel(
            "/mnt/user-data/uploads/Doanh_số_bán_hàng_ONLINE_Tháng_02_2026.xlsx",
            sheet_name=None, header=None
        )
    return all_sheets

@st.cache_data
def parse_sales_sheet(all_sheets, sheet_name, san_label):
    df = all_sheets[sheet_name]
    header_row = None
    for i, row in df.iterrows():
        if 'STT' in str(row.values) and 'Tên hàng hoá' in str(row.values):
            header_row = i
            break
    if header_row is None:
        return pd.DataFrame()

    data_rows = df.iloc[header_row + 1:].copy()
    data_rows.columns = range(len(data_rows.columns))

    col_map = {0:'STT', 1:'Khach_hang', 2:'Ma_don_hang', 3:'Ma_san_pham',
               4:'Ten_hang_hoa', 5:'So_luong', 6:'Gia_von', 7:'Tong_gia_von',
               8:'Tien_ve_TK_dk', 9:'Doanh_thu_thuc', 10:'Ngay', 11:'Loi_nhuan', 12:'Trang_thai'}

    cols_available = {k: v for k, v in col_map.items() if k < len(data_rows.columns)}
    df_clean = data_rows[[c for c in cols_available.keys()]].copy()
    df_clean.columns = list(cols_available.values())
    df_clean = df_clean[df_clean['Ten_hang_hoa'].notna() & (df_clean['Ten_hang_hoa'].astype(str) != 'nan')]
    df_clean['San'] = san_label

    is_tang = df_clean['Ma_san_pham'].astype(str).str.endswith('-T', na=False)
    df_clean['La_hang_tang'] = is_tang

    for col in ['So_luong', 'Gia_von', 'Tong_gia_von', 'Tien_ve_TK_dk', 'Loi_nhuan']:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    if 'Ngay' in df_clean.columns:
        df_clean['Ngay'] = pd.to_datetime(df_clean['Ngay'], errors='coerce')

    return df_clean

@st.cache_data
def parse_sl_sheet(all_sheets):
    df = all_sheets['Tổng SL bán ra']
    sl = df.iloc[4:].copy()
    sl.columns = range(len(sl.columns))
    sl = sl.rename(columns={0:'STT', 1:'Thuong_hieu', 2:'Ma_sp', 4:'Ten_hang',
                             5:'Tiktok', 6:'Shopee', 7:'Lazada', 8:'Tong_SL', 9:'Gia_von'})
    sl = sl[sl['Ten_hang'].notna() & (sl['Ten_hang'].astype(str) != 'nan')]
    sl['Tong_SL'] = pd.to_numeric(sl['Tong_SL'], errors='coerce').fillna(0)
    sl['Gia_von'] = pd.to_numeric(sl['Gia_von'], errors='coerce').fillna(0)
    for c in ['Tiktok', 'Shopee', 'Lazada']:
        sl[c] = pd.to_numeric(sl[c], errors='coerce').fillna(0)
    sl = sl[sl['Tong_SL'] > 0]
    # Exclude summary rows
    sl = sl[~sl['STT'].astype(str).str.upper().str.contains('TỔNG|TOTAL|NAN', na=False)]
    return sl

@st.cache_data
def parse_hoan_hang(all_sheets):
    df = all_sheets['ĐƠN HOÀN HÀNG']
    header_row = None
    for i, row in df.iterrows():
        if 'STT' in str(row.values) and 'Tên hàng hoá' in str(row.values):
            header_row = i
            break
    if header_row is None:
        return pd.DataFrame()
    data_rows = df.iloc[header_row + 1:].copy()
    data_rows.columns = range(len(data_rows.columns))
    col_map = {0:'San', 1:'STT', 2:'Khach_hang', 3:'Ma_don_hang', 4:'Ma_sp',
               5:'Ten_hang', 6:'So_luong', 7:'Gia_von', 8:'Tong_gia_von', 10:'Ngay_hoan', 11:'Von_con_lai'}
    cols_av = {k: v for k, v in col_map.items() if k < len(data_rows.columns)}
    df_c = data_rows[[c for c in cols_av.keys()]].copy()
    df_c.columns = list(cols_av.values())
    df_c = df_c[df_c['Ten_hang'].notna() & (df_c['Ten_hang'].astype(str) != 'nan')]
    df_c['Tong_gia_von'] = pd.to_numeric(df_c['Tong_gia_von'], errors='coerce')
    df_c['So_luong'] = pd.to_numeric(df_c['So_luong'], errors='coerce')
    df_c = df_c[df_c['Tong_gia_von'] > 0]
    return df_c


# ─── LOAD ──────────────────────────────────────────────────────────────────────
all_sheets = load_data()
tt = parse_sales_sheet(all_sheets, 'TIKTOKSHOP', 'TiktokShop')
sp = parse_sales_sheet(all_sheets, 'SHOPEE', 'Shopee')
lz = parse_sales_sheet(all_sheets, 'LAZADA', 'Lazada')
all_data = pd.concat([tt, sp, lz], ignore_index=True)
sl_data = parse_sl_sheet(all_sheets)
hoan_data = parse_hoan_hang(all_sheets)

# Separate actual sales vs gift items
real_sales = all_data[~all_data['La_hang_tang']]
tang_items = all_data[all_data['La_hang_tang']]

# ─── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
  <div>
    <h1>🛍️ Phân Tích Doanh Số Bán Hàng Online</h1>
    <p>📅 Tháng 02/2026 &nbsp;|&nbsp; Sàn TMĐT: TiktokShop · Shopee · Lazada &nbsp;|&nbsp; Cập nhật tự động từ file Excel</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── SIDEBAR FILTER ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎛️ Bộ lọc dữ liệu")
    san_filter = st.multiselect(
        "Chọn sàn TMĐT", options=['TiktokShop', 'Shopee', 'Lazada'],
        default=['TiktokShop', 'Shopee', 'Lazada']
    )
    include_tang = st.checkbox("Bao gồm hàng tặng", value=False)
    st.markdown("---")
    st.markdown("**📊 Tổng quan nhanh**")
    total_orders = all_data[all_data['San'].isin(san_filter)]['Ma_don_hang'].nunique()
    st.metric("Đơn hàng (dòng SP)", len(all_data[all_data['San'].isin(san_filter)]))
    st.metric("Sản phẩm bán ra", int(all_data[all_data['San'].isin(san_filter)]['So_luong'].sum()))
    st.markdown("---")
    st.markdown("### 📌 Về báo cáo")
    st.info("Dữ liệu từ file Excel tháng 02/2026. Phân tích bao gồm: doanh thu, giá vốn, lợi nhuận, hàng tặng, hoàn hàng và top sản phẩm.")

# Apply filters
filtered = all_data[all_data['San'].isin(san_filter)]
if not include_tang:
    filtered_main = filtered[~filtered['La_hang_tang']]
else:
    filtered_main = filtered

# ─── KPI SECTION ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">📈 Chỉ Số Tổng Quan Tháng 02/2026</div>', unsafe_allow_html=True)

total_gia_von = filtered_main['Tong_gia_von'].sum()
total_tien_ve = filtered_main['Tien_ve_TK_dk'].sum()
total_loi_nhuan = filtered_main['Loi_nhuan'].sum()
total_sl = filtered_main['So_luong'].sum()
total_tang_cost = tang_items[tang_items['San'].isin(san_filter)]['Tong_gia_von'].sum()
total_hoan = hoan_data['Tong_gia_von'].sum() if len(hoan_data) > 0 else 0
profit_margin = (total_loi_nhuan / total_tien_ve * 100) if total_tien_ve > 0 else 0

c1, c2, c3, c4, c5, c6 = st.columns(6)
kpis = [
    (c1, "💰 Tiền Về TK (Dự kiến)", f"{total_tien_ve:,.0f}₫", "Tổng tiền dự kiến nhận về", "#3b82f6"),
    (c2, "📦 Tổng Giá Vốn", f"{total_gia_von:,.0f}₫", "Chi phí hàng bán ra", "#8b5cf6"),
    (c3, "📊 Lợi Nhuận", f"{total_loi_nhuan:,.0f}₫", f"Biên LN: {profit_margin:.1f}%", "#22c55e" if total_loi_nhuan > 0 else "#ef4444"),
    (c4, "🛒 Sản Lượng Bán", f"{int(total_sl):,}", "Tổng số lượng SP", "#f59e0b"),
    (c5, "🎁 Chi Phí Hàng Tặng", f"{total_tang_cost:,.0f}₫", "Giá vốn hàng tặng kèm", "#ec4899"),
    (c6, "↩️ Hàng Hoàn", f"{total_hoan:,.0f}₫", f"{len(hoan_data)} dòng SP hoàn", "#ef4444"),
]
for col, label, val, sub, color in kpis:
    with col:
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color:{color}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value" style="color:{color}">{val}</div>
            <div class="kpi-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

# ─── TABS ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏪 Hiệu suất theo Sàn",
    "📦 Phân tích Sản phẩm",
    "💸 Lợi nhuận & Chi phí",
    "↩️ Hoàn hàng & Rủi ro",
    "🔍 Chi tiết dữ liệu"
])


# ══════════════════════════════════════════════════════════════════════
# TAB 1: Hiệu suất theo Sàn
# ══════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">🏪 So Sánh Hiệu Suất Các Sàn TMĐT</div>', unsafe_allow_html=True)

    platform_stats = filtered_main.groupby('San').agg(
        Tong_gia_von=('Tong_gia_von', 'sum'),
        Tien_ve_TK=('Tien_ve_TK_dk', 'sum'),
        Loi_nhuan=('Loi_nhuan', 'sum'),
        So_luong=('So_luong', 'sum'),
        So_dong=('Ten_hang_hoa', 'count')
    ).reset_index()
    platform_stats['Bien_LN_%'] = (platform_stats['Loi_nhuan'] / platform_stats['Tien_ve_TK'] * 100).round(1)
    platform_stats['Gia_von_TB'] = (platform_stats['Tong_gia_von'] / platform_stats['So_luong']).round(0)

    colors_san = {'TiktokShop': '#fe2c55', 'Shopee': '#ff5722', 'Lazada': '#ff6900'}
    color_list = [colors_san.get(s, '#888') for s in platform_stats['San']]

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_bar(
            name='Giá Vốn', x=platform_stats['San'], y=platform_stats['Tong_gia_von'],
            marker_color='#94a3b8', text=platform_stats['Tong_gia_von'].apply(lambda x: f"{x/1e6:.2f}tr"),
            textposition='outside'
        )
        fig.add_bar(
            name='Tiền về TK', x=platform_stats['San'], y=platform_stats['Tien_ve_TK'],
            marker_color='#3b82f6', text=platform_stats['Tien_ve_TK'].apply(lambda x: f"{x/1e6:.2f}tr"),
            textposition='outside'
        )
        fig.update_layout(
            title='💰 Giá vốn vs Tiền về TK (triệu VNĐ)',
            barmode='group', height=380,
            plot_bgcolor='white', paper_bgcolor='white',
            legend=dict(orientation='h', y=-0.2),
            font=dict(family='Be Vietnam Pro')
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = go.Figure()
        bar_colors = ['#22c55e' if v >= 0 else '#ef4444' for v in platform_stats['Loi_nhuan']]
        fig2.add_bar(
            x=platform_stats['San'], y=platform_stats['Loi_nhuan'],
            marker_color=bar_colors,
            text=platform_stats['Loi_nhuan'].apply(lambda x: f"{x/1e6:.2f}tr"),
            textposition='outside'
        )
        fig2.add_hline(y=0, line_dash='dash', line_color='gray', line_width=1)
        fig2.update_layout(
            title='📊 Lợi nhuận theo sàn (triệu VNĐ)',
            height=380, plot_bgcolor='white', paper_bgcolor='white',
            font=dict(family='Be Vietnam Pro')
        )
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig3 = px.pie(
            platform_stats, values='So_luong', names='San',
            title='🛒 Phân bổ sản lượng bán theo sàn',
            color='San', color_discrete_map=colors_san,
            hole=0.45
        )
        fig3.update_layout(height=350, font=dict(family='Be Vietnam Pro'))
        fig3.update_traces(textinfo='label+percent+value')
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        fig4 = px.pie(
            platform_stats, values='Tong_gia_von', names='San',
            title='💸 Phân bổ Giá vốn theo sàn',
            color='San', color_discrete_map=colors_san,
            hole=0.45
        )
        fig4.update_layout(height=350, font=dict(family='Be Vietnam Pro'))
        fig4.update_traces(textinfo='label+percent')
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown('<div class="section-title">📋 Bảng tổng hợp theo sàn</div>', unsafe_allow_html=True)
    display_ps = platform_stats.copy()
    display_ps['Tong_gia_von'] = display_ps['Tong_gia_von'].apply(lambda x: f"{x:,.0f}₫")
    display_ps['Tien_ve_TK'] = display_ps['Tien_ve_TK'].apply(lambda x: f"{x:,.0f}₫")
    display_ps['Loi_nhuan'] = display_ps['Loi_nhuan'].apply(lambda x: f"{x:,.0f}₫")
    display_ps['Bien_LN_%'] = display_ps['Bien_LN_%'].apply(lambda x: f"{x:.1f}%")
    display_ps.columns = ['Sàn TMĐT', 'Tổng Giá Vốn', 'Tiền Về TK', 'Lợi Nhuận', 'SL Bán', 'Số dòng SP', 'Biên LN%', 'Giá vốn TB']
    st.dataframe(display_ps, use_container_width=True, hide_index=True)

    # Insight
    best_san = platform_stats.loc[platform_stats['Loi_nhuan'].idxmax(), 'San']
    worst_san = platform_stats.loc[platform_stats['Loi_nhuan'].idxmin(), 'San']
    st.markdown(f"""
    <div class="insight-box insight-positive">
    ✅ <strong>Sàn tốt nhất:</strong> <strong>{best_san}</strong> có lợi nhuận cao nhất trong tháng 02/2026.
    </div>
    <div class="insight-box insight-negative">
    ⚠️ <strong>Cần chú ý:</strong> <strong>{worst_san}</strong> đang có lợi nhuận âm — cần xem xét lại chiến lược giá, chi phí hàng tặng và cách chạy flash sale.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 2: Phân tích Sản phẩm
# ══════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">📦 Phân Tích Sản Phẩm Chi Tiết</div>', unsafe_allow_html=True)

    col_f1, col_f2 = st.columns([2, 1])
    with col_f1:
        top_n = st.slider("Số sản phẩm hiển thị Top", 5, 30, 15)
    with col_f2:
        metric_choice = st.selectbox("Xếp hạng theo", ['Sản lượng', 'Giá vốn', 'Lợi nhuận'])

    # Product aggregation (real sales only = SL data)
    sl_real = sl_data[~sl_data['Ten_hang'].astype(str).str.contains('Hàng tặng', na=False)]
    sl_tang = sl_data[sl_data['Ten_hang'].astype(str).str.contains('Hàng tặng', na=False)]

    metric_col = {'Sản lượng': 'Tong_SL', 'Giá vốn': 'Gia_von', 'Lợi nhuận': 'Tong_SL'}[metric_choice]

    top_products = sl_real.nlargest(top_n, 'Tong_SL')

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            top_products.sort_values('Tong_SL'),
            x='Tong_SL', y='Ten_hang',
            orientation='h',
            title=f'🏆 Top {top_n} sản phẩm bán chạy nhất (theo sản lượng)',
            color='Tong_SL',
            color_continuous_scale='Blues',
            labels={'Tong_SL': 'Sản lượng', 'Ten_hang': 'Sản phẩm'}
        )
        fig.update_layout(
            height=520, plot_bgcolor='white', paper_bgcolor='white',
            showlegend=False, font=dict(family='Be Vietnam Pro'),
            yaxis={'categoryorder': 'total ascending'},
            coloraxis_showscale=False
        )
        fig.update_traces(texttemplate='%{x}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Platform breakdown for top products
        top_names = top_products['Ten_hang'].head(10).tolist()
        top_with_platform = top_products[top_products['Ten_hang'].isin(top_names)][
            ['Ten_hang', 'Tiktok', 'Shopee', 'Lazada']
        ].melt(id_vars='Ten_hang', var_name='San', value_name='SL')
        top_with_platform = top_with_platform[top_with_platform['SL'] > 0]

        fig2 = px.bar(
            top_with_platform, x='SL', y='Ten_hang', color='San',
            orientation='h',
            title='🛒 Phân bổ sản lượng theo sàn (Top 10 SP)',
            color_discrete_map={'Tiktok': '#fe2c55', 'Shopee': '#ff5722', 'Lazada': '#ff6900'},
            barmode='stack'
        )
        fig2.update_layout(
            height=520, plot_bgcolor='white', paper_bgcolor='white',
            font=dict(family='Be Vietnam Pro'),
            yaxis={'categoryorder': 'total ascending'},
            legend=dict(orientation='h', y=-0.15)
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Hàng tặng analysis
    st.markdown('<div class="section-title">🎁 Phân Tích Hàng Tặng Kèm</div>', unsafe_allow_html=True)
    tang_cost_by_product = tang_items[tang_items['San'].isin(san_filter)].groupby('Ten_hang_hoa').agg(
        SL=('So_luong', 'sum'),
        Chi_phi=('Tong_gia_von', 'sum')
    ).reset_index().sort_values('Chi_phi', ascending=False).head(15)

    col3, col4 = st.columns(2)
    with col3:
        if len(tang_cost_by_product) > 0:
            fig3 = px.bar(
                tang_cost_by_product.sort_values('Chi_phi'),
                x='Chi_phi', y='Ten_hang_hoa',
                orientation='h',
                title='💸 Chi phí giá vốn hàng tặng theo loại (VNĐ)',
                color='Chi_phi', color_continuous_scale='Oranges',
                labels={'Chi_phi': 'Tổng chi phí (VNĐ)', 'Ten_hang_hoa': 'Hàng tặng'}
            )
            fig3.update_layout(height=400, plot_bgcolor='white', paper_bgcolor='white',
                               coloraxis_showscale=False, font=dict(family='Be Vietnam Pro'))
            fig3.update_traces(texttemplate='%{x:,.0f}₫', textposition='outside')
            st.plotly_chart(fig3, use_container_width=True)

    with col4:
        tang_by_san = tang_items[tang_items['San'].isin(san_filter)].groupby('San').agg(
            SL=('So_luong', 'sum'),
            Chi_phi=('Tong_gia_von', 'sum')
        ).reset_index()
        if len(tang_by_san) > 0:
            fig4 = go.Figure()
            fig4.add_bar(name='Sản lượng tặng', x=tang_by_san['San'], y=tang_by_san['SL'],
                         marker_color='#f59e0b', yaxis='y')
            fig4.add_scatter(name='Chi phí (VNĐ)', x=tang_by_san['San'], y=tang_by_san['Chi_phi'],
                             mode='lines+markers+text', line=dict(color='#ef4444', width=2),
                             marker=dict(size=10), yaxis='y2',
                             text=tang_by_san['Chi_phi'].apply(lambda x: f"{x/1e6:.2f}tr"),
                             textposition='top center')
            fig4.update_layout(
                title='🎁 Hàng tặng: Sản lượng & Chi phí theo sàn',
                height=400, plot_bgcolor='white', paper_bgcolor='white',
                yaxis=dict(title='Sản lượng'),
                yaxis2=dict(title='Chi phí (VNĐ)', overlaying='y', side='right'),
                legend=dict(orientation='h', y=-0.2),
                font=dict(family='Be Vietnam Pro')
            )
            st.plotly_chart(fig4, use_container_width=True)

    tang_pct = total_tang_cost / total_gia_von * 100 if total_gia_von > 0 else 0
    st.markdown(f"""
    <div class="insight-box">
    🎁 <strong>Hàng tặng chiếm {tang_pct:.1f}%</strong> tổng giá vốn ({total_tang_cost:,.0f}₫). 
    Đây là chi phí ẩn quan trọng khi tính lợi nhuận thực — cần theo dõi chặt để tránh mất kiểm soát chi phí.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 3: Lợi nhuận & Chi phí
# ══════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">💸 Phân Tích Lợi Nhuận & Cơ Cấu Chi Phí</div>', unsafe_allow_html=True)

    # Profitable vs loss products
    prod_pnl = filtered_main.groupby('Ten_hang_hoa').agg(
        Tong_gia_von=('Tong_gia_von', 'sum'),
        Tien_ve_TK=('Tien_ve_TK_dk', 'sum'),
        Loi_nhuan=('Loi_nhuan', 'sum'),
        So_luong=('So_luong', 'sum'),
        La_tang=('La_hang_tang', 'first')
    ).reset_index()
    prod_pnl_real = prod_pnl[~prod_pnl['La_tang']]

    col1, col2 = st.columns(2)
    with col1:
        top_profit = prod_pnl_real.nlargest(12, 'Loi_nhuan')
        fig = px.bar(
            top_profit.sort_values('Loi_nhuan'),
            x='Loi_nhuan', y='Ten_hang_hoa',
            orientation='h',
            title='✅ Top sản phẩm có lợi nhuận cao nhất',
            color='Loi_nhuan', color_continuous_scale='Greens',
            labels={'Loi_nhuan': 'Lợi nhuận (VNĐ)', 'Ten_hang_hoa': ''}
        )
        fig.update_layout(height=450, plot_bgcolor='white', paper_bgcolor='white',
                          coloraxis_showscale=False, font=dict(family='Be Vietnam Pro'))
        fig.update_traces(texttemplate='%{x:,.0f}₫', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        top_loss = prod_pnl_real.nsmallest(12, 'Loi_nhuan')
        fig2 = px.bar(
            top_loss.sort_values('Loi_nhuan', ascending=False),
            x='Loi_nhuan', y='Ten_hang_hoa',
            orientation='h',
            title='❌ Top sản phẩm lỗ nặng nhất',
            color='Loi_nhuan', color_continuous_scale='Reds_r',
            labels={'Loi_nhuan': 'Lợi nhuận (VNĐ)', 'Ten_hang_hoa': ''}
        )
        fig2.update_layout(height=450, plot_bgcolor='white', paper_bgcolor='white',
                           coloraxis_showscale=False, font=dict(family='Be Vietnam Pro'))
        fig2.add_vline(x=0, line_dash='dash', line_color='gray')
        st.plotly_chart(fig2, use_container_width=True)

    # Waterfall chart of cost structure
    st.markdown('<div class="section-title">🪣 Cơ Cấu Dòng Tiền (Waterfall)</div>', unsafe_allow_html=True)

    total_tv = filtered_main['Tien_ve_TK_dk'].sum()
    total_gv = -filtered_main[~filtered_main['La_hang_tang']]['Tong_gia_von'].sum()
    total_tang_gv = -tang_items[tang_items['San'].isin(san_filter)]['Tong_gia_von'].sum()
    total_hoan_v = -total_hoan
    net = total_tv + total_gv + total_tang_gv

    fig3 = go.Figure(go.Waterfall(
        name="Dòng tiền", orientation="v",
        measure=["absolute", "relative", "relative", "relative", "total"],
        x=["Tiền về TK (dự kiến)", "(-) Giá vốn hàng bán", "(-) Chi phí hàng tặng", "(-) Hàng hoàn", "Lợi nhuận ròng"],
        y=[total_tv, total_gv, total_tang_gv, total_hoan_v, 0],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "#22c55e"}},
        decreasing={"marker": {"color": "#ef4444"}},
        totals={"marker": {"color": "#3b82f6"}},
        text=[f"{total_tv/1e6:.1f}tr", f"{total_gv/1e6:.1f}tr",
              f"{total_tang_gv/1e6:.1f}tr", f"{total_hoan_v/1e6:.1f}tr",
              f"{net/1e6:.1f}tr"],
        textposition="outside"
    ))
    fig3.update_layout(
        title="💧 Waterfall: Từ Tiền Về TK → Lợi Nhuận Ròng",
        height=420, plot_bgcolor='white', paper_bgcolor='white',
        font=dict(family='Be Vietnam Pro')
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Scatter: Volume vs Profit
    st.markdown('<div class="section-title">📉 Ma Trận Sản Lượng vs Lợi Nhuận</div>', unsafe_allow_html=True)
    scatter_data = prod_pnl_real[prod_pnl_real['So_luong'] > 0].copy()
    scatter_data['Bien_LN'] = (scatter_data['Loi_nhuan'] / scatter_data['Tong_gia_von'] * 100).round(1)
    scatter_data['Ten_ngan'] = scatter_data['Ten_hang_hoa'].apply(lambda x: x[:30] + '...' if len(str(x)) > 30 else x)

    fig4 = px.scatter(
        scatter_data, x='So_luong', y='Loi_nhuan',
        size=scatter_data['Tong_gia_von'].clip(lower=1),
        color='Bien_LN', color_continuous_scale='RdYlGn',
        hover_name='Ten_hang_hoa',
        hover_data={'So_luong': True, 'Loi_nhuan': ':,.0f', 'Tong_gia_von': ':,.0f', 'Bien_LN': ':.1f'},
        title='📉 Phân tán: Sản lượng vs Lợi nhuận (kích cỡ = Giá vốn)',
        labels={'So_luong': 'Sản lượng', 'Loi_nhuan': 'Lợi nhuận (VNĐ)', 'Bien_LN': 'Biên LN%'}
    )
    fig4.add_hline(y=0, line_dash='dash', line_color='gray', opacity=0.6)
    fig4.add_vline(x=scatter_data['So_luong'].median(), line_dash='dot', line_color='blue', opacity=0.4)
    fig4.update_layout(height=450, plot_bgcolor='white', paper_bgcolor='white',
                       font=dict(family='Be Vietnam Pro'))
    st.plotly_chart(fig4, use_container_width=True)

    loss_prods = prod_pnl_real[prod_pnl_real['Loi_nhuan'] < 0]
    top_loss_name = loss_prods.nsmallest(1, 'Loi_nhuan')['Ten_hang_hoa'].values[0] if len(loss_prods) > 0 else "N/A"
    top_loss_val = loss_prods.nsmallest(1, 'Loi_nhuan')['Loi_nhuan'].values[0] if len(loss_prods) > 0 else 0
    st.markdown(f"""
    <div class="insight-box insight-negative">
    ⚠️ Có <strong>{len(loss_prods)}</strong> sản phẩm đang lỗ trong tháng. SP lỗ nặng nhất: 
    <strong>{top_loss_name}</strong> với mức lỗ <strong>{top_loss_val:,.0f}₫</strong>. 
    Cần review lại cấu trúc giá hoặc dừng bán để tránh lỗ thêm.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 4: Hoàn hàng & Rủi ro
# ══════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">↩️ Phân Tích Đơn Hoàn Hàng</div>', unsafe_allow_html=True)

    if len(hoan_data) > 0:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="kpi-card" style="border-left-color:#ef4444">
                <div class="kpi-label">Tổng SP Hoàn</div>
                <div class="kpi-value" style="color:#ef4444">{len(hoan_data)}</div>
                <div class="kpi-sub">Dòng sản phẩm hoàn trả</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="kpi-card" style="border-left-color:#f59e0b">
                <div class="kpi-label">Giá Vốn Bị Hoàn</div>
                <div class="kpi-value" style="color:#f59e0b">{total_hoan:,.0f}₫</div>
                <div class="kpi-sub">Tổng giá vốn hàng hoàn</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            hoan_pct = total_hoan / total_gia_von * 100 if total_gia_von > 0 else 0
            st.markdown(f"""
            <div class="kpi-card" style="border-left-color:#8b5cf6">
                <div class="kpi-label">Tỷ lệ Hoàn / Giá vốn</div>
                <div class="kpi-value" style="color:#8b5cf6">{hoan_pct:.1f}%</div>
                <div class="kpi-sub">Mức ảnh hưởng lên chi phí</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("#### 📋 Chi tiết đơn hoàn hàng")
        hoan_display = hoan_data[['San', 'Ten_hang', 'So_luong', 'Gia_von', 'Tong_gia_von']].copy()
        hoan_display = hoan_display[hoan_display['Tong_gia_von'] > 0]
        hoan_display['Tong_gia_von'] = hoan_display['Tong_gia_von'].apply(lambda x: f"{x:,.0f}₫")
        hoan_display['Gia_von'] = hoan_display['Gia_von'].apply(lambda x: f"{x:,.0f}₫")
        hoan_display.columns = ['Sàn', 'Tên hàng', 'Số lượng', 'Giá vốn/đơn', 'Tổng giá vốn']
        st.dataframe(hoan_display, use_container_width=True, hide_index=True)

        # Chart hàng hoàn theo sản phẩm
        hoan_prod = hoan_data[hoan_data['Tong_gia_von'] > 0].groupby('Ten_hang')['Tong_gia_von'].sum().reset_index()
        if len(hoan_prod) > 0:
            fig = px.bar(
                hoan_prod.sort_values('Tong_gia_von', ascending=False),
                x='Ten_hang', y='Tong_gia_von',
                title='↩️ Giá vốn bị hoàn theo sản phẩm',
                color='Tong_gia_von', color_continuous_scale='Reds',
                labels={'Ten_hang': 'Sản phẩm', 'Tong_gia_von': 'Giá vốn (VNĐ)'}
            )
            fig.update_layout(height=380, plot_bgcolor='white', paper_bgcolor='white',
                              coloraxis_showscale=False, font=dict(family='Be Vietnam Pro'),
                              xaxis_tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"""
        <div class="insight-box insight-negative">
        ↩️ Tỷ lệ hoàn hàng <strong>{hoan_pct:.1f}%</strong> so với tổng giá vốn.
        Giá vốn bị hoàn: <strong>{total_hoan:,.0f}₫</strong>. 
        Đây là tiền vốn bị "đóng băng" — cần theo dõi lý do hoàn hàng để giảm tỷ lệ này.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Không có dữ liệu hoàn hàng trong tháng này.")

    # Risk Matrix
    st.markdown('<div class="section-title">⚠️ Ma Trận Rủi Ro Sản Phẩm</div>', unsafe_allow_html=True)
    risk_data = prod_pnl_real.copy()
    risk_data['Muc_do_rui_ro'] = 'Thấp'
    risk_data.loc[(risk_data['Loi_nhuan'] < 0) & (risk_data['Tong_gia_von'] > 500000), 'Muc_do_rui_ro'] = 'Cao'
    risk_data.loc[(risk_data['Loi_nhuan'] < 0) & (risk_data['Tong_gia_von'] <= 500000), 'Muc_do_rui_ro'] = 'Trung bình'
    risk_data.loc[risk_data['Loi_nhuan'] > 0, 'Muc_do_rui_ro'] = 'An toàn'

    risk_count = risk_data['Muc_do_rui_ro'].value_counts().reset_index()
    risk_count.columns = ['Mức độ', 'Số lượng SP']
    color_risk = {'An toàn': '#22c55e', 'Thấp': '#84cc16', 'Trung bình': '#f59e0b', 'Cao': '#ef4444'}

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        fig_r = px.pie(risk_count, values='Số lượng SP', names='Mức độ',
                       title='📊 Phân bổ rủi ro sản phẩm',
                       color='Mức độ', color_discrete_map=color_risk, hole=0.4)
        fig_r.update_layout(height=350, font=dict(family='Be Vietnam Pro'))
        st.plotly_chart(fig_r, use_container_width=True)

    with col_r2:
        high_risk = risk_data[risk_data['Muc_do_rui_ro'] == 'Cao'].sort_values('Loi_nhuan').head(10)
        if len(high_risk) > 0:
            st.markdown("**🔴 Sản phẩm rủi ro cao cần xử lý ngay:**")
            for _, row in high_risk.iterrows():
                nm = str(row['Ten_hang_hoa'])[:45]
                st.markdown(f"""
                <div style="background:#fff5f5; border-left:3px solid #ef4444; padding:8px 12px; border-radius:6px; margin:4px 0; font-size:0.85rem">
                  ❌ <b>{nm}</b><br>
                  <span style="color:#888">Lỗ: <b style="color:#ef4444">{row['Loi_nhuan']:,.0f}₫</b> | Giá vốn: {row['Tong_gia_von']:,.0f}₫</span>
                </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 5: Chi tiết dữ liệu
# ══════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-title">🔍 Dữ Liệu Chi Tiết Tổng Hợp</div>', unsafe_allow_html=True)

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        search_term = st.text_input("🔎 Tìm kiếm sản phẩm", placeholder="Nhập tên SP...")
    with col_s2:
        san_detail = st.multiselect("Lọc theo sàn", ['TiktokShop', 'Shopee', 'Lazada'],
                                    default=['TiktokShop', 'Shopee', 'Lazada'], key='detail_san')

    detail_data = filtered_main[filtered_main['San'].isin(san_detail)].copy()
    if search_term:
        detail_data = detail_data[detail_data['Ten_hang_hoa'].astype(str).str.contains(search_term, case=False, na=False)]

    show_cols = ['San', 'Ten_hang_hoa', 'So_luong', 'Gia_von', 'Tong_gia_von', 'Tien_ve_TK_dk', 'Loi_nhuan', 'La_hang_tang']
    detail_display = detail_data[show_cols].copy()
    detail_display['Loi_nhuan_color'] = detail_display['Loi_nhuan'].apply(lambda x: '🟢' if x >= 0 else '🔴')
    detail_display['Tong_gia_von'] = detail_display['Tong_gia_von'].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else '')
    detail_display['Tien_ve_TK_dk'] = detail_display['Tien_ve_TK_dk'].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else '')
    detail_display['Loi_nhuan'] = detail_display['Loi_nhuan'].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else '')
    detail_display['La_hang_tang'] = detail_display['La_hang_tang'].map({True: '🎁 Tặng', False: '✅ Bán'})
    detail_display.columns = ['Sàn', 'Tên hàng', 'SL', 'Giá vốn/sp', 'Tổng giá vốn', 'Tiền về TK', 'Lợi nhuận', 'Loại', '±']

    st.dataframe(detail_display[['±', 'Sàn', 'Tên hàng', 'SL', 'Tổng giá vốn', 'Tiền về TK', 'Lợi nhuận', 'Loại']],
                 use_container_width=True, height=500, hide_index=True)

    st.caption(f"📊 Hiển thị {len(detail_display):,} dòng | "
               f"Tổng SL: {detail_data['So_luong'].sum():.0f} | "
               f"Tổng giá vốn: {detail_data['Tong_gia_von'].sum():,.0f}₫ | "
               f"Tổng lợi nhuận: {detail_data['Loi_nhuan'].sum():,.0f}₫")

    st.markdown('<div class="section-title">📊 Tổng Hợp theo Thương Hiệu</div>', unsafe_allow_html=True)
    brand_data = sl_data.copy()
    brand_data['Thuong_hieu_fill'] = brand_data['Thuong_hieu'].ffill()
    brand_stats = brand_data[~brand_data['Ten_hang'].astype(str).str.contains('Hàng tặng|TỔNG', na=False)].groupby('Thuong_hieu_fill').agg(
        SP_count=('Ten_hang', 'count'),
        Tong_SL=('Tong_SL', 'sum'),
        Tong_GV=('Gia_von', lambda x: (x * brand_data.loc[x.index, 'Tong_SL']).sum())
    ).reset_index().sort_values('Tong_SL', ascending=False).head(15)

    fig_brand = px.bar(
        brand_stats, x='Thuong_hieu_fill', y='Tong_SL',
        title='🏷️ Sản lượng bán theo thương hiệu',
        color='Tong_SL', color_continuous_scale='Viridis',
        labels={'Thuong_hieu_fill': 'Thương hiệu', 'Tong_SL': 'Tổng sản lượng'}
    )
    fig_brand.update_layout(height=380, plot_bgcolor='white', paper_bgcolor='white',
                            coloraxis_showscale=False, font=dict(family='Be Vietnam Pro'),
                            xaxis_tickangle=-30)
    fig_brand.update_traces(texttemplate='%{y}', textposition='outside')
    st.plotly_chart(fig_brand, use_container_width=True)

# ─── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#aaa; font-size:0.8rem; padding:10px">
    📊 Hệ thống phân tích TMĐT | Dữ liệu: Tháng 02/2026 | TiktokShop · Shopee · Lazada
</div>
""", unsafe_allow_html=True)
