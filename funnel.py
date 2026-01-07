import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ==========================================
# 1. CẤU HÌNH & CSS
# ==========================================
st.set_page_config(
    page_title="Ultimate Spa Dashboard",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS tùy chỉnh để giao diện đẹp hơn
st.markdown("""
<style>
    [data-testid="stMetricValue"] {
        font-size: 24px;
        font-weight: bold;
        color: #2E86C1;
    }
    .main-header {
        font-size: 30px; 
        font-weight: bold; 
        color: #4CAF50;
        text-align: center;
        margin-bottom: 20px;
    }
    div.block-container {padding-top: 1rem;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. XỬ LÝ DỮ LIỆU
# ==========================================
@st.cache_data
def load_data(file):
    try:
        df = pd.read_excel(file)
        
        # Chuẩn hóa tên cột
        df.columns = df.columns.str.lower().str.strip()
        rename_map = {
            'ma_kh': 'ma_kh', 'mã kh': 'ma_kh',
            'ten_khach_hang': 'ten_khach',
            'ngay_tao_dat_hen': 'trans_date',
            'ma_bill': 'ma_bill', 'mã bill': 'ma_bill',
            'dich_vu': 'dich_vu', 'dv': 'dich_vu', 'sp': 'sp',
            'so_luong': 'so_luong', 'số lượng': 'so_luong', 'số lượng sp': 'so_luong',
            'doanh thu': 'doanh_thu',
            'loai_khach': 'segment', 'loại khách': 'segment'
        }
        df.rename(columns=rename_map, inplace=True)
        
        # Xử lý dữ liệu
        df['item_name'] = df['dich_vu'].fillna(df['sp']).fillna("Không xác định")
        df['trans_date'] = pd.to_datetime(df['trans_date'], errors='coerce')
        df['doanh_thu'] = df['doanh_thu'].fillna(0)
        
        # Thêm các cột thời gian phụ
        df['Day'] = df['trans_date'].dt.date
        df['Weekday'] = df['trans_date'].dt.day_name()
        
        return df
    except Exception as e:
        st.error(f"Lỗi file: {e}")
        return None

# ==========================================
# 3. SIDEBAR & BỘ LỌC (FILTER)
# ==========================================
st.sidebar.header("🔍 Bộ Lọc Dữ Liệu")
uploaded_file = st.sidebar.file_uploader("Upload File Excel", type=["xlsx"])

if uploaded_file:
    df_original = load_data(uploaded_file)
    
    if df_original is not None:
        # --- CẤU HÌNH BỘ LỌC ---
        
        # 1. Lọc Thời Gian
        min_date = df_original['trans_date'].min().date()
        max_date = df_original['trans_date'].max().date()
        
        date_range = st.sidebar.date_input(
            "Chọn khoảng thời gian",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        # 2. Lọc Loại Khách
        all_segments = df_original['segment'].unique()
        selected_segments = st.sidebar.multiselect(
            "Chọn Loại Khách",
            options=all_segments,
            default=all_segments
        )
        
        # 3. Lọc Dịch Vụ (Top 20 dịch vụ phổ biến để đỡ rối)
        top_services = df_original['item_name'].value_counts().head(20).index.tolist()
        selected_services = st.sidebar.multiselect(
            "Lọc theo Dịch vụ (Top 20)",
            options=top_services,
            default=None,
            help="Để trống để chọn tất cả"
        )
        
        # --- ÁP DỤNG BỘ LỌC ---
        if len(date_range) == 2:
            start_date, end_date = date_range
            mask = (
                (df_original['trans_date'].dt.date >= start_date) & 
                (df_original['trans_date'].dt.date <= end_date) &
                (df_original['segment'].isin(selected_segments))
            )
            df = df_original.loc[mask]
            
            # Lọc dịch vụ nếu có chọn
            if selected_services:
                df = df[df['item_name'].isin(selected_services)]
        else:
            df = df_original

        # ==========================================
        # 4. DASHBOARD CHÍNH
        # ==========================================
        st.markdown("<div class='main-header'>✨ BÁO CÁO HIỆU QUẢ KINH DOANH SPA ✨</div>", unsafe_allow_html=True)
        
        # --- HÀNG 1: METRIC CARDS (CHỈ SỐ TỔNG QUAN) ---
        total_rev = df['doanh_thu'].sum()
        total_orders = df['ma_bill'].nunique()
        total_cust = df['ma_kh'].nunique()
        avg_order = total_rev / total_orders if total_orders else 0
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("💰 Doanh Thu", f"{total_rev:,.0f}")
        c2.metric("🧾 Đơn Hàng", f"{total_orders:,}")
        c3.metric("👥 Khách Hàng", f"{total_cust:,}")
        c4.metric("🏷️ TB Giá Trị Đơn", f"{avg_order:,.0f}")
        
        st.markdown("---")

        # --- TABS GIAO DIỆN ---
        tab1, tab2, tab3 = st.tabs(["📊 Tổng Quan & Xu Hướng", "🧩 Phân Tích Dịch Vụ (TreeMap)", "📅 Lịch & Chi Tiết"])
        
        # === TAB 1: TỔNG QUAN & XU HƯỚNG ===
        with tab1:
            col_left, col_right = st.columns([2, 1])
            
            with col_left:
                # Biểu đồ miền (Area Chart) - Nhìn xu hướng rõ hơn Line
                daily_trend = df.groupby(['Day', 'segment'])['doanh_thu'].sum().reset_index()
                fig_trend = px.area(daily_trend, x='Day', y='doanh_thu', color='segment',
                                    title='🌊 Diễn biến Doanh thu theo ngày (Chồng lấn)',
                                    labels={'doanh_thu': 'Doanh Thu', 'Day': 'Ngày'},
                                    color_discrete_map={'Khách Cũ': '#2E86C1', 'Khách Mới': '#FF5733'})
                st.plotly_chart(fig_trend, use_container_width=True)
                
            with col_right:
                # Biểu đồ Sunburst (Mặt trời) - Rất đẹp để xem cấu trúc
                # Tầng 1: Loại khách -> Tầng 2: Dịch vụ (Top 10)
                # Gom nhóm dữ liệu nhỏ lại thành "Other" để biểu đồ không bị rối
                df_sunburst = df.copy()
                top_10_sv = df_sunburst['item_name'].value_counts().head(10).index
                df_sunburst.loc[~df_sunburst['item_name'].isin(top_10_sv), 'item_name'] = 'Dịch vụ khác'
                
                sunburst_data = df_sunburst.groupby(['segment', 'item_name'])['doanh_thu'].sum().reset_index()
                
                fig_sun = px.sunburst(sunburst_data, path=['segment', 'item_name'], values='doanh_thu',
                                      title='🌞 Cấu trúc Doanh thu (Sunburst)',
                                      color='segment',
                                      color_discrete_map={'Khách Cũ': '#2E86C1', 'Khách Mới': '#FF5733'})
                st.plotly_chart(fig_sun, use_container_width=True)

        # === TAB 2: PHÂN TÍCH DỊCH VỤ (TREEMAP) ===
        with tab2:
            st.subheader("🌳 Bản đồ nhiệt Dịch vụ (TreeMap)")
            st.caption("Ô càng to -> Doanh thu càng lớn. Màu càng đậm -> Giá trị đơn càng cao.")
            
            # Tính toán dữ liệu cho TreeMap
            tree_data = df.groupby('item_name').agg(
                Tong_Doanh_Thu=('doanh_thu', 'sum'),
                So_Luong=('so_luong', 'sum'),
                Gia_TB=('doanh_thu', 'mean')
            ).reset_index()
            
            # Chỉ lấy Top 30 dịch vụ để hiển thị cho đẹp
            tree_data = tree_data.sort_values('Tong_Doanh_Thu', ascending=False).head(30)
            
            fig_tree = px.treemap(tree_data, path=['item_name'], values='Tong_Doanh_Thu',
                                  color='Tong_Doanh_Thu', # Màu sắc theo doanh thu
                                  hover_data=['So_Luong', 'Gia_TB'],
                                  color_continuous_scale='Greens',
                                  title='Top 30 Dịch vụ đóng góp Doanh thu cao nhất')
            fig_tree.update_traces(textinfo="label+value+percent entry")
            st.plotly_chart(fig_tree, use_container_width=True)
            
            # Biểu đồ so sánh Top Dịch vụ theo Loại Khách
            st.divider()
            st.subheader("⚔️ So sánh Top Dịch vụ: Khách Cũ vs Mới")
            
            c_old, c_new = st.columns(2)
            
            with c_old:
                top_old = df[df['segment']=='Khách Cũ'].groupby('item_name')['doanh_thu'].sum().nlargest(10).reset_index()
                fig_bar_old = px.bar(top_old, x='doanh_thu', y='item_name', orientation='h',
                                     title='Top Dịch vụ Khách Cũ', color_discrete_sequence=['#2E86C1'])
                fig_bar_old.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_bar_old, use_container_width=True)
                
            with c_new:
                top_new = df[df['segment']=='Khách Mới'].groupby('item_name')['doanh_thu'].sum().nlargest(10).reset_index()
                fig_bar_new = px.bar(top_new, x='doanh_thu', y='item_name', orientation='h',
                                     title='Top Dịch vụ Khách Mới (Hero Product)', color_discrete_sequence=['#FF5733'])
                fig_bar_new.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_bar_new, use_container_width=True)

        # === TAB 3: LỊCH & CHI TIẾT ===
        with tab3:
            col_cal, col_data = st.columns([1, 1])
            
            with col_cal:
                st.subheader("📅 Lịch nhiệt (Ngày nào đông khách?)")
                # Chuẩn bị dữ liệu Heatmap
                cal_data = df.groupby('Day')['ma_bill'].nunique().reset_index()
                cal_data.columns = ['Date', 'Orders']
                
                fig_cal = px.bar(cal_data, x='Date', y='Orders',
                                 title='Số lượng đơn hàng theo ngày',
                                 color='Orders', color_continuous_scale='Oranges')
                st.plotly_chart(fig_cal, use_container_width=True)
                
            with col_data:
                st.subheader("📄 Dữ liệu chi tiết")
                # Hiển thị bảng dữ liệu có thể sort/filter
                st.dataframe(
                    df[['trans_date', 'ma_bill', 'ten_khach', 'item_name', 'doanh_thu', 'segment']]
                    .sort_values('doanh_thu', ascending=False)
                    .style.format({'doanh_thu': '{:,.0f}', 'trans_date': '{:%d-%m-%Y}'}),
                    height=400
                )
    
    else:
        st.error("File Excel không đúng định dạng. Vui lòng kiểm tra lại tên cột.")

else:
    # Màn hình Welcome
    st.info("👈 Hãy tải file Excel 'Bao_Cao_Thang_12_Final.xlsx' lên thanh bên trái để xem Dashboard.")
    st.markdown("""
    ### Dashboard này giúp bạn thấy gì?
    1. **TreeMap (Sơ đồ cây):** Nhìn ngay được mảng dịch vụ nào đang "hái ra tiền".
    2. **Sunburst (Biểu đồ tròn phân cấp):** Thấy rõ cơ cấu khách hàng.
    3. **Bộ lọc linh hoạt:** Muốn xem riêng "Triệt nách" hay xem riêng "Tuần đầu tháng 12" đều được.
    """)
