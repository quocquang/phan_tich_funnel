import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# 1. CẤU HÌNH TRANG & STYLE
# ==========================================
st.set_page_config(
    page_title="Deep Dive Spa Dashboard - T12/2025",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS cho đẹp hơn
st.markdown("""
<style>
    .metric-card {background-color: #f0f2f6; border-radius: 10px; padding: 15px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);}
    .stTabs [data-baseweb="tab-list"] {gap: 10px;}
    .stTabs [data-baseweb="tab"] {height: 50px; white-space: pre-wrap; background-color: #ffffff; border-radius: 5px; box-shadow: 0px 2px 5px rgba(0,0,0,0.05);}
    .stTabs [aria-selected="true"] {background-color: #4CAF50; color: white;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. HÀM XỬ LÝ DỮ LIỆU THÔNG MINH
# ==========================================
@st.cache_data
def load_and_clean_data(file):
    try:
        df = pd.read_excel(file)
        
        # 1. Chuẩn hóa tên cột (fix mọi lỗi viết hoa thường/dấu cách)
        df.columns = df.columns.str.lower().str.strip()
        
        # Mapping tên cột từ file ảnh của bạn sang tên chuẩn code
        rename_map = {
            'ma_kh': 'ma_kh', 'mã kh': 'ma_kh',
            'ten_khach_hang': 'ten_khach',
            'ngay_mua_lan_dau': 'first_date',
            'ngay_tao_dat_hen': 'trans_date',
            'ma_bill': 'ma_bill', 'mã bill': 'ma_bill',
            'dich_vu': 'dich_vu', 'dv': 'dich_vu',
            'sp': 'sp',
            'so_luong': 'so_luong', 'số lượng': 'so_luong',
            'gia_ban': 'don_gia',
            'doanh thu': 'doanh_thu', 'doanh_th': 'doanh_thu',
            'loai_khach': 'segment', 'loại khách': 'segment'
        }
        df.rename(columns=rename_map, inplace=True)
        
        # 2. Xử lý dữ liệu
        # Tạo cột Tên Hàng Hóa chung (Gộp SP và DV)
        df['item_name'] = df['dich_vu'].fillna(df['sp']).fillna("Không xác định")
        
        # Xử lý thời gian
        df['trans_date'] = pd.to_datetime(df['trans_date'], errors='coerce')
        
        # Trích xuất thêm thông tin thời gian để phân tích sâu
        df['Day'] = df['trans_date'].dt.date
        df['Hour'] = df['trans_date'].dt.hour
        df['Weekday'] = df['trans_date'].dt.day_name()
        
        # Xử lý tiền tệ
        df['doanh_thu'] = df['doanh_thu'].fillna(0)
        # Nếu doanh thu = 0 mà có đơn giá, tự tính lại
        mask_zero = df['doanh_thu'] == 0
        df.loc[mask_zero, 'doanh_thu'] = df.loc[mask_zero, 'don_gia'] * df.loc[mask_zero, 'so_luong']
        
        return df
    except Exception as e:
        st.error(f"Lỗi xử lý file: {e}")
        return None

# ==========================================
# 3. GIAO DIỆN CHÍNH
# ==========================================

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2928/2928892.png", width=100)
    st.title("Admin Control")
    uploaded_file = st.file_uploader("📂 Upload File Excel Báo Cáo", type=["xlsx"])
    
    st.divider()
    st.info("💡 Mẹo: File cần có cột 'Loai_Khach' đã được chuẩn hóa từ bước trước.")

# --- MAIN CONTENT ---
if uploaded_file:
    df = load_and_clean_data(uploaded_file)
    
    if df is not None:
        # --- TITLE & KPI ---
        st.title("📊 Dashboard Phân Tích Chuyên Sâu T12/2025")
        st.markdown(f"*Dữ liệu cập nhật đến: {df['trans_date'].max().strftime('%d/%m/%Y')}*")
        
        # TÍNH TOÁN KPI
        total_rev = df['doanh_thu'].sum()
        total_orders = df['ma_bill'].nunique()
        total_customers = df['ma_kh'].nunique()
        aov = total_rev / total_orders if total_orders else 0
        
        # Hiển thị KPI Cards
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("💰 Tổng Doanh Thu", f"{total_rev:,.0f} đ", delta_color="normal")
        kpi2.metric("🧾 Tổng Đơn Hàng", f"{total_orders}", delta="Transactions")
        kpi3.metric("👥 Tổng Khách", f"{total_customers}", delta="Unique Users")
        kpi4.metric("🏷️ Giá Trị TB/Đơn (AOV)", f"{aov:,.0f} đ", help="Trung bình 1 bill khách trả bao nhiêu")
        
        st.divider()

        # --- TABS PHÂN TÍCH ---
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 1. Sức Khỏe Tài Chính", 
            "👥 2. Chân Dung Khách Hàng", 
            "🛍️ 3. Sản Phẩm & Combo",
            "⏰ 4. Xu Hướng & Vận Hành"
        ])

        # =================================================
        # TAB 1: SỨC KHỎE TÀI CHÍNH (FINANCIAL HEALTH)
        # =================================================
        with tab1:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # 1. Biểu đồ đường: Doanh thu theo ngày (Tách màu Old/New)
                daily_trend = df.groupby(['Day', 'segment'])['doanh_thu'].sum().reset_index()
                fig_trend = px.line(daily_trend, x='Day', y='doanh_thu', color='segment',
                                    title='🔥 Diễn Biến Doanh Thu Theo Ngày (Real-time Trend)',
                                    labels={'doanh_thu': 'Doanh Thu', 'Day': 'Ngày', 'segment': 'Loại Khách'},
                                    color_discrete_map={'Khách Cũ': '#2E86C1', 'Khách Mới': '#E74C3C'},
                                    markers=True)
                fig_trend.update_layout(hovermode="x unified")
                st.plotly_chart(fig_trend, use_container_width=True)
            
            with col2:
                # 2. Donut Chart: Tỷ trọng đóng góp
                rev_share = df.groupby('segment')['doanh_thu'].sum().reset_index()
                fig_pie = px.pie(rev_share, values='doanh_thu', names='segment', hole=0.4,
                                 title='💰 Ai đang nuôi sống Spa?',
                                 color='segment',
                                 color_discrete_map={'Khách Cũ': '#2E86C1', 'Khách Mới': '#E74C3C'})
                st.plotly_chart(fig_pie, use_container_width=True)

            # 3. Phân tích AOV (Average Order Value)
            st.subheader("💸 Phân tích độ 'Chịu Chi' (AOV Analysis)")
            bill_values = df.groupby(['ma_bill', 'segment'])['doanh_thu'].sum().reset_index()
            
            fig_box = px.box(bill_values, x='segment', y='doanh_thu', color='segment',
                             title='So sánh giá trị đơn hàng trung bình (Box Plot)',
                             points="all", # Hiện tất cả các điểm để thấy khách VIP
                             color_discrete_map={'Khách Cũ': '#2E86C1', 'Khách Mới': '#E74C3C'},
                             labels={'doanh_thu': 'Giá trị đơn hàng'})
            st.plotly_chart(fig_box, use_container_width=True)
            st.caption("ℹ️ *Biểu đồ Box Plot giúp bạn thấy khách Mới hay Cũ chịu chi hơn. Các điểm chấm phía trên cao là những đơn hàng 'khủng' (Outliers).*")

        # =================================================
        # TAB 2: CHÂN DUNG KHÁCH HÀNG (CUSTOMER INSIGHTS)
        # =================================================
        with tab2:
            col_cust1, col_cust2 = st.columns(2)
            
            with col_cust1:
                # 1. Tần suất mua hàng (Frequency)
                st.subheader("🔄 Độ Trung Thành (Tần Suất Quay Lại)")
                freq = df.groupby(['ma_kh', 'segment'])['ma_bill'].nunique().reset_index()
                # Phân nhóm tần suất
                freq['Frequency_Group'] = pd.cut(freq['ma_bill'], 
                                               bins=[0, 1, 2, 5, 100], 
                                               labels=['1 lần (Vãng lai)', '2 lần (Tiềm năng)', '3-5 lần (Thân thiết)', '>5 lần (VIP Ruột)'])
                
                fig_freq = px.histogram(freq, x='Frequency_Group', color='segment', barmode='group',
                                      title='Phân bổ tần suất mua hàng trong tháng',
                                      color_discrete_map={'Khách Cũ': '#2E86C1', 'Khách Mới': '#E74C3C'})
                st.plotly_chart(fig_freq, use_container_width=True)

            with col_cust2:
                # 2. Top Khách Hàng (Pareto)
                st.subheader("🏆 Bảng Vàng Khách VIP (Top Contributors)")
                top_customers = df.groupby(['ten_khach', 'segment'])['doanh_thu'].sum().reset_index().sort_values('doanh_thu', ascending=False).head(10)
                
                fig_bar_cust = px.bar(top_customers, x='doanh_thu', y='ten_khach', orientation='h', color='segment',
                                    title='Top 10 Khách chi tiêu cao nhất',
                                    color_discrete_map={'Khách Cũ': '#2E86C1', 'Khách Mới': '#E74C3C'},
                                    text_auto='.2s')
                fig_bar_cust.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_bar_cust, use_container_width=True)

        # =================================================
        # TAB 3: SẢN PHẨM & COMBO (PRODUCT PERFORMANCE)
        # =================================================
        with tab3:
            st.subheader("🛍️ Ma Trận Sản Phẩm (BCG Matrix Simualtion)")
            
            # Tính metrics cho từng sản phẩm
            prod_perf = df.groupby('item_name').agg(
                Revenue=('doanh_thu', 'sum'),
                Quantity=('so_luong', 'sum'),
                Unique_Buyers=('ma_kh', 'nunique')
            ).reset_index()
            
            # Scatter Plot: Doanh thu vs Số người mua
            fig_scatter = px.scatter(prod_perf, x='Unique_Buyers', y='Revenue', 
                                   size='Quantity', color='Revenue',
                                   hover_name='item_name',
                                   title='Phân loại sản phẩm: Bò Sữa (Doanh thu cao) vs Mồi Câu (Nhiều người mua)',
                                   labels={'Unique_Buyers': 'Số người mua (Độ phổ biến)', 'Revenue': 'Tổng Doanh Thu'},
                                   color_continuous_scale='Viridis')
            # Thêm đường trung bình để chia 4 góc phần tư
            fig_scatter.add_vline(x=prod_perf['Unique_Buyers'].mean(), line_dash="dash", line_color="green")
            fig_scatter.add_hline(y=prod_perf['Revenue'].mean(), line_dash="dash", line_color="green")
            
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            st.markdown("""
            * **Góc trên phải (Cao tiền, Đông khách):** Sản phẩm Ngôi Sao (Cần duy trì).
            * **Góc dưới phải (Thấp tiền, Đông khách):** Sản phẩm Mồi Câu (Dùng để hút khách mới).
            * **Góc trên trái (Cao tiền, Ít khách):** Sản phẩm Cao Cấp (Dành cho VIP).
            """)
            
            st.divider()
            
            # Hero Product cho Khách Mới
            st.subheader("👶 Hero Product: Khách Mới thường mua gì đầu tiên?")
            new_cust_prod = df[df['segment']=='Khách Mới'].groupby('item_name')['ma_kh'].nunique().sort_values(ascending=False).head(10).reset_index()
            fig_hero = px.bar(new_cust_prod, x='item_name', y='ma_kh',
                            title='Top Dịch vụ thu hút Khách Mới nhất',
                            color_discrete_sequence=['#E74C3C'])
            st.plotly_chart(fig_hero, use_container_width=True)

        # =================================================
        # TAB 4: VẬN HÀNH & THỜI GIAN (OPERATIONAL)
        # =================================================
        with tab4:
            col_time1, col_time2 = st.columns(2)
            
            with col_time1:
                # Heatmap: Giờ cao điểm
                st.subheader("🔥 Khung Giờ Vàng (Peak Hours)")
                hourly_sales = df.groupby('Hour')['ma_bill'].nunique().reset_index()
                fig_area = px.area(hourly_sales, x='Hour', y='ma_bill',
                                 title='Mật độ đơn hàng theo khung giờ',
                                 labels={'Hour': 'Giờ trong ngày', 'ma_bill': 'Số lượng đơn'})
                st.plotly_chart(fig_area, use_container_width=True)
            
            with col_time2:
                # Doanh thu theo thứ trong tuần
                st.subheader("📅 Hiệu quả các ngày trong tuần")
                # Sắp xếp thứ tự
                days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                weekday_sales = df.groupby('Weekday')['doanh_thu'].sum().reindex(days_order).reset_index()
                
                fig_bar_week = px.bar(weekday_sales, x='Weekday', y='doanh_thu',
                                    title='Doanh thu theo thứ (Weekday Analysis)',
                                    color='doanh_thu', color_continuous_scale='Blues')
                st.plotly_chart(fig_bar_week, use_container_width=True)

else:
    # Màn hình chờ
    st.info("👋 Xin chào! Vui lòng upload file Excel 'Bao_Cao_Thang_12_Final.xlsx' ở cột bên trái để bắt đầu phân tích.")
    st.write("---")
    st.image("https://media.giphy.com/media/L1R1TVTh2RhtR5Jgww/giphy.gif", width=300)
