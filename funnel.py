import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import openpyxl
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# Cấu hình trang Streamlit
st.set_page_config(layout="wide", page_title="Phân Tích Funnel")

# CSS tùy chỉnh giao diện
st.markdown("""
    <style>
    .main > div {
        padding: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------- Hàm xử lý dữ liệu ------------------- #
@st.cache_data
def load_data(file):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
        else:
            st.error("Định dạng file không được hỗ trợ. Vui lòng sử dụng CSV hoặc Excel.")
            return None

        # Loại bỏ khoảng trắng thừa ở tên cột
        df.columns = df.columns.str.strip()

        # Chuyển đổi các cột ngày tháng
        date_columns = ["Ngày dự kiến kí HĐ", "Thời điểm tạo"]
        for col in date_columns:
            df[col] = pd.to_datetime(df[col], format="%d/%m/%Y", errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"Lỗi khi đọc file: {str(e)}")
        return None

# ------------------- Hàm lọc dữ liệu ------------------- #

def apply_filters(df, filters):
    filtered_df = df.copy()
    
    if filters.get("Tên khách hàng"):
        filtered_df = filtered_df[filtered_df["Tên khách hàng"].isin(filters["Tên khách hàng"])]
        
    if filters.get("Giai đoạn"):
        filtered_df = filtered_df[filtered_df["Giai đoạn"].isin(filters["Giai đoạn"])]
        
    if filters.get("Tỉ lệ thắng"):
        filtered_df = filtered_df[filtered_df["Tỉ lệ thắng"].isin(filters["Tỉ lệ thắng"])]
        
    if filters.get("Tỉnh/TP"):
        filtered_df = filtered_df[filtered_df["Tỉnh/TP"].isin(filters["Tỉnh/TP"])]
    
    if filters.get("Nhóm khách hàng theo chính sách công nợ"):
        filtered_df = filtered_df[
            filtered_df["Nhóm khách hàng theo chính sách công nợ"].isin(
                filters["Nhóm khách hàng theo chính sách công nợ"]
            )
        ]
        
    if filters.get("Ngành hàng"):
        filtered_df = filtered_df[filtered_df["Ngành hàng"].isin(filters["Ngành hàng"])]
        
    return filtered_df

def show_filters(df):
    st.sidebar.header("Bộ lọc")
    
    # Xử lý giá trị NaN nếu cần cho các cột hiển thị
    if "Tên khách hàng" in df.columns:
        df["Tên khách hàng"].fillna("Unknown", inplace=True)
    if "Ngành hàng" in df.columns:
        df["Ngành hàng"].fillna("Unknown", inplace=True)
    
    filters = {
        "Tên khách hàng": st.sidebar.multiselect("Tên khách hàng:", options=sorted(df["Tên khách hàng"].unique())),
        "Giai đoạn": st.sidebar.multiselect("Giai đoạn:", options=sorted(df["Giai đoạn"].unique())),
        "Tỉ lệ thắng": st.sidebar.multiselect("Tỉ lệ thắng:", options=sorted(df["Tỉ lệ thắng"].unique())),
        "Tỉnh/TP": st.sidebar.multiselect("Tỉnh/TP:", options=sorted(df["Tỉnh/TP"].astype(str).unique())),
        "Nhóm khách hàng theo chính sách công nợ": st.sidebar.multiselect("Nhóm khách hàng theo chính sách công nợ:", options=sorted(df["Nhóm khách hàng theo chính sách công nợ"].unique())),
        "Ngành hàng": st.sidebar.multiselect("Ngành hàng:", options=sorted(df["Ngành hàng"].unique())),
    }
    
    return apply_filters(df, filters)

# ------------------- Hàm hiển thị trang bìa ------------------- #

def show_cover_page():
    st.title("Phân Tích Funnel")
    st.write("""
        **Mục tiêu:** Ứng dụng này giúp phân tích Funnel, theo dõi hiệu suất và xác định các khoản Funnel.
        **Tính năng:**
        - Lọc dữ liệu theo thời gian, khu vực, trạng thái, và người quản lý.
        - Trình bày trực quan các chỉ số chính và biểu đồ phân tích chi tiết.
        - Xuất dữ liệu đã lọc dưới dạng CSV/Excel.
    """)
    st.image("https://github.com/user-attachments/assets/f263bd14-23a4-4735-b082-1d10ade1bbb0", use_column_width=True)

# ------------------- Hàm main ------------------- #

# Tính các chỉ số phân tích mô tả
def calculate_descriptive_metrics(df):
    metrics = {}
    
    # 1. Tổng số cơ hội
    metrics['total_opportunities'] = len(df)
    
    # 2. Phân bố theo giai đoạn
    stage_dist = df['Giai đoạn'].value_counts(normalize=True) * 100
    metrics['stage_distribution'] = stage_dist
    
    # 3. Tổng doanh thu dự kiến
    metrics['total_expected_revenue'] = df['Doanh thu dự kiến'].sum()
    
    # 4. Doanh thu dự kiến trung bình
    metrics['avg_expected_revenue'] = df['Doanh thu dự kiến'].mean()
    
    # 5. Doanh thu dự kiến trung vị
    metrics['median_expected_revenue'] = df['Doanh thu dự kiến'].median()
    
    # 6. Độ lệch chuẩn của doanh thu dự kiến
    metrics['std_expected_revenue'] = df['Doanh thu dự kiến'].std()
    
    # 7. Tỉ lệ thắng trung bình
    metrics['avg_win_rate'] = df['Tỉ lệ thắng'].mean()
    
    # 8. Thời gian trung bình từ tạo đến ký HĐ
    df['time_to_sign'] = (df['Ngày dự kiến kí HĐ'] - df['Thời điểm tạo']).dt.days
    metrics['avg_time_to_sign'] = df['time_to_sign'].mean()
    
    return metrics

# Tính các chỉ số phân tích dự đoán
def calculate_predictive_metrics(df):
    # Chuẩn bị dữ liệu cho mô hình
    le = LabelEncoder()
    df_model = df.copy()
    
    # Mã hóa các biến categorical
    categorical_cols = ['Giai đoạn', 'Trạng thái', 'Ngành hàng', 'Nhân viên kinh doanh']
    for col in categorical_cols:
        df_model[col] = le.fit_transform(df_model[col])
    
    # Feature importance
    X = df_model[categorical_cols + ['Doanh thu dự kiến', 'Tỉ lệ thắng']]
    y = df_model['time_to_sign']
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X, y)
    
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return feature_importance

# Tính các chỉ số phân tích chuẩn đoán
def calculate_diagnostic_metrics(df):
    # Ma trận tương quan
    numeric_cols = ['Doanh thu dự kiến', 'Tỉ lệ thắng', 'time_to_sign']
    corr_matrix = df[numeric_cols].corr()
    
    # Phân tích hiệu suất nhân viên
    sales_performance = df.groupby('Nhân viên kinh doanh').agg({
        'Doanh thu dự kiến': ['count', 'mean', 'sum'],
        'Tỉ lệ thắng': 'mean'
    }).round(2)
    
    return corr_matrix, sales_performance

# Hàm main chính
def main():
    st.title('🎯 Hệ Thống Phân Tích Funnel Nâng Cao')
    
    # Hiển thị trang bìa
    show_cover_page()
    
    # Tải file dữ liệu từ sidebar
    uploaded_file = st.sidebar.file_uploader("📂 Tải lên file dữ liệu", 
                                               type=['csv', 'xlsx', 'xls'],
                                               help="Hỗ trợ định dạng CSV và Excel")
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        
        if df is not None:
            # Áp dụng bộ lọc dữ liệu
            filtered_df = show_filters(df)
            
            # Đảm bảo các cột tài chính là số
            filtered_df['Doanh thu dự kiến'] = pd.to_numeric(filtered_df['Doanh thu dự kiến'], errors='coerce').fillna(0)
            
            # Tính toán các chỉ số
            desc_metrics = calculate_descriptive_metrics(filtered_df)
            feature_importance = calculate_predictive_metrics(filtered_df)
            corr_matrix, sales_performance = calculate_diagnostic_metrics(filtered_df)

            # Hiển thị dashboard
            st.header("1. Chỉ số Phân tích Mô tả")

            # Metrics trong 3 cột
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Tổng số cơ hội", f"{desc_metrics['total_opportunities']:,}")
            with col2:
                st.metric("Tổng doanh thu dự kiến", f"{desc_metrics['total_expected_revenue']:,.0f} VND")
            with col3:
                st.metric("Tỉ lệ thắng trung bình", f"{desc_metrics['avg_win_rate']:.1f}%")

            # Thêm các metrics khác
            col4, col5, col6 = st.columns(3)
            with col4:
                st.metric("Doanh thu trung bình", f"{desc_metrics['avg_expected_revenue']:,.0f} VND")
            with col5:
                st.metric("Doanh thu trung vị", f"{desc_metrics['median_expected_revenue']:,.0f} VND")
            with col6:
                st.metric("Thời gian trung bình đến ký HĐ", f"{desc_metrics['avg_time_to_sign']:.1f} ngày")

            # Biểu đồ phân tích
            st.subheader("Phân bố doanh thu theo giai đoạn")
            fig1 = px.box(filtered_df, x="Giai đoạn", y="Doanh thu dự kiến",
                          title="Phân bố doanh thu dự kiến theo giai đoạn")
            st.plotly_chart(fig1, use_container_width=True)

            # Phân tích dự đoán
            st.header("2. Chỉ số Phân tích Dự đoán")

            # Feature importance
            st.subheader("Tầm quan trọng của các yếu tố")
            fig2 = px.bar(feature_importance, x='importance', y='feature', orientation='h',
                          title="Feature Importance trong dự đoán thời gian ký hợp đồng")
            st.plotly_chart(fig2, use_container_width=True)

            # Phân tích chuẩn đoán
            st.header("3. Chỉ số Phân tích Chuẩn đoán")

            # Ma trận tương quan
            st.subheader("Ma trận tương quan")
            fig3 = px.imshow(corr_matrix, 
                             labels=dict(color="Correlation"),
                             title="Ma trận tương quan giữa các biến số")
            st.plotly_chart(fig3, use_container_width=True)

            # Hiệu suất nhân viên
            st.subheader("Phân tích hiệu suất nhân viên kinh doanh")
            st.dataframe(sales_performance)

            # Thêm tính năng phân tích theo thời gian
            st.header("4. Phân tích theo thời gian")

            # Timeline của các cơ hội
            fig4 = px.timeline(
                filtered_df,
                x_start="Thời điểm tạo",
                x_end="Ngày dự kiến kí HĐ",
                y="Tên cơ hội",
                title="Timeline các cơ hội"
            )
            st.plotly_chart(fig4, use_container_width=True)

            # Dữ liệu chi tiết
            st.header("5. Dữ liệu chi tiết")
            st.dataframe(
                filtered_df.style.format({
                    "Doanh thu dự kiến": "{:,.0f}",
                    "Tỉ lệ thắng": "{:.1f}%"
                })
            )

            # Download button
            csv = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Tải dữ liệu",
                csv,
                "data.csv",
                "text/csv",
                key='download-csv'
            )
        else:
            st.info("Vui lòng tải lên file dữ liệu để bắt đầu phân tích.")

if __name__ == '__main__':
    main()
