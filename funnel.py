import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import openpyxl
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# Cấu hình trang Streamlit (đảm bảo lệnh này nằm ở đầu file)
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
        "Tên khách hàng": st.sidebar.multiselect(
            "Tên khách hàng:", 
            options=sorted(df["Tên khách hàng"].unique())
        ),
        "Giai đoạn": st.sidebar.multiselect(
            "Giai đoạn:", 
            options=sorted(df["Giai đoạn"].unique())
        ),
        "Tỉ lệ thắng": st.sidebar.multiselect(
            "Tỉ lệ thắng:", 
            options=sorted(df["Tỉ lệ thắng"].unique())
        ),
        "Tỉnh/TP": st.sidebar.multiselect(
            "Tỉnh/TP:", 
            options=sorted(df["Tỉnh/TP"].astype(str).unique())
        ),
        "Nhóm khách hàng theo chính sách công nợ": st.sidebar.multiselect(
            "Nhóm khách hàng theo chính sách công nợ:", 
            options=sorted(df["Nhóm khách hàng theo chính sách công nợ"].unique())
        ),
        "Ngành hàng": st.sidebar.multiselect(
            "Ngành hàng:", 
            options=sorted(df["Ngành hàng"].unique())
        ),
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

# ------------------- Hàm tính toán các chỉ số ------------------- #

# I. Chỉ số phân tích mô tả
def calculate_descriptive_metrics(df):
    metrics = {}
    
    # 1. Tổng số cơ hội
    metrics['total_opportunities'] = len(df)
    
    # 2. Phân bố theo giai đoạn (tính phần trăm)
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
    
    # 7. Tỉ lệ thắng trung bình (chuyển đổi cột nếu cần)
    df['Tỉ lệ thắng'] = pd.to_numeric(df['Tỉ lệ thắng'], errors='coerce')
    metrics['avg_win_rate'] = df['Tỉ lệ thắng'].mean()
    
    # 8. Thời gian trung bình từ tạo đến ký HĐ
    # Nếu chưa tồn tại cột "time_to_sign", tính toán từ 2 cột ngày
    if "time_to_sign" not in df.columns:
        df['time_to_sign'] = (df['Ngày dự kiến kí HĐ'] - df['Thời điểm tạo']).dt.days
    metrics['avg_time_to_sign'] = df['time_to_sign'].mean()
    
    return metrics

# II. Chỉ số phân tích dự đoán
def calculate_predictive_metrics(df):
    # Tạo một bản sao để không làm thay đổi dữ liệu gốc
    df_model = df.copy()
    
    # Nếu chưa có cột time_to_sign, tính toán nó
    if "time_to_sign" not in df_model.columns:
        df_model["time_to_sign"] = (df_model["Ngày dự kiến kí HĐ"] - df_model["Thời điểm tạo"]).dt.days

    # Danh sách các cột categorical cần mã hóa. Kiểm tra sự tồn tại của từng cột.
    categorical_cols = []
    for col in ['Giai đoạn', 'Trạng thái', 'Ngành hàng', 'Nhân viên kinh doanh']:
        if col in df_model.columns:
            categorical_cols.append(col)
    
    le = LabelEncoder()
    for col in categorical_cols:
        # Ép dữ liệu về chuỗi để đảm bảo không lỗi
        df_model[col] = le.fit_transform(df_model[col].astype(str))
    
    # Chọn các tính năng (features)
    # Nếu "Tỉ lệ thắng" chưa ở dạng numeric, hãy đảm bảo nó đã được chuyển đổi
    X_cols = categorical_cols + ['Doanh thu dự kiến', 'Tỉ lệ thắng']
    X = df_model[X_cols].fillna(0)
    y = df_model["time_to_sign"].fillna(0)
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X, y)
    
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return feature_importance

# III. Chỉ số phân tích chuẩn đoán
def calculate_diagnostic_metrics(df):
    # Ma trận tương quan giữa các cột số
    numeric_cols = ['Doanh thu dự kiến', 'Tỉ lệ thắng', 'time_to_sign']
    corr_matrix = df[numeric_cols].corr()
    
    # Phân tích hiệu suất nhân viên kinh doanh (nếu cột này tồn tại)
    if "Nhân viên kinh doanh" in df.columns:
        sales_performance = df.groupby("Nhân viên kinh doanh").agg({
            'Doanh thu dự kiến': ['count', 'mean', 'sum'],
            'Tỉ lệ thắng': 'mean'
        }).round(2)
    else:
        sales_performance = None
    
    return corr_matrix, sales_performance

# ------------------- Hàm main ------------------- #
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
            
            # Đảm bảo cột "Doanh thu dự kiến" là số
            filtered_df['Doanh thu dự kiến'] = pd.to_numeric(filtered_df['Doanh thu dự kiến'], errors='coerce').fillna(0)
            
            # Tính toán các chỉ số
            desc_metrics = calculate_descriptive_metrics(filtered_df)
            feature_importance = calculate_predictive_metrics(filtered_df)
            corr_matrix, sales_performance = calculate_diagnostic_metrics(filtered_df)
            
            # Hiển thị Dashboard
            st.header("1. Chỉ số Phân tích Mô tả")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Tổng số cơ hội", f"{desc_metrics['total_opportunities']:,}")
            with col2:
                st.metric("Tổng doanh thu dự kiến", f"{desc_metrics['total_expected_revenue']:,.0f} VND")
            with col3:
                st.metric("Tỉ lệ thắng trung bình", f"{desc_metrics['avg_win_rate']:.1f}%")
            
            col4, col5, col6 = st.columns(3)
            with col4:
                st.metric("Doanh thu trung bình", f"{desc_metrics['avg_expected_revenue']:,.0f} VND")
            with col5:
                st.metric("Doanh thu trung vị", f"{desc_metrics['median_expected_revenue']:,.0f} VND")
            with col6:
                st.metric("Thời gian trung bình đến ký HĐ", f"{desc_metrics['avg_time_to_sign']:.1f} ngày")
            
            st.subheader("Phân bố doanh thu theo giai đoạn")
            fig1 = px.box(filtered_df, x="Giai đoạn", y="Doanh thu dự kiến",
                          title="Phân bố doanh thu dự kiến theo giai đoạn")
            st.plotly_chart(fig1, use_container_width=True)
            
            st.header("2. Chỉ số Phân tích Dự đoán")
            st.subheader("Tầm quan trọng của các yếu tố")
            fig2 = px.bar(feature_importance, x='importance', y='feature', orientation='h',
                          title="Feature Importance trong dự đoán thời gian ký HĐ")
            st.plotly_chart(fig2, use_container_width=True)
            
            st.header("3. Chỉ số Phân tích Chuẩn đoán")
            st.subheader("Ma trận tương quan")
            fig3 = px.imshow(corr_matrix, 
                             labels=dict(color="Correlation"),
                             title="Ma trận tương quan giữa các biến số")
            st.plotly_chart(fig3, use_container_width=True)
            
            st.subheader("Phân tích hiệu suất nhân viên kinh doanh")
            if sales_performance is not None:
                st.dataframe(sales_performance)
            else:
                st.info("Không có dữ liệu cho phân tích nhân viên kinh doanh.")
            
            st.header("4. Phân tích theo thời gian")
            fig4 = px.timeline(
                filtered_df,
                x_start="Thời điểm tạo",
                x_end="Ngày dự kiến kí HĐ",
                y="Tên cơ hội",
                title="Timeline các cơ hội"
            )
            st.plotly_chart(fig4, use_container_width=True)
            
            st.header("5. Dữ liệu chi tiết")
            st.dataframe(
                filtered_df.style.format({
                    "Doanh thu dự kiến": "{:,.0f}",
                    "Tỉ lệ thắng": "{:.1f}%"
                })
            )
            
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
