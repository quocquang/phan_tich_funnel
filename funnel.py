import pandas as pd
import streamlit as st
import plotly.express as px
from io import BytesIO
import numpy as np

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
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                
        # Chuyển đổi cột 'Tỉ lệ thắng' từ dạng chuỗi sang số
        if 'Tỉ lệ thắng' in df.columns:
            df['Tỉ lệ thắng'] = df['Tỉ lệ thắng'].str.rstrip('%').astype('float') / 100
        
        return df
    except Exception as e:
        st.error(f"Lỗi khi đọc file: {str(e)}")
        return None

def show_filters(df):
    st.sidebar.header("Bộ lọc")
    
    filters = {}
    
    # Chỉ thêm filter cho các cột tồn tại trong DataFrame
    if "Tên khách hàng" in df.columns:
        filters["Tên khách hàng"] = st.sidebar.multiselect(
            "Tên khách hàng:",
            options=sorted(df["Tên khách hàng"].dropna().unique())
        )
    
    if "Giai đoạn" in df.columns:
        filters["Giai đoạn"] = st.sidebar.multiselect(
            "Giai đoạn:",
            options=sorted(df["Giai đoạn"].dropna().unique())
        )
    
    if "Tỉnh/TP" in df.columns:
        filters["Tỉnh/TP"] = st.sidebar.multiselect(
            "Tỉnh/TP:",
            options=sorted(df["Tỉnh/TP"].dropna().astype(str).unique())
        )
    
    return apply_filters(df, filters)

def apply_filters(df, filters):
    filtered_df = df.copy()
    
    for column, selected_values in filters.items():
        if selected_values:  # Chỉ áp dụng filter nếu có giá trị được chọn
            filtered_df = filtered_df[filtered_df[column].isin(selected_values)]
    
    return filtered_df

def show_basic_metrics(df):
    try:
        # Chuyển đổi 'Doanh thu dự kiến' sang kiểu số
        df['Doanh thu dự kiến'] = pd.to_numeric(df['Doanh thu dự kiến'], errors='coerce')
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            total_opportunities = len(df)
            st.metric("Tổng số cơ hội", f"{total_opportunities:,}")
        
        with col2:
            total_revenue = df['Doanh thu dự kiến'].sum()
            st.metric("Tổng doanh thu dự kiến", f"{total_revenue:,.0f} VND")
        
        with col3:
            avg_revenue = df['Doanh thu dự kiến'].mean()
            st.metric("Doanh thu trung bình", f"{avg_revenue:,.0f} VND")

        with col4:
            win_rate = df[df['Trạng thái'] == 'Active']['Tỉ lệ thắng'].mean()
            st.metric("Tỉ lệ thắng trung bình", f"{win_rate:.1f}%")
        
        with col5:
            active_opportunities = len(df[df['Trạng thái'] == 'Active'])
            st.metric("Số cơ hội đang hoạt động", f"{active_opportunities:,}")
        
        with col6:
            avg_conversion_time = (df['Ngày dự kiến kí HĐ'] - df['Thời điểm tạo']).mean()
            st.metric("Thời gian chuyển đổi trung bình", f"{avg_conversion_time.days} ngày")
            
    except Exception as e:
        st.error(f"Lỗi khi tính toán metrics cơ bản: {str(e)}")

def show_revenue_by_stage(df):
    try:
        if 'Giai đoạn' in df.columns and 'Doanh thu dự kiến' in df.columns:
            fig = px.box(df, 
                        x="Giai đoạn", 
                        y="Doanh thu dự kiến",
                        title="Phân bố doanh thu dự kiến theo giai đoạn")
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Lỗi khi tạo biểu đồ phân bố doanh thu: {str(e)}")

def show_detailed_data(df):
    try:
        st.subheader("Dữ liệu chi tiết")
        
        # Tạo một bản sao của df để định dạng dữ liệu
        formatted_df = df.copy()
        
        # Chuyển đổi 'Doanh thu dự kiến' và 'Tỉ lệ thắng' sang định dạng số
        if 'Doanh thu dự kiến' in formatted_df.columns:
            formatted_df['Doanh thu dự kiến'] = pd.to_numeric(formatted_df['Doanh thu dự kiến'], errors='coerce')
        if 'Tỉ lệ thắng' in formatted_df.columns:
            formatted_df['Tỉ lệ thắng'] = pd.to_numeric(formatted_df['Tỉ lệ thắng'], errors='coerce')
        
        # Định dạng dữ liệu khi hiển thị
        st.dataframe(
            formatted_df.style.format({
                "Doanh thu dự kiến": "{:,.0f}",
                "Tỉ lệ thắng": "{:.1f}%"
            })
        )
        
        # Download button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Tải xuống dữ liệu",
            csv,
            "data.csv",
            "text/csv",
            key='download-csv'
        )
    except Exception as e:
        st.error(f"Lỗi khi hiển thị dữ liệu chi tiết: {str(e)}")

def show_opportunities_by_customer(df):
    try:
        customer_opportunities = df['Tên khách hàng'].value_counts().reset_index()
        customer_opportunities.columns = ['Tên khách hàng', 'Số cơ hội']
        
        fig = px.bar(customer_opportunities, 
                     x='Tên khách hàng', 
                     y='Số cơ hội',
                     title="Số cơ hội theo khách hàng",
                     labels={'Tên khách hàng': 'Tên khách hàng', 'Số cơ hội': 'Số cơ hội'})
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Lỗi khi tạo biểu đồ số cơ hội theo khách hàng: {str(e)}")

def show_revenue_by_region(df):
    try:
        if 'Tỉnh/TP' in df.columns and 'Doanh thu dự kiến' in df.columns:
            revenue_by_region = df.groupby('Tỉnh/TP')['Doanh thu dự kiến'].sum().reset_index()
            
            fig = px.bar(revenue_by_region, 
                         x='Tỉnh/TP', 
                         y='Doanh thu dự kiến',
                         title="Doanh thu dự kiến theo vùng",
                         labels={'Tỉnh/TP': 'Tỉnh/TP', 'Doanh thu dự kiến': 'Doanh thu dự kiến (VND)'})
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Lỗi khi tạo biểu đồ doanh thu theo vùng: {str(e)}")

def show_revenue_by_product(df):
    try:
        if 'Ngành hàng' in df.columns and 'Doanh thu dự kiến' in df.columns:
            revenue_by_product = df.groupby('Ngành hàng')['Doanh thu dự kiến'].sum().reset_index()
            
            fig = px.pie(revenue_by_product, 
                         names='Ngành hàng', 
                         values='Doanh thu dự kiến',
                         title="Doanh thu dự kiến theo ngành hàng")
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Lỗi khi tạo biểu đồ doanh thu theo ngành hàng: {str(e)}")

def show_conversion_rate_by_stage(df):
    try:
        if 'Giai đoạn' in df.columns:
            conversion_rate_by_stage = df.groupby('Giai đoạn')['Tỉ lệ thắng'].mean().reset_index()
            
            fig = px.bar(conversion_rate_by_stage, 
                         x='Giai đoạn', 
                         y='Tỉ lệ thắng',
                         title="Tỉ lệ chuyển đổi theo giai đoạn",
                         labels={'Giai đoạn': 'Giai đoạn', 'Tỉ lệ thắng': 'Tỉ lệ thắng'})
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Lỗi khi tạo biểu đồ tỉ lệ chuyển đổi theo giai đoạn: {str(e)}")

def show_sales_by_salesperson(df):
    try:
        if 'Nhân viên kinh doanh' in df.columns and 'Doanh thu dự kiến' in df.columns:
            sales_by_salesperson = df.groupby('Nhân viên kinh doanh')['Doanh thu dự kiến'].sum().reset_index()
            
            fig = px.bar(sales_by_salesperson, 
                         x='Nhân viên kinh doanh', 
                         y='Doanh thu dự kiến',
                         title="Doanh thu theo nhân viên kinh doanh",
                         labels={'Nhân viên kinh doanh': 'Nhân viên kinh doanh', 'Doanh thu dự kiến': 'Doanh thu dự kiến (VND)'})
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Lỗi khi tạo biểu đồ doanh thu theo nhân viên kinh doanh: {str(e)}")

def show_opportunities_by_competitor(df):
    try:
        if 'Đối thủ' in df.columns:
            opportunities_by_competitor = df['Đối thủ'].value_counts().reset_index()
            opportunities_by_competitor.columns = ['Đối thủ', 'Số cơ hội']
            
            fig = px.bar(opportunities_by_competitor, 
                         x='Đối thủ', 
                         y='Số cơ hội',
                         title="Số cơ hội theo đối thủ",
                         labels={'Đối thủ': 'Đối thủ', 'Số cơ hội': 'Số cơ hội'})
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Lỗi khi tạo biểu đồ số cơ hội theo đối thủ: {str(e)}")

def main():
    st.title('🎯 Phân Tích Funnel')
    
    uploaded_file = st.sidebar.file_uploader(
        "📂 Tải lên file dữ liệu",
        type=['csv', 'xlsx', 'xls'],
        help="Hỗ trợ định dạng CSV và Excel"
    )
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        
        if df is not None:
            # Áp dụng bộ lọc
            filtered_df = show_filters(df)
            
            # Hiển thị các metrics cơ bản
            st.header("1. Tổng quan")
            show_basic_metrics(filtered_df)
            
            # Hiển thị biểu đồ phân bố doanh thu
            st.header("2. Phân tích doanh thu")
            show_revenue_by_stage(filtered_df)
            
            # Hiển thị biểu đồ số cơ hội theo khách hàng
            st.header("3. Số cơ hội theo khách hàng")
            show_opportunities_by_customer(filtered_df)
            
            # Hiển thị biểu đồ doanh thu theo vùng
            st.header("4. Doanh thu theo vùng")
            show_revenue_by_region(filtered_df)
            
            # Hiển thị biểu đồ doanh thu theo ngành hàng
            st.header("5. Doanh thu theo ngành hàng")
            show_revenue_by_product(filtered_df)
            
            # Hiển thị biểu đồ tỉ lệ chuyển đổi theo giai đoạn
            st.header("6. Tỉ lệ chuyển đổi theo giai đoạn")
            show_conversion_rate_by_stage(filtered_df)

            # Hiển thị biểu đồ doanh thu theo nhân viên kinh doanh
            st.header("7. Doanh thu theo nhân viên kinh doanh")
            show_sales_by_salesperson(filtered_df)
            
            # Hiển thị biểu đồ số cơ hội theo đối thủ
            st.header("8. Số cơ hội theo đối thủ")
            show_opportunities_by_competitor(filtered_df)
            
            # Hiển thị dữ liệu chi tiết
            st.header("9. Dữ liệu chi tiết")
            show_detailed_data(filtered_df)
            
        else:
            st.info("⚠️ Vui lòng kiểm tra lại file dữ liệu của bạn.")
    else:
        st.info("👆 Vui lòng tải lên file dữ liệu để bắt đầu phân tích.")

if __name__ == '__main__':
    main()
