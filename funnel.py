import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import numpy as np
from datetime import datetime, timedelta

# Cấu hình trang Streamlit
st.set_page_config(layout="wide", page_title="Phân Tích Funnel Bán Hàng", page_icon="🎯")

# CSS tùy chỉnh giao diện
st.markdown("""
    <style>
    .main > div {
        padding: 2rem;
        background-color: #f8f9fa;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: white;
        padding: 0.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        background-color: #f1f3f5;
        border-radius: 0.3rem;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e9ecef;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #4e73df;
        color: white;
    }
    .metric-card {
        background-color: white;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    }
    .stSlider > div > div > div > div {
        color: #4e73df;
    }
    .stButton > button {
        background-color: #4e73df;
        color: white;
        border-radius: 0.3rem;
    }
    </style>
""", unsafe_allow_html=True)

# Tiêu đề và logo
col1, col2 = st.columns([1, 5])
with col1:
    st.image("https://github.com/user-attachments/assets/f263bd14-23a4-4735-b082-1d10ade1bbb0", width=80)  # Thay bằng logo công ty
with col2:
    st.title("🎯 Phân Tích Funnel Bán Hàng")

# Hàm tải dữ liệu
@st.cache_data
def load_data(file):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
        else:
            st.error("Vui lòng tải lên file CSV hoặc Excel.")
            return None

        # Chuẩn hóa dữ liệu
        df.columns = df.columns.str.strip()
        date_columns = ["Ngày dự kiến kí HĐ", "Thời điểm tạo"]
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        if 'Tỉ lệ thắng' in df.columns:
            df['Tỉ lệ thắng'] = df['Tỉ lệ thắng'].str.rstrip('%').astype('float') / 100
        if 'Doanh thu dự kiến' in df.columns:
            df['Doanh thu dự kiến'] = pd.to_numeric(df['Doanh thu dự kiến'], errors='coerce')
        return df
    except Exception as e:
        st.error(f"Lỗi khi đọc file: {str(e)}")
        return None

# Hàm hiển thị bộ lọc
def show_filters(df):
    with st.sidebar.expander("🔍 Bộ Lọc", expanded=True):
        filters = {}
        
        if "Thời điểm tạo" in df.columns:
            min_date = df["Thời điểm tạo"].min().date()
            max_date = df["Thời điểm tạo"].max().date()
            date_range = st.date_input("Phạm vi thời gian:", value=(min_date, max_date))
            if len(date_range) == 2:
                filters["Thời điểm tạo"] = date_range

        if "Nhân viên kinh doanh" in df.columns:
            filters["Nhân viên kinh doanh"] = st.multiselect(
                "Nhân viên kinh doanh:", options=sorted(df["Nhân viên kinh doanh"].dropna().unique())
            )
        
        if "Tỉnh/TP" in df.columns:
            filters["Tỉnh/TP"] = st.multiselect(
                "Tỉnh/TP:", options=sorted(df["Tỉnh/TP"].dropna().unique())
            )

        if "Doanh thu dự kiến" in df.columns:
            min_revenue = int(df["Doanh thu dự kiến"].min())
            max_revenue = int(df["Doanh thu dự kiến"].max())
            use_slider = st.checkbox("Sử dụng Slider cho Khoảng doanh thu", value=True)
            
            if use_slider:
                revenue_range = st.slider(
                    "Khoảng doanh thu (VND):", min_revenue, max_revenue, (min_revenue, max_revenue),
                    step=1000000, format="%d VND"
                )
                filters["Doanh thu dự kiến"] = revenue_range
            else:
                col1, col2 = st.columns(2)
                with col1:
                    min_val = st.number_input("Doanh thu tối thiểu (VND):", min_value=min_revenue, max_value=max_revenue, value=min_revenue, step=1000000)
                with col2:
                    max_val = st.number_input("Doanh thu tối đa (VND):", min_value=min_revenue, max_value=max_revenue, value=max_revenue, step=1000000)
                filters["Doanh thu dự kiến"] = (min_val, max_val)

    return apply_filters(df, filters)

# Áp dụng bộ lọc
def apply_filters(df, filters):
    filtered_df = df.copy()
    for column, values in filters.items():
        if column == "Thời điểm tạo" and len(values) == 2:
            start_date, end_date = values
            filtered_df = filtered_df[
                (filtered_df[column].dt.date >= start_date) & 
                (filtered_df[column].dt.date <= end_date)
            ]
        elif column == "Doanh thu dự kiến" and len(values) == 2:
            min_val, max_val = values
            filtered_df = filtered_df[
                (filtered_df[column] >= min_val) & 
                (filtered_df[column] <= max_val)
            ]
        elif values:
            filtered_df = filtered_df[filtered_df[column].isin(values)]
    return filtered_df

# Dashboard tổng quan
def show_dashboard(df):
    st.subheader("📊 Dashboard Tổng Quan")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_opps = len(df)
        st.metric("Tổng cơ hội", f"{total_opps:,}", help="Số lượng cơ hội trong dữ liệu")
    
    with col2:
        total_revenue = df["Doanh thu dự kiến"].sum()
        st.metric("Doanh thu dự kiến", f"{total_revenue:,.0f} VND", help="Tổng doanh thu dự kiến")
    
    with col3:
        if "Tỉ lệ thắng" in df.columns:
            avg_win_rate = df["Tỉ lệ thắng"].mean() * 100
            st.metric("Tỉ lệ thắng TB", f"{avg_win_rate:.1f}%", help="Tỉ lệ thắng trung bình")

    # Biểu đồ Funnel mini
    if "Giai đoạn" in df.columns:
        stage_counts = df["Giai đoạn"].value_counts()
        fig = px.funnel(stage_counts, x=stage_counts.values, y=stage_counts.index)
        fig.update_layout(height=300, title="Funnel Mini")
        st.plotly_chart(fig, use_container_width=True)

# Phân tích Funnel
def show_funnel_analysis(df):
    if "Giai đoạn" not in df.columns:
        st.warning("Dữ liệu không chứa cột 'Giai đoạn'.")
        return
    
    st.subheader("🎯 Phân tích Funnel")
    stage_data = df.groupby("Giai đoạn").agg({
        "Tên cơ hội": "count",
        "Doanh thu dự kiến": "sum",
        "Tỉ lệ thắng": "mean"
    }).rename(columns={"Tên cơ hội": "Số cơ hội"}).reset_index()

    # Định dạng dữ liệu
    stage_data["Doanh thu dự kiến"] = stage_data["Doanh thu dự kiến"].apply(lambda x: f"{x:,.0f} VND")
    stage_data["Tỉ lệ thắng"] = stage_data["Tỉ lệ thắng"].apply(lambda x: f"{x:.2%}")

    # Biểu đồ Funnel
    fig = go.Figure(go.Funnel(
        y=stage_data["Giai đoạn"],
        x=stage_data["Số cơ hội"],
        textinfo="value+percent initial",
        marker={"color": "#4e73df"},
        customdata=stage_data[["Doanh thu dự kiến", "Tỉ lệ thắng"]],
        hovertemplate="Giai đoạn: %{y}<br>Số cơ hội: %{x}<br>Doanh thu dự kiến: %{customdata[0]}<br>Tỉ lệ thắng: %{customdata[1]}"
    ))
    fig.update_layout(title="Phân tích Funnel theo Giai đoạn")
    st.plotly_chart(fig, use_container_width=True)

    # Chi tiết các cơ hội ở mỗi giai đoạn
    st.subheader("Chi tiết các cơ hội ở mỗi giai đoạn")
    for stage in stage_data["Giai đoạn"]:
        stage_opps = df[df["Giai đoạn"] == stage]
        with st.expander(f"{stage} - {stage_data[stage_data['Giai đoạn'] == stage]['Số cơ hội'].values[0]} cơ hội"):
            st.dataframe(stage_opps[["Tên cơ hội", "Doanh thu dự kiến", "Tỉ lệ thắng", "Nhân viên kinh doanh"]])

# Phân tích theo nhân viên kinh doanh
def show_salesperson_analysis(df):
    if "Nhân viên kinh doanh" not in df.columns:
        st.warning("Dữ liệu không chứa cột 'Nhân viên kinh doanh'.")
        return
    
    st.subheader("📈 Phân tích theo Nhân viên Kinh doanh")
    salesperson_data = df.groupby("Nhân viên kinh doanh").agg({
        "Doanh thu dự kiến": "sum",
        "Tỉ lệ thắng": "mean",
        "Giai đoạn": "count"
    }).rename(columns={"Giai đoạn": "Số cơ hội"}).reset_index()

    # Biểu đồ doanh thu
    fig1 = px.bar(
        salesperson_data, x="Nhân viên kinh doanh", y="Doanh thu dự kiến",
        title="Doanh thu dự kiến theo nhân viên", text_auto=".2s", color_discrete_sequence=["#4e73df"]
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Biểu đồ số cơ hội
    fig2 = px.bar(
        salesperson_data, x="Nhân viên kinh doanh", y="Số cơ hội",
        title="Số cơ hội theo nhân viên", color_discrete_sequence=["#36b9cc"]
    )
    st.plotly_chart(fig2, use_container_width=True)

# Phân tích theo khu vực địa lý
def show_area_analysis(df):
    if "Tỉnh/TP" not in df.columns:
        st.warning("Dữ liệu không chứa cột 'Tỉnh/TP'.")
        return
    
    st.subheader("🌍 Phân tích theo Khu vực")
    area_data = df.groupby("Tỉnh/TP").agg({
        "Doanh thu dự kiến": "sum",
        "Tỉ lệ thắng": "mean",
        "Giai đoạn": "count"
    }).rename(columns={"Giai đoạn": "Số cơ hội"}).reset_index()

    # Biểu đồ doanh thu
    fig1 = px.bar(
        area_data, x="Tỉnh/TP", y="Doanh thu dự kiến",
        title="Doanh thu dự kiến theo khu vực", text_auto=".2s", color_discrete_sequence=["#1cc88a"]
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Biểu đồ số cơ hội
    fig2 = px.bar(
        area_data, x="Tỉnh/TP", y="Số cơ hội",
        title="Số cơ hội theo khu vực", color_discrete_sequence=["#f6c23e"]
    )
    st.plotly_chart(fig2, use_container_width=True)

# Phân tích theo ngành hàng
def show_industry_analysis(df):
    if "Ngành hàng" not in df.columns:
        st.warning("Dữ liệu không chứa cột 'Ngành hàng'.")
        return
    
    st.subheader("🏭 Phân tích theo Ngành hàng")
    industry_data = df.groupby("Ngành hàng").agg({
        "Doanh thu dự kiến": "sum",
        "Giai đoạn": "count"
    }).rename(columns={"Giai đoạn": "Số cơ hội"}).reset_index()

    # Biểu đồ tròn doanh thu
    fig1 = px.pie(industry_data, values="Doanh thu dự kiến", names="Ngành hàng", title="Doanh thu dự kiến theo ngành hàng")
    st.plotly_chart(fig1, use_container_width=True)

    # Biểu đồ cột số cơ hội
    fig2 = px.bar(industry_data, x="Ngành hàng", y="Số cơ hội", title="Số cơ hội theo ngành hàng", color_discrete_sequence=["#f6c23e"])
    st.plotly_chart(fig2, use_container_width=True)

# Phân tích chu kỳ bán hàng
def show_sales_cycle_analysis(df):
    if "Thời điểm tạo" not in df.columns or "Ngày dự kiến kí HĐ" not in df.columns:
        st.warning("Dữ liệu không chứa cột 'Thời điểm tạo' hoặc 'Ngày dự kiến kí HĐ'.")
        return
    
    st.subheader("⏳ Phân tích Chu kỳ Bán hàng")
    df["Thời gian chuyển đổi (ngày)"] = (df["Ngày dự kiến kí HĐ"] - df["Thời điểm tạo"]).dt.days
    cycle_data = df.groupby("Giai đoạn")["Thời gian chuyển đổi (ngày)"].mean().reset_index()

    # Biểu đồ cột
    fig = px.bar(cycle_data, x="Giai đoạn", y="Thời gian chuyển đổi (ngày)", title="Thời gian trung bình giữa các giai đoạn (ngày)", color_discrete_sequence=["#1cc88a"])
    st.plotly_chart(fig, use_container_width=True)

# Các hàm phân tích bổ sung
def show_revenue_by_stage(df):
    try:
        if 'Giai đoạn' in df.columns and 'Doanh thu dự kiến' in df.columns:
            fig = px.box(df, x="Giai đoạn", y="Doanh thu dự kiến", title="Phân bố doanh thu dự kiến theo giai đoạn")
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Lỗi khi tạo biểu đồ phân bố doanh thu: {str(e)}")

def show_opportunities_by_customer(df):
    try:
        customer_opportunities = df['Tên khách hàng'].value_counts().reset_index()
        customer_opportunities.columns = ['Tên khách hàng', 'Số cơ hội']
        
        fig = px.bar(customer_opportunities, x='Tên khách hàng', y='Số cơ hội', title="Số cơ hội theo khách hàng")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Lỗi khi tạo biểu đồ số cơ hội theo khách hàng: {str(e)}")

def show_revenue_by_region(df):
    try:
        if 'Tỉnh/TP' in df.columns and 'Doanh thu dự kiến' in df.columns:
            revenue_by_region = df.groupby('Tỉnh/TP')['Doanh thu dự kiến'].sum().reset_index()
            
            fig = px.bar(revenue_by_region, x='Tỉnh/TP', y='Doanh thu dự kiến', title="Doanh thu dự kiến theo vùng")
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Lỗi khi tạo biểu đồ doanh thu theo vùng: {str(e)}")

def show_revenue_by_product(df):
    try:
        if 'Ngành hàng' in df.columns and 'Doanh thu dự kiến' in df.columns:
            revenue_by_product = df.groupby('Ngành hàng')['Doanh thu dự kiến'].sum().reset_index()
            
            fig = px.pie(revenue_by_product, names='Ngành hàng', values='Doanh thu dự kiến', title="Doanh thu dự kiến theo ngành hàng")
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Lỗi khi tạo biểu đồ doanh thu theo ngành hàng: {str(e)}")

def show_conversion_rate_by_stage(df):
    try:
        if 'Giai đoạn' in df.columns:
            conversion_rate_by_stage = df.groupby('Giai đoạn')['Tỉ lệ thắng'].mean().reset_index()
            
            fig = px.bar(conversion_rate_by_stage, x='Giai đoạn', y='Tỉ lệ thắng', title="Tỉ lệ chuyển đổi theo giai đoạn")
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Lỗi khi tạo biểu đồ tỉ lệ chuyển đổi theo giai đoạn: {str(e)}")

def show_opportunities_by_industry(df):
    try:
        if 'Ngành hàng' in df.columns:
            opportunities_by_industry = df['Ngành hàng'].value_counts().reset_index()
            opportunities_by_industry.columns = ['Ngành hàng', 'Số cơ hội']
            
            fig = px.bar(opportunities_by_industry, x='Ngành hàng', y='Số cơ hội', title="Số cơ hội theo ngành hàng")
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Lỗi khi tạo biểu đồ số cơ hội theo ngành hàng: {str(e)}")

# Hàm hiển thị dữ liệu chi tiết và xuất dữ liệu
def show_detailed_data(df):
    st.subheader("📋 Dữ liệu chi tiết")
    st.dataframe(df.style.format({"Doanh thu dự kiến": "{:,.0f}", "Tỉ lệ thắng": "{:.1f}%"}))
    
    # Export to CSV
    csv = df.to_csv(index=False)
    st.download_button(
        label="Export to CSV",
        data=csv,
        file_name="funnel_data.csv",
        mime="text/csv",
        key="download-csv"
    )

    # Export to Excel
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    writer.close()  # Đóng writer thay vì sử dụng save()
    excel_data = output.getvalue()
    st.download_button(
        label="Export to Excel",
        data=excel_data,
        file_name="funnel_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="download-excel"
    )

# Main
file = st.sidebar.file_uploader("Tải file dữ liệu (CSV/Excel)", type=["csv", "xlsx"])
if file:
    df = load_data(file)
    if df is not None:
        filtered_df = show_filters(df)

        # Tabs
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12 = st.tabs([
            "Tổng quan", "Funnel", "Nhân viên", "Khu vực", "Ngành hàng", "Chu kỳ bán hàng",
            "Doanh thu theo giai đoạn", "Cơ hội theo khách hàng", "Doanh thu theo vùng",
            "Doanh thu theo ngành", "Tỉ lệ chuyển đổi", "Dữ liệu chi tiết"
        ])
        
        with tab1:
            show_dashboard(filtered_df)
        
        with tab2:
            show_funnel_analysis(filtered_df)
        
        with tab3:
            show_salesperson_analysis(filtered_df)
        
        with tab4:
            show_area_analysis(filtered_df)
        
        with tab5:
            show_industry_analysis(filtered_df)
        
        with tab6:
            show_sales_cycle_analysis(filtered_df)
        
        with tab7:
            show_revenue_by_stage(filtered_df)
        
        with tab8:
            show_opportunities_by_customer(filtered_df)
        
        with tab9:
            show_revenue_by_region(filtered_df)
        
        with tab10:
            show_revenue_by_product(filtered_df)
        
        with tab11:
            show_conversion_rate_by_stage(filtered_df)
        
        with tab12:
            show_detailed_data(filtered_df)
