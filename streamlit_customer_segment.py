# import libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn import metrics
import joblib
from sklearn.cluster import KMeans
import plotly.express as px
from sklearn.metrics import silhouette_score
import seaborn as sns



st.title("Data Science")
st.write("## Customer Segment Project")



menu = ["Overview", "Product Insights", "Build Project", "New Prediction"]
choice = st.sidebar.selectbox('Menu', menu)
# Thông tin nhóm thực hiện trong sidebar
st.sidebar.markdown(" \n")  
st.sidebar.markdown(" \n") 
st.sidebar.markdown(" \n") 
st.sidebar.markdown("---")  
st.sidebar.markdown("### 👥 Project Members:")
st.sidebar.markdown("- Trần Hiểu Băng  \n- Mai Hồng Hà")

st.sidebar.markdown("👩‍🏫 **Instructor:**  \nCô Khuất Thùy Phương")

st.sidebar.markdown("📅 **Date of Submission:**  \n20/04/2025")


df = pd.read_csv("df_no_outliers_with_no.csv")
# Load scaler và model
def load_model():
    scaler = joblib.load("scaler.pkl")
    model = joblib.load("kmeans_model.pkl")
    return scaler, model
scaler, model = load_model()
if choice == 'Overview':    
    st.subheader("Overview")
    st.write("""
    #### Company Overview:
    Store X in the U.S. primarily sells essential products to customers, including vegetables, fruits, meat, fish, eggs, dairy, and beverages. The store's customers are retail buyers.
    The owner of Store X wants to increase sales, introduce products to the right target customers, and provide excellent customer service to enhance customer satisfaction.
    """)  
    st.image("X_image.png", caption="Store X Overview", use_container_width=True)
    st.write("""
    #### Problem to Address:
    Develop a customer segmentation system based on the information provided by the store.
    """)  

elif choice == 'Product Insights':
    st.title("📦 Phân tích Sản phẩm & Giao dịch")
    
    try:
        # Đọc dữ liệu
        import pandas as pd
        products = pd.read_csv("Products_with_Categories.csv")
        transactions = pd.read_csv("Transactions.csv")

        # Gộp 2 bảng
        dff = transactions.merge(products, on="productId", how="left")
        dff["Revenue"] = dff["items"] * dff["price"]
        # Kiểm tra và đọc cột Date
        if "Date" in dff.columns:
            dff["Date"] = pd.to_datetime(dff["Date"], format="%d-%m-%Y", errors="coerce")  # Chuyển sang datetime
            dff["Month"] = dff["Date"].dt.to_period("M").astype(str)    # Lấy tháng
            dff["Weekday"] = dff["Date"].dt.day_name()                  # Lấy thứ trong tuần
        else:
            print("Cột 'Date' không tồn tại trong dataframe.")



        # Chọn phần phân tích
        analysis_type = st.radio("🔎 Chọn loại phân tích:", [
            "Tổng quan dữ liệu", "Top sản phẩm bán chạy","Top sản phẩm bán kém",
            "Doanh thu theo danh mục", "Phân bố giá sản phẩm",  "Số lượng bán theo tháng", "Số lượng bán theo thứ trong tuần"
        ])

        # 1. Tổng quan dữ liệu
        if analysis_type == "Tổng quan dữ liệu":
            st.subheader("📋 Dataset (Top 10)")
            st.dataframe(dff.head(10))
            st.write(f"🔢 Tổng số giao dịch: {len(dff)}")
            st.write(f"📦 Số lượng sản phẩm khác nhau: {dff['productName'].nunique()}")
            st.write(f"📂 Số danh mục: {dff['Category'].nunique()}")

        # 2. Top sản phẩm bán chạy
        elif analysis_type == "Top sản phẩm bán chạy":
            top_products = dff.groupby("productName")["items"].sum().sort_values(ascending=False).head(10)
            fig, ax = plt.subplots()
            top_products.plot(kind="barh", color="skyblue", ax=ax)
            ax.set_title("Top 10 sản phẩm bán chạy")
            ax.set_xlabel("Số lượng bán")
            ax.invert_yaxis()
            st.pyplot(fig)
        
        # 3. Top sản phẩm bán chạy
        elif analysis_type == "Top sản phẩm bán kém":
            bottom = dff.groupby("productName")["items"].sum().sort_values().head(10)
            fig, ax = plt.subplots()
            bottom.plot(kind="barh", color="skyblue", ax=ax)
            ax.set_title("Top 10 sản phẩm bán kém nhất")
            ax.set_xlabel("Số lượng bán")
            ax.invert_yaxis()
            st.pyplot(fig)

        # . Doanh thu theo danh mục
        elif analysis_type == "Doanh thu theo danh mục":
            rev_by_cat = dff.groupby("Category")["Revenue"].sum().sort_values(ascending=False)
            fig, ax = plt.subplots()
            rev_by_cat.plot(kind="barh", color="coral", ax=ax)
            ax.set_title("Tổng doanh thu theo danh mục")
            ax.set_xlabel("Doanh thu")
            ax.set_ylabel("Danh mục")
            #plt.xticks(rotation=45)
            st.pyplot(fig)

        # 4. Phân bố giá sản phẩm
        elif analysis_type == "Phân bố giá sản phẩm":
            fig, ax = plt.subplots()
            sns.histplot(products["price"], bins=30, kde=True, color="green", ax=ax)
            ax.set_title("Phân bố giá sản phẩm")
            st.pyplot(fig)

        elif analysis_type == "Số lượng bán theo tháng":
            monthly_sales = dff.groupby("Month")["items"].sum().sort_index()
            fig, ax = plt.subplots()
            monthly_sales.plot(kind="bar", color="blue", ax=ax)
            ax.set_title("Tổng số lượng bán theo tháng")
            ax.set_ylabel("Số lượng")
            ax.set_xlabel("Tháng")
            #plt.xticks(rotation=45)
            st.pyplot(fig)


        elif analysis_type == "Số lượng bán theo thứ trong tuần":
            weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            weekday_sales = dff.groupby("Weekday")["items"].sum().reindex(weekday_order)
            fig, ax = plt.subplots()
            weekday_sales.plot(kind="bar", color="orange", ax=ax)
            ax.set_title("Tổng số lượng bán theo thứ trong tuần")
            ax.set_ylabel("Số lượng")
            plt.xticks(rotation=30)
            st.pyplot(fig)



    except Exception as e:
        st.error(f"Lỗi khi xử lý dữ liệu: {e}")

elif choice == 'Build Project':
    st.subheader("Build Project")
    st.write("#### Data Preprocessing")
    
    
# Chuẩn hóa và dự đoán
    X_scaled = scaler.transform(df[["Recency", "Frequency", "Monetary"]])
    df["Cluster"] = model.predict(X_scaled)
    st.write("##### Show data:")
    st.dataframe(df[["Recency", "Frequency", "Monetary"]].head())
    st.write("#### Elbow Method:")
    distortions = []
    K = range(1, 11)
    for k in K:
        kmeanModel = KMeans(n_clusters=k, random_state=42)
        kmeanModel.fit(X_scaled)
        distortions.append(kmeanModel.inertia_)

    fig1, ax1 = plt.subplots()
    ax1.plot(K, distortions, 'bo-')
    ax1.set_xlabel("Số cụm (k)")
    ax1.set_ylabel("Distortion (Inertia)")
    ax1.set_title("Phương pháp Elbow")
    st.pyplot(fig1)
    
    # --- Phần Silhouette Analysis ---
    st.write("#### Silhouette Analysis:")

    silhouette_scores = []
    K2 = range(2, 11)
    for k in K2:
        kmeanModel = KMeans(n_clusters=k, random_state=42)
        kmeanModel.fit(X_scaled)
        score = silhouette_score(X_scaled, kmeanModel.labels_)
        silhouette_scores.append(score)

    fig2, ax2 = plt.subplots()
    ax2.plot(K2, silhouette_scores, 'bo-')
    ax2.set_xlabel("Số cụm (k)")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Analysis")
    st.pyplot(fig2)

    st.write("#### Treemap (with k = 5):")
    summary = df.groupby("Cluster").agg({
    "Recency": "mean",
    "Frequency": "mean",
    "Monetary": "mean"
    }).round(0)

    summary["Customer Count"] = df["Cluster"].value_counts()
    summary["Percentage"] = (summary["Customer Count"] / summary["Customer Count"].sum() * 100).round(2)

    summary = summary.reset_index()

    # Tạo label đẹp để hiển thị trong từng ô Treemap
    summary["label"] = summary.apply(
        lambda row: f"Cluster {int(row['Cluster'])}<br>"
                f"{int(row['Recency'])} days<br>"
                f"{int(row['Frequency'])} orders<br>"
                f"{int(row['Monetary'])} $<br>"
                f"{int(row['Customer Count'])} customers ({row['Percentage']}%)",
        axis=1
        )

    # Vẽ treemap

    fig_treemap = px.treemap(
    summary,
    path=["label"],
    values="Customer Count",
    color="Monetary",
    color_continuous_scale="RdBu"
)
    st.plotly_chart(fig_treemap, use_container_width=True)
    # Vẽ Scatter plot
    scatter_data = summary[["Cluster", "Recency", "Frequency", "Monetary"]].rename(columns={
    "Recency": "RecencyMean",
    "Frequency": "FrequencyMean",
    "Monetary": "MonetaryMean"
})
    scatter_data["Cluster"] = scatter_data["Cluster"].astype(str)

    fig3 = px.scatter(
    scatter_data,
    x="RecencyMean",
    y="MonetaryMean",
    color="Cluster",
    size="FrequencyMean",
    hover_data=["RecencyMean", "FrequencyMean", "MonetaryMean"],
    template="plotly_white",
    title="Scatter Plot",
    size_max=40,
)
    st.plotly_chart(fig3, use_container_width=True)

elif choice == 'New Prediction':
    # Tạo dữ liệu cho bảng
    data = {
    "Cluster Number": [0, 1, 2, 3, 4],
    "Customer Segment": [
        "Loyal Customers",    # Cụm 0
        "Lost Customers",     # Cụm 1
        "VIP Customers",      # Cụm 2
        "Dormant Customers",  # Cụm 3
        "Potential Loyalists" # Cụm 4
    ],
    "Characteristics": [
        "Low recency, fairly frequent, moderate-high spending",   # Cụm 0
        "Very high recency, low frequency, low spending",         # Cụm 1
        "Low recency, high frequency, high spending",             # Cụm 2
        "Moderately high recency, moderate frequency, moderate spending",  # Cụm 3
        "Moderate recency, low frequency, low spending (potential to buy more)" # Cụm 4
    ]}


    # Tạo DataFrame từ data
    df_segments = pd.DataFrame(data)

    cluster_strategies = {
    0: {
        "title": "Loyal Customers 🫶",
        "color": "#d1e7dd",
        "strategy": [
            "Duy trì chăm sóc định kỳ.",
            "Cung cấp mã giảm giá nhỏ để giữ chân.",
            "Mời đánh giá/chia sẻ trải nghiệm."
        ]
    },
    1: {
        "title": "Lost Customers 😞",
        "color": "#f8d7da",
        "strategy": [
            "Gửi email nhắc nhở, ưu đãi quay lại.",
            "Khảo sát lý do bỏ đi.",
            "Chạy remarketing (Facebook/Google Ads)."
        ]
    },
    2: {
        "title": "VIP Customers 😎",
        "color": "#e0f7fa",
        "strategy": [
            "Tặng ưu đãi VIP, quyền truy cập sớm sản phẩm mới.",
            "Mời tham gia chương trình Beta/Câu lạc bộ.",
            "Lấy feedback dịch vụ, cá nhân hóa chăm sóc."
        ]
    },
    3: {
        "title": "Dormant Customers 💤",
        "color": "#fff3cd",
        "strategy": [
            "Gửi thông báo khuyến mãi giới hạn.",
            "Gợi ý sản phẩm đã từng xem/mua.",
            "Khuyến khích tương tác lại qua email/app."
        ]
    },
    4: {
        "title": "Potential Loyalists 🚀",
        "color": "#cfe2ff",
        "strategy": [
            "Theo dõi hành vi mua để đẩy khuyến mãi đúng lúc.",
            "Ưu đãi miễn phí vận chuyển.",
            "Kích hoạt thông qua loyalty point."
        ]
    }
}
    def show_cluster_strategy(cluster_id):
        info = cluster_strategies.get(cluster_id)
        if info:
            st.subheader(f"🎯 Strategy for {info['title']}")
            for point in info["strategy"]:
                st.markdown(f"- {point}")


    # Hiển thị bảng trong Streamlit
    st.markdown("### 1️⃣ Customer Segments")
    st.table(df_segments)
    # Chọn data
    # === 2. Dự đoán theo slider (Recency, Frequency, Monetary) ===
    st.markdown("### 2️⃣ Predict using sliders")
    name = st.text_input("Name of Customer")

    recency_val = st.slider("Recency", int(df["Recency"].min()), int(df["Recency"].max()), int(df["Recency"].min()))
    frequency_val = st.slider("Frequency", int(df["Frequency"].min()), int(df["Frequency"].max()), int(df["Frequency"].min()))
    monetary_val = st.slider("Monetary", float(df["Monetary"].min()), float(df["Monetary"].max()), float(df["Monetary"].min()))

    new_data = pd.DataFrame({
        "Recency": [recency_val],
        "Frequency": [frequency_val],
        "Monetary": [monetary_val]
    })
    st.write("### Dữ liệu người dùng nhập:")
    st.dataframe(new_data)

    # Dự đoán và lưu vào session_state
    data_scaled_slider = scaler.transform(new_data)
    predicted_cluster_slider = model.predict(data_scaled_slider)[0]
    segment_slider = df_segments.loc[df_segments["Cluster Number"] == predicted_cluster_slider, "Customer Segment"].values[0]

    st.write("### Dự đoán cụm khách hàng:")
    st.write(f"Customer belongs to Cluster {predicted_cluster_slider} - {segment_slider}.")

    if st.button("Hiển thị chiến lược kinh doanh", key="strategy_slider"):
        show_cluster_strategy(predicted_cluster_slider)

    # === 3. Dự đoán theo Member_number ===
    st.markdown("### 3️⃣ Predict using Member_number")
    st.subheader("Nhập Member_number")
    min_member = int(df["Member_number"].min())
    max_member = int(df["Member_number"].max())

    member_val = st.number_input(
        "Member_number (từ {} đến {})".format(min_member, max_member),
        min_value=min_member,
        max_value=max_member,
        value=min_member,
        step=1
    )

    st.write(f"Bạn đã nhập Member_number: {member_val}")

    if st.button("Xem thông tin & Dự đoán", key="member_button"):
        member_data = df[df["Member_number"] == member_val]

        if not member_data.empty:
            data_to_scale = member_data[["Recency", "Frequency", "Monetary"]]
            scaled_data = scaler.transform(data_to_scale)
            predicted_cluster_member = model.predict(scaled_data)[0]

            row_segment = df_segments.loc[df_segments["Cluster Number"] == predicted_cluster_member, "Customer Segment"]
            segment_name = row_segment.values[0] if not row_segment.empty else "Unknown"

            # Lưu vào session
            st.session_state["member_data"] = member_data
            st.session_state["predicted_cluster_member"] = predicted_cluster_member
            st.session_state["segment_name"] = segment_name
        else:
            st.session_state["member_data"] = None
            st.session_state["predicted_cluster_member"] = None
            st.session_state["segment_name"] = None

    # Hiển thị kết quả dự đoán (nếu đã lưu trước đó)
    if "member_data" in st.session_state and st.session_state["member_data"] is not None:
        st.subheader(f"Thông tin của Member_number: {member_val}")
        st.dataframe(st.session_state["member_data"])
        st.write(f"**Member_number {member_val}** thuộc **Cluster {st.session_state['predicted_cluster_member']}** - **{st.session_state['segment_name']}**.")

        if st.button("Hiển thị chiến lược kinh doanh (theo cụm dự đoán)", key="strategy_member"):
            show_cluster_strategy(st.session_state["predicted_cluster_member"])
    elif "member_data" in st.session_state and st.session_state["member_data"] is None:
        st.error("Không tìm thấy thông tin của Member_number này trong dữ liệu.")

    st.markdown("### 4️⃣ Upload file to predict")

    st.markdown("📤 Tải lên file dữ liệu (CSV hoặc Excel)")
    st.markdown("ℹ️ **Required columns** in the uploaded file: `Name`, `Recency`, `Frequency`, `Monetary`")
    uploaded_file = st.file_uploader("Drag and drop file here", type=["csv", "xlsx"])

    if uploaded_file is not None:
        file_ext = uploaded_file.name.split('.')[-1]

        if file_ext == "csv":
            df_upload = pd.read_csv(uploaded_file)
        elif file_ext == "xlsx":
            df_upload = pd.read_excel(uploaded_file)
        else:
            st.error("❌ Chỉ hỗ trợ file CSV hoặc Excel (.xlsx)")
            st.stop()
        
        st.dataframe(df_upload)


        required_cols = {"Name", "Recency", "Frequency", "Monetary"}
        if required_cols.issubset(df_upload.columns):
            # Chuẩn hóa và dự đoán
            X = df_upload[["Recency", "Frequency", "Monetary"]]
            X_scaled = scaler.transform(X)
            cluster_preds = model.predict(X_scaled)
            df_upload["Predicted Cluster"] = cluster_preds

            # Thêm nhãn cụm
            df_upload = df_upload.merge(
                df_segments[["Cluster Number", "Customer Segment"]],
                left_on="Predicted Cluster",
                right_on="Cluster Number",
                how="left"
            )
            # Xóa cột dư thừa
            df_upload.drop(columns=["Cluster Number"], inplace=True)

            st.write("### 📊 Kết quả dự đoán:")
            st.dataframe(df_upload[["Name", "Recency", "Frequency", "Monetary", "Predicted Cluster", "Customer Segment"]])
        else:
            st.error("⚠️ File CSV phải có đầy đủ các cột: Name, Recency, Frequency, Monetary")

        import io

        csv = df_upload.to_csv(index=False, sep=";", encoding="utf-8")
        st.download_button(
            label="📥 Tải kết quả dự đoán xuống (.csv)",
            data=csv,
            file_name="ket_qua_du_doan.csv",
            mime="text/csv"
        )
