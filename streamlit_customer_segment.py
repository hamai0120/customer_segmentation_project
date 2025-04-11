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



st.title("Data Science")
st.write("## Customer Segment Project")



menu = ["Overview", "Build Project", "New Prediction"]
choice = st.sidebar.selectbox('Menu', menu)
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

    # Hiển thị bảng trong Streamlit
    st.write("##### 1. Customer Segments")
    st.table(df_segments)
    # Chọn data
    st.write("##### 2. Input/Select data")
    name = st.text_input("Name of Customer")
    # Tìm min/max của Recency
    min_recency = int(df["Recency"].min())
    max_recency = int(df["Recency"].max())

    # Thanh trượt cho Recency
    recency_val = st.slider(
        "Recency",
        min_value=min_recency,
        max_value=max_recency,
        value=min_recency  
    )

    # Tìm min/max của Frequency
    min_frequency = int(df["Frequency"].min())
    max_frequency = int(df["Frequency"].max())

    # Thanh trượt cho Frequency
    frequency_val = st.slider(
        "Frequency",
        min_value=min_frequency,
        max_value=max_frequency,
        value=min_frequency
    )

    # Tìm min/max của Monetary 
    min_monetary = float(df["Monetary"].min())
    max_monetary = float(df["Monetary"].max())

    # Thanh trượt cho Monetary
    monetary_val = st.slider(
        "Monetary",
        min_value=min_monetary,
        max_value=max_monetary,
        value=min_monetary
    )

    # Hiển thị kết quả người dùng chọn
    new_data = pd.DataFrame({
    "Recency": [recency_val],
    "Frequency": [frequency_val],
    "Monetary": [monetary_val]
})  
    st.write("### Dữ liệu người dùng nhập:")
    st.dataframe(new_data)

    # Scale dữ liệu (chuẩn hóa) dùng scaler đã lưu
    new_data_scaled = scaler.transform(new_data)

    # Dùng model KMeans đã lưu để dự đoán cluster cho dữ liệu mới
    pred_cluster = model.predict(new_data_scaled)

    st.write("### Dự đoán cụm khách hàng:")
    predicted_cluster = pred_cluster[0]
    st.write(f"Customer {name} belongs to Cluster {predicted_cluster} - {df_segments.loc[df_segments['Cluster Number'] == predicted_cluster, 'Customer Segment'].values[0]}.")


    st.write("##### 3. Dự đoán dựa theo Member_number")

    # Xác định min, max của Member_number (ép về int để dùng cho number_input)
    min_member = int(df["Member_number"].min())
    max_member = int(df["Member_number"].max())

    st.subheader("Nhập Member_number")

    member_val = st.number_input(
        "Member_number (từ {} đến {})".format(min_member, max_member),
        min_value=min_member,
        max_value=max_member,
        value=min_member,
        step=1
    )

    st.write(f"Bạn đã nhập Member_number: {member_val}")
    if st.button("Xem thông tin & Dự đoán"):
    # Tìm dòng dữ liệu theo member_number (giả sử member_number là cột trong df)
        member_data = df[df["Member_number"] == member_val]
    
        if not member_data.empty:
            st.subheader(f"Thông tin của Member_number: {member_val}")
            st.dataframe(member_data)
            
            # Lấy các cột cần thiết để dự đoán: Recency, Frequency, Monetary
            data_to_scale = member_data[["Recency", "Frequency", "Monetary"]]
            
            # Chuẩn hóa dữ liệu
            scaled_data = scaler.transform(data_to_scale)
            
            # Dự đoán cụm
            predicted_cluster = model.predict(scaled_data)[0]
            # Lấy tên segment từ bảng df_segments
            row_segment = df_segments.loc[df_segments["Cluster Number"] == predicted_cluster, "Customer Segment"]
            
            if not row_segment.empty:
                segment_name = row_segment.values[0]
            else:
                segment_name = "Unknown"

            st.write(f"**Member_number {member_val}** thuộc **Cluster {predicted_cluster}** - **{segment_name}**.")
        else:
            st.error("Không tìm thấy thông tin của Member_number này trong dữ liệu.")