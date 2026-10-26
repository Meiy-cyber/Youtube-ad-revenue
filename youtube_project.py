import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(page_title="YouTube Analytics Pro", page_icon="📺", layout="wide")

# Saved model
@st.cache_resource
def load_model():
    return joblib.load('youtube_model.pkl')

model = load_model()

# Sidebar Navigation
st.sidebar.title("📌 Navigation Menu")
page = st.sidebar.radio("Go to", ["🚀 Revenue Predictor", "📊 Dataset Insights & Overview"])

# ================= PAGE 1: REVENUE PREDICTOR =================
if page == "🚀 Revenue Predictor":
    st.title("📺 YouTube Ad Revenue Prediction App")
    st.markdown("### Estimate your video's potential earnings instantly using Machine Learning!")
    st.write("---")

    # Layout creation (2 Columns)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Engagement Metrics")
        views = st.slider("Views", min_value=0, max_value=1000000, value=15000, step=100)
        likes = st.slider("Likes", min_value=0, max_value=50000, value=1200, step=10)
        comments = st.slider("Comments", min_value=0, max_value=5000, value=150, step=5)

    with col2:
        st.subheader("⏱️ Content & Channel Details")
        watch_time_minutes = st.number_input("Watch Time (Minutes)", min_value=0.0, value=3500.0, step=50.0)
        video_length_minutes = st.slider("Video Length (Minutes)", min_value=1.0, max_value=180.0, value=12.0)
        subscribers = st.number_input("Subscribers", min_value=0, value=25000, step=500)

    # Engagement rate calculation
    if views > 0:
        engagement_rate = (likes + comments) / views
    else:
        engagement_rate = 0.0

    st.write("---")

    # Center Button for Prediction
    if st.button("🚀 Predict Ad Revenue", use_container_width=True):
        input_data = pd.DataFrame([[
            views, likes, comments, watch_time_minutes, 
            video_length_minutes, subscribers, engagement_rate
        ]], columns=['views', 'likes', 'comments', 'watch_time_minutes', 'video_length_minutes', 'subscribers', 'engagement_rate'])
        
        # Prediction
        prediction = model.predict(input_data)
        
        # Success Result Display
        st.success(f"🎉 Estimated Ad Revenue: **${prediction[0]:.2f} USD**")
        
        # Additional Metrics Cards
        m1, m2, m3 = st.columns(3)
        m1.metric("Engagement Rate", f"{engagement_rate:.4%}")
        m2.metric("Est. Revenue per 1k Views", f"${(prediction[0] / views * 1000):.2f}" if views > 0 else "$0.00")
        m3.metric("Video Duration", f"{video_length_minutes} mins")
        
        st.balloons()

# ================= PAGE 2: DATASET INSIGHTS =================
elif page == "📊 Dataset Insights & Overview":
    st.title("📊 Dataset Insights & Project Overview")
    st.markdown("### Learn more about the YouTube Traffic Analytics metrics.")
    st.write("---")
    
    st.info("💡 **Dataset Explanation:** Each row in the dataset represents performance metrics for a specific video on a specific day.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔹 Core Metrics")
        st.markdown("- **`video_id`**: Unique identifier for the video.")
        st.markdown("- **`date`**: Upload or report date.")
        st.markdown("- **`views`, `likes`, `comments`**: Primary video performance statistics.")
        st.markdown("- **`subscribers`**: Total subscriber count of the channel.")
        
    with col2:
        st.markdown("### 🔹 Engagement & Target")
        st.markdown("- **`watch_time_minutes`**: Total watch time accumulated.")
        st.markdown("- **`video_length_minutes`**: Duration of the content.")
        st.markdown("- **`engagement_rate`**: Custom engineered feature (`(likes + comments) / views`).")
        st.markdown("- **`ad_revenue_usd`**: Revenue generated (**Target Variable**).")
        
    st.write("---")
    st.subheader("🛠️ Tech Stack Used")
    st.markdown("Python, Pandas, NumPy, Scikit-learn, Random Forest Regressor, Joblib, and Streamlit.")