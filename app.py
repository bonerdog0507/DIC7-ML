import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import io

# 讓 matplotlib 支援中文顯示 
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] # 針對 Windows 通常有效
plt.rcParams['axes.unicode_minus'] = False

@st.cache_data
def generate_synthetic_data(n, a, b, noise_std, random_seed):
    """產生用於迴歸模型的合成資料 (已快取)。"""
    np.random.seed(random_seed)
    x = np.random.uniform(-100, 100, n)
    # y = a*x + b + noise
    noise = np.random.normal(0.0, noise_std, n)
    y = a * x + b + noise
    
    df = pd.DataFrame({"x": x, "y": y})
    params = {"a": a, "b": b, "noise_std": noise_std}
    return df, params

@st.cache_data
def prepare_data(df):
    """將特徵進行切割與標準化處理 (已快取)。"""
    X = df[['x']]
    y = df['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler

@st.cache_resource
def train_model(_X_train_scaled, _y_train):
    """訓練線性迴歸模型 (Resource 快取)。"""
    model = LinearRegression()
    model.fit(_X_train_scaled, _y_train)
    return model

def main():
    st.set_page_config(page_title="CRISP-DM 階段", layout="wide")
    
    # 側邊欄 (Sidebar)
    st.sidebar.title("CRISP-DM 工作流程")
    st.sidebar.markdown("透過合成的線性迴歸問題來探索 CRISP-DM 方法論。")
    st.sidebar.header("資料產生參數")
    
    a = st.sidebar.slider("斜率 (a)", -50.00, 50.00, 5.00)
    b = st.sidebar.slider("截距 (b)", -50.00, 50.00, 10.00)
    noise_std = st.sidebar.slider("雜訊標準差 (σ)", 0.00, 50.00, 10.00)
    n = st.sidebar.slider("資料點數量 (n)", 100, 1000, 500)
    random_seed = st.sidebar.slider("隨機種子", 0, 100, 42)
    
    # Enable pipeline automatically on page load
    st.session_state['generated'] = True
    
    if st.session_state['generated']:
        df, params = generate_synthetic_data(n, a, b, noise_std, random_seed)
        X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = prepare_data(df)
        model = train_model(X_train_scaled, y_train)

    # 主畫面 (Main Area)
    st.title("CRISP-DM 方法論 (Methodology)")
    st.markdown("一個結構化的規劃方法，適用於資料分析、資料探勘與機器學習專案。")
    st.divider()

    # Split into 2 columns for cleaner UI
    col1, col2 = st.columns([1, 1])

    with col1:
        # 1. 商業理解
        st.header("1. 商業理解 (Business Understanding)")
        st.write("從商業角度出發，確立專案的目標與需求。")
        
        # 2. 資料理解
        st.header("2. 資料理解 (Data Understanding)")
        if 'generated' in st.session_state:
            st.dataframe(df.head(), use_container_width=True)
        else:
            st.info("等待資料產生中...")
            
        # 3. 資料準備
        st.header("3. 資料準備 (Data Preparation)")
        if 'generated' in st.session_state:
            st.write(f"- **訓練集 (Training set)**: X 維度 {X_train_scaled.shape}, y 維度 {y_train.shape}")
            st.write(f"- **測試集 (Testing set)**: X 維度 {X_test_scaled.shape}, y 維度 {y_test.shape}")
            st.success("成功完成資料切割（比例 80/20）及特徵的標準化縮放。")
        else:
            st.info("等待資料產生中...")

    with col2:
        # 4. 建立模型
        st.header("4. 建立模型 (Modeling)")
        if 'generated' in st.session_state:
            st.write("**訓練的模型: 線性迴歸 (Linear Regression)**")
            st.write(f"- **學習到的斜率 (Learned slope)**: {model.coef_[0]:.4f}")
            st.write(f"- **學習到的截距 (Learned intercept)**: {model.intercept_:.4f}")
        else:
            st.info("等待資料準備中...")
            
        # 5. 模型評估
        st.header("5. 模型評估 (Evaluation)")
        if 'generated' in st.session_state:
            y_pred = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("MSE", f"{mse:.4f}")
            mc2.metric("RMSE", f"{rmse:.4f}")
            mc3.metric("R²", f"{r2:.4f}")
        else:
            st.info("等待建立模型中...")
            
        # 6. 佈署
        st.header("6. 佈署 (Deployment)")
        if 'generated' in st.session_state:
            dep_col1, dep_col2 = st.columns(2)
            
            with dep_col1:
                st.subheader("預測")
                user_x = st.number_input("輸入 'x' 進行預測:", value=0.0, step=1.0)
                if st.button("輸出預測 y"):
                    # 將使用者的 input 透過 scaler 轉換
                    user_x_scaled = scaler.transform(np.array([[user_x]]))
                    pred_y = model.predict(user_x_scaled)[0]
                    st.success(f"**預測結果 y**: {pred_y:.4f}")
                    
            with dep_col2:
                st.subheader("匯出")
                # 將模型跟標準化函數一起打包以方便未來使用
                buffer = io.BytesIO()
                joblib.dump({"model": model, "scaler": scaler}, buffer)
                buffer.seek(0)
                
                st.download_button(
                    label="下載模型 (.pkl)",
                    data=buffer,
                    file_name="trained_model_pipeline.pkl",
                    mime="application/octet-stream"
                )
        else:
            st.info("先完成 Pipeline 即可佈署。")

    # 視覺化圖表
    if 'generated' in st.session_state:
        st.divider()
        st.subheader("視覺化分析 (Visual Analysis)")
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.scatter(df['x'], df['y'], alpha=0.5, label='實際資料 (Actual Data)', color='royalblue')
        
        x_min, x_max = df['x'].min(), df['x'].max()
        x_line = np.linspace(x_min, x_max, 100).reshape(-1, 1)
        x_line_scaled = scaler.transform(x_line)
        y_line = model.predict(x_line_scaled)
        
        ax.plot(x_line, y_line, color='red', linewidth=3, label='迴歸擬合線 (Regression Fit)')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('線性迴歸預測與實際資料表現')
        ax.legend()
        st.pyplot(fig)

if __name__ == "__main__":
    main()
