import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# 设置matplotlib字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 加载保存的模型和相关组件
@st.cache_resource
def load_models():
    try:
        # 加载核心模型和编码器
        model = joblib.load('best_svm_model.pkl')
        le = joblib.load('label_encoder.pkl')
        
        # 加载训练时使用的特征缩放器和特征统计信息
        scaler = joblib.load('scaler.pkl')  # 用于归一化的缩放器
        with open('feature_names.txt', 'r') as f:
            feature_names = f.read().splitlines()
            
        # 获取连续特征名称（从特征范围定义中提取）
        continuous_features = [name for name, props in feature_ranges.items() 
                             if props["type"] == "numerical"]
            
        return model, le, scaler, feature_names, continuous_features
    except FileNotFoundError as e:
        st.error(f"缺失必要文件: {e}")
        return None, None, None, None, None
    except Exception as e:
        st.error(f"加载模型时出错: {str(e)}")
        return None, None, None, None, None

# 特征范围定义（包含类型信息用于区分连续/分类特征）
feature_ranges = {
    "Body Surface Area":{"type": "numerical", "min": 0.000, "max": 5.000, "default": 1.730 },
    "History of syncope": {"type": "categorical", "options": [0, 1], "default": 0},
    "N-terminal pro B-type natriuretic peptide": {"type": "numerical", "min": 0.000, "max": 50000.000, "default": 670.236},
    "Hematocrit": {"type": "numerical", "min": 0.000, "max": 1.000, "default": 0.411},
    "Maximal left ventricular outflow tract gradients": {"type": "numerical", "min": 0.000, "max": 1000.000, "default": 102.800},
    "Pulmonary hypertension": {"type": "categorical", "options": [0, 1], "default": 0},
    "Abnormal exercise blood pressure response": {"type": "categorical", "options": [0, 1], "default": 0},
    "History of Septal Reduction Therapy": {"type": "categorical", "options": [0, 1], "default": 0},
    "Usage of Angiotensin Converting Enzyme Inhibitors or Angiotensin Receptor Blockers": {"type": "categorical", "options": [0, 1], "default": 0},
    "Usage of diuretics": {"type": "categorical", "options": [0, 1], "default": 1},
    "Usage of anticoagulant or antiplatelet drugs": {"type": "categorical", "options": [0, 1], "default": 1},
    "Creatinine": {"type": "numerical", "min": 0.000, "max": 1000.000, "default": 87.00},
    "Maximum wall thickness": {"type": "numerical", "min": 0.000, "max": 1000.000, "default": 37.00},
    "Moderate to severe Systolic Anterior Motion": {"type": "categorical", "options": [0, 1], "default": 1},
    "Diastolic blood pressure": {"type": "numerical", "min": 0.000, "max": 1000.000, "default": 95.00},
    "Heart rate": {"type": "numerical", "min": 0.000, "max": 1000.000, "default": 84.00},
}

# 验证特征顺序一致性
def check_feature_consistency(input_features, trained_features):
    if list(input_features.keys()) != trained_features:
        st.warning("特征顺序与模型训练时不一致！这可能导致预测错误。")
        st.info(f"训练时特征顺序: {', '.join(trained_features[:3])}...")
        st.info(f"当前输入顺序: {', '.join(list(input_features.keys())[:3])}...")
        return False
    return True

# 连续特征归一化处理函数（与训练时保持一致）
def normalize_continuous_features(features_df, scaler, continuous_features):
    """
    对连续特征进行归一化处理，分类特征保持不变
    
    参数:
    - features_df: 包含所有特征的DataFrame
    - scaler: 训练时使用的缩放器
    - continuous_features: 连续特征名称列表
    
    返回:
    - 处理后的特征DataFrame
    """
    # 复制原始数据以避免修改
    normalized_df = features_df.copy()
    
    # 仅对连续特征进行归一化
    if continuous_features and len(continuous_features) > 0:
        # 获取连续特征的值
        continuous_data = normalized_df[continuous_features].values
        
        # 应用训练好的缩放器进行归一化
        normalized_continuous = scaler.transform(continuous_data)
        
        # 将归一化后的值放回DataFrame
        normalized_df[continuous_features] = normalized_continuous
    
    return normalized_df

# Streamlit界面
st.title("PIMSRA围手术期MACE发生概率预测模型")

# 加载模型组件
model, le, scaler, feature_names, continuous_features = load_models()

# 如果模型加载成功，显示应用内容
if model and le and scaler and feature_names and continuous_features:
    # 检查特征顺序一致性
    check_feature_consistency(feature_ranges, feature_names)
    
    # 显示归一化信息
    with st.expander("关于特征归一化", expanded=False):
        st.info(f"系统将自动对以下连续特征进行归一化处理（与模型训练时保持一致）：\n{', '.join(continuous_features)}")
    
    # 动态生成输入项
    st.header("输入以下特征值：")
    feature_data = {}  # 使用字典存储特征名和值，确保顺序正确
    
    for feature, properties in feature_ranges.items():
        if properties["type"] == "numerical":
            value = st.number_input(
                label=f"{feature} ({properties['min']}-{properties['max']})",
                min_value=float(properties["min"]),
                max_value=float(properties["max"]),
                value=float(properties["default"]),
                format="%.3f"
            )
        elif properties["type"] == "categorical":
            # 为分类变量添加说明
            options_with_labels = {0: "否", 1: "是"}
            selected_label = st.selectbox(
                label=f"{feature}",
                options=list(options_with_labels.keys()),
                format_func=lambda x: options_with_labels[x],
                index=properties["options"].index(properties["default"])
            )
            value = selected_label
        feature_data[feature] = value

    # 转换为DataFrame（保持特征顺序）
    features_df = pd.DataFrame([feature_data], columns=feature_names)
    
    # 预测与SHAP可视化
    if st.button("预测"):
        try:
            # 关键步骤：对连续特征进行归一化处理
            with st.spinner("正在处理输入特征（包含归一化）..."):
                normalized_features_df = normalize_continuous_features(
                    features_df, 
                    scaler, 
                    continuous_features
                )
            
            # 转换为模型输入格式
            features_scaled = normalized_features_df.values
            
            # 模型预测（使用归一化后的特征）
            predicted_class = model.predict(features_scaled)[0]
            predicted_proba = model.predict_proba(features_scaled)[0]
            
            # 解码预测类别
            predicted_class_label = le.inverse_transform([predicted_class])[0]
            
            # 提取预测的类别概率
            target_probability = predicted_proba[1] * 100  # 关注事件发生的概率
            
            # 风险分层逻辑
            if target_probability < 30:
                risk_level = "低风险"
            elif 30 <= target_probability < 70:
                risk_level = "中风险"
            else:
                risk_level = "高风险"
            
            # 显示输入回顾和归一化结果
            with st.expander("查看输入特征及归一化结果", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("原始输入值：")
                    st.dataframe(features_df.T, height=400)
                with col2:
                    st.subheader("归一化后的值：")
                    st.dataframe(normalized_features_df.T, height=400)
            
            # 显示预测结果
            st.subheader("预测结果:")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("PIMSRA围手术期MACE发生概率", f"{target_probability:.2f}%")
            with col2:
                st.metric("风险等级", risk_level)
            
            # 临床解读
            st.subheader("临床解读:")
            if risk_level == "低风险":
                st.write(" - 低风险 (<30%): 建议常规监测，遵循标准临床流程")
            elif risk_level == "中风险":
                st.write(" - 中风险 (30-70%): 建议加强随访观察，考虑优化药物治疗方案")
            else:
                st.write(" - 高风险 (≥70%): 建议考虑立即临床干预，制定详细的风险降低策略")
            st.info("注: 本预测工具仅作为临床参考，不能替代医生的专业判断。")
            
            # 计算SHAP值并展示特征重要性
            st.subheader("模型可解释性 (SHAP分析):")
            
            # 创建背景数据（使用默认值作为参考并归一化）
            default_values = {name: props["default"] for name, props in feature_ranges.items()}
            background_df = pd.DataFrame([default_values], columns=feature_names)
            background_normalized = normalize_continuous_features(
                background_df, 
                scaler, 
                continuous_features
            )
            
            # 定义预测函数（返回正类概率）
            def predict_fn(X):
                if isinstance(X, pd.DataFrame):
                    X = X.values
                return model.predict_proba(X)[:, 1]  # 只关注正类概率
            
            # 初始化解释器
            with st.spinner("正在计算特征重要性..."):
                try:
                    # 使用KernelExplainer适合SVM模型
                    explainer = shap.KernelExplainer(
                        predict_fn, 
                        background_normalized.values, 
                        link="logit"
                    )
                    # 计算SHAP值
                    shap_values = explainer.shap_values(normalized_features_df.values, nsamples=50)
                    
                    # 1. 力导向图
                    st.write("### 特征对当前预测的影响:")
                    force_plot = shap.force_plot(
                        explainer.expected_value,
                        shap_values[0],
                        normalized_features_df.iloc[0],
                        matplotlib=False,
                        show=False,
                        feature_names=feature_names
                    )
                    
                    shap.save_html("shap_force_plot.html", force_plot)
                    st.components.v1.html(open("shap_force_plot.html", "r").read(), height=200)
                    
                    # 2. 特征重要性条形图
                    st.write("### 特征重要性排序:")
                    plt.figure(figsize=(10, 8))
                    shap.summary_plot(
                        shap_values, 
                        normalized_features_df,
                        feature_names=feature_names,
                        plot_type="bar",
                        show=False,
                        sort=True
                    )
                    plt.tight_layout()
                    st.pyplot(plt.gcf())
                    plt.close()
                    
                except Exception as e:
                    st.warning(f"SHAP分析出错: {str(e)}")
                    with st.expander("查看详细错误信息"):
                        st.error(str(e))
        
        except Exception as e:
            st.error(f"预测过程出错: {str(e)}")
            with st.expander("查看详细错误信息"):
                st.error(str(e))
else:
    st.error("模型加载失败，请确保以下文件存在：")
    st.info("1. best_svm_model.pkl - 最佳SVM模型")
    st.info("2. label_encoder.pkl - 标签编码器")
    st.info("3. scaler.pkl - 特征缩放器（如StandardScaler）")
    st.info("4. feature_names.txt - 训练特征名称列表")

# 添加如何生成缺失文件的指导
with st.expander("如何生成缺失的缩放器和特征名称文件？", expanded=False):
    st.code("""
# 在模型训练代码中添加以下代码保存缩放器和特征名称
from sklearn.preprocessing import StandardScaler

# 假设您的训练数据是X_train_scale（已归一化）和X_train（原始数据）
# 训练缩放器
scaler = StandardScaler()
scaler.fit(X_train[continuous_features])  # 只对连续特征拟合

# 保存缩放器
joblib.dump(scaler, 'scaler.pkl')

# 保存特征名称
with open('feature_names.txt', 'w') as f:
    for name in X_train.columns:
        f.write(f"{name}\\n")
    """, language="python")
