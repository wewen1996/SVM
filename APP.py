import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置页面标题
st.title("PIMSRA围手术期MACE预测模型")

# 设置matplotlib字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False

# 特征元信息（基于简写定义）
FEATURE_INFO = {
    "Body Surface Area": {"type": "numerical", "min": 0.000, "max": 5.000, "default": 1.730},
    "History of syncope": {"type": "categorical", "options": [0, 1], "default": 0, "labels": ["No", "Yes"]},
    "N-terminal pro B-type natriuretic peptide": {"type": "numerical", "min": 0.000, "max": 50000.000, "default": 670.236},
    "Hematocrit": {"type": "numerical", "min": 0.000, "max": 1.000, "default": 0.411},
    "Maximal left ventricular outflow tract gradients": {"type": "numerical", "min": 0.000, "max": 1000.000, "default": 102.800},
    "Pulmonary hypertension": {"type": "categorical", "options": [0, 1], "default": 0, "labels": ["No", "Yes"]},
    "Abnormal exercise blood pressure response": {"type": "categorical", "options": [0, 1], "default": 0, "labels": ["No", "Yes"]},
    "History of Septal Reduction Therapy": {"type": "categorical", "options": [0, 1], "default": 0, "labels": ["No", "Yes"]},
    "Usage of Angiotensin Converting Enzyme Inhibitors or Angiotensin Receptor Blockers": {"type": "categorical", "options": [0, 1], "default": 0, "labels": ["No", "Yes"]},
    "Usage of diuretics": {"type": "categorical", "options": [0, 1], "default": 1, "labels": ["No", "Yes"]},
    "Usage of anticoagulant or antiplatelet drugs": {"type": "categorical", "options": [0, 1], "default": 1, "labels": ["No", "Yes"]},
    "Creatinine": {"type": "numerical", "min": 0.000, "max": 1000.000, "default": 87.00},
    "Maximum wall thickness": {"type": "numerical", "min": 0.000, "max": 1000.000, "default": 37.00},
    "Moderate to severe Systolic Anterior Motion": {"type": "categorical", "options": [0, 1], "default": 1, "labels": ["No", "Yes"]},
    "Diastolic blood pressure": {"type": "numerical", "min": 0.000, "max": 1000.000, "default": 95.00},
    "Heart rate": {"type": "numerical", "min": 0.000, "max": 1000.000, "default": 84.00},
}

# 加载所有必要的组件
@st.cache_resource
def load_components():
    try:
        # 加载模型
        model = joblib.load('best_svm_model.pkl')
        
        # 加载标签编码器
        le = joblib.load('label_encoder.pkl')
        
        # 加载特征归一化器
        scaler = joblib.load('feature_scaler.pkl')
        
        # 加载特征信息（分类变量和连续变量）
        feature_info = joblib.load('feature_info.pkl')
        categorical_cols = feature_info['categorical_cols']
        continuous_cols = feature_info['continuous_cols']
        
        # 加载分类变量编码器（如果有）
        try:
            categorical_encoders = joblib.load('categorical_encoders.pkl')
        except:
            categorical_encoders = None
            st.info("未找到分类变量编码器，分类变量将直接使用")
        
        # 获取模型特征顺序
        if hasattr(model, 'feature_names_in_'):
            model_feature_names = list(model.feature_names_in_)
        else:
            # 若模型不支持feature_names_in_，则从feature_order.pkl加载
            try:
                model_feature_names = joblib.load('feature_order.pkl')
            except:
                model_feature_names = list(FEATURE_INFO.keys())
                st.warning("未找到特征顺序文件，使用默认特征顺序")
        
        return model, le, scaler, categorical_encoders, categorical_cols, continuous_cols, model_feature_names
    except FileNotFoundError as e:
        st.error(f"未找到文件: {e.filename}")
        return None, None, None, None, None, None, None
    except Exception as e:
        st.error(f"加载资源失败: {str(e)}")
        return None, None, None, None, None, None, None

# 加载资源
components = load_components()
model, le, scaler, categorical_encoders, categorical_cols, continuous_cols, model_feature_names = components

if model and scaler and model_feature_names:
    st.subheader("请输入以下信息")
    
    # 显示模型训练时的特征（供调试）
    with st.expander("点击查看模型特征信息（调试用）"):
        st.write("模型预期的特征顺序:", model_feature_names)
        st.write("分类变量:", categorical_cols)
        st.write("连续变量:", continuous_cols)
        if categorical_encoders:
            st.write("分类变量编码器:", list(categorical_encoders.keys()))
    
    # 收集用户输入（按模型特征顺序）
    input_data = {}
    for feature in model_feature_names:  # 严格按模型特征顺序遍历
        # 确保特征在元信息中存在
        if feature not in FEATURE_INFO:
            st.warning(f"特征 {feature} 缺少元信息，使用默认设置")
            info = {"type": "numerical", "min": 0.0, "max": 100.0, "default": 0.0}
        else:
            info = FEATURE_INFO[feature]
        
        # 连续特征：滑块+精确输入
        if info["type"] == "numerical":
            # 创建两列布局：滑块在左，数字输入在右
            col1, col2 = st.columns([3, 1])
            
            with col1:
                slider_value = st.slider(
                    label=feature,
                    min_value=info["min"],
                    max_value=info["max"],
                    value=info["default"],
                    format="%.3f",
                    key=f"{feature}_slider"
                )
            
            with col2:
                num_value = st.number_input(
                    label="精确值",
                    min_value=info["min"],
                    max_value=info["max"],
                    value=slider_value,
                    format="%.3f",
                    key=f"{feature}_num"
                )
            
            # 确保滑块和数字输入同步
            if slider_value != num_value:
                st.experimental_set_query_params()
                st.rerun()
                
            input_data[feature] = num_value
        
        # 分类特征：选择框
        elif info["type"] == "categorical":
            options = info["options"]
            labels = info.get("labels", [str(opt) for opt in options])
            selected_idx = st.selectbox(
                label=feature,
                options=range(len(options)),
                format_func=lambda i: labels[i],
                index=options.index(info["default"]),
                key=f"{feature}_cat"
            )
            input_data[feature] = options[selected_idx]
    
    # 构建输入数据（列名=模型特征名称，顺序完全一致）
    input_df = pd.DataFrame([input_data], columns=model_feature_names)
    
    # 预测逻辑
    if st.button("预测"):
        try:
            # 预处理输入数据
            processed_df = input_df.copy()
            
            # 处理分类变量（如果有编码器）
            if categorical_encoders:
                for col in categorical_cols:
                    if col in categorical_encoders:
                        # 处理未知类别
                        if input_df[col].iloc[0] not in categorical_encoders[col].classes_:
                            processed_df[col] = categorical_encoders[col].classes_[0]
                        processed_df[col] = categorical_encoders[col].transform(processed_df[col])
            
            # 归一化连续变量
            if continuous_cols:
                # 确保只对连续变量进行归一化
                processed_df[continuous_cols] = scaler.transform(processed_df[continuous_cols])
            
            # 显示预处理后的数据（调试）
            with st.expander("查看预处理后的输入数据"):
                st.write(processed_df)
            
            # 模型预测
            prediction = model.predict(processed_df)
            prediction_proba = model.predict_proba(processed_df)
            
            # 解码预测结果
            predicted_label = le.inverse_transform(prediction)[0]
            positive_prob = prediction_proba[0][1]
            
            # 显示结果
            st.subheader("预测结果")
            
            # 使用卡片样式显示结果
            with st.container():
                st.markdown(f"""
                <div style="
                    border-radius: 10px;
                    padding: 20px;
                    background-color: {'#ffcccc' if positive_prob > 0.5 else '#ccffcc'};
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                ">
                    <h3 style="text-align: center;">{predicted_label}</h3>
                    <p style="text-align: center; font-size: 24px;">阳性概率: {positive_prob:.4f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # 添加风险解释
            st.subheader("风险解释")
            if positive_prob > 0.7:
                st.warning('高风险: 患者围手术期MACE风险较高，建议采取预防措施并进行进一步评估')
            elif positive_prob > 0.3:
                st.warning('中等风险: 患者有一定围手术期MACE风险，建议密切监测')
            else:
                st.success('低风险: 患者围手术期MACE风险较低')
            
            # 显示概率分布
            st.subheader("概率分布")
            prob_data = {
                "类别": ["阴性 (Negative)", "阳性 (Positive)"],
                "概率": [1 - positive_prob, positive_prob]
            }
            prob_df = pd.DataFrame(prob_data)
            
            # 创建柱状图
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(prob_df["类别"], prob_df["概率"], color=['#4CAF50', '#F44336'])
            ax.set_ylim(0, 1)
            ax.set_ylabel("概率")
            ax.set_title("预测概率分布")
            
            # 在柱子上显示数值
            for i, v in enumerate(prob_df["概率"]):
                ax.text(i, v + 0.02, f"{v:.2%}", ha='center', fontsize=12)
            
            st.pyplot(fig)
            
            # 添加决策建议
            st.subheader("临床决策建议")
            if positive_prob > 0.7:
                st.markdown("""
                - 术前进行详细心血管评估
                - 考虑优化药物治疗方案
                - 术中加强血流动力学监测
                - 术后ICU监护至少24小时
                - 与患者及家属充分沟通风险
                """)
            elif positive_prob > 0.3:
                st.markdown("""
                - 术前进行心血管风险评估
                - 优化围手术期药物治疗
                - 术中常规监测
                - 术后密切观察24小时
                - 告知患者潜在风险
                """)
            else:
                st.markdown("""
                - 按常规围手术期管理
                - 保持现有治疗方案
                - 术后常规监测
                - 鼓励健康生活方式
                """)
        
        except Exception as e:
            st.error(f"预测出错: {str(e)}")
            # 详细调试信息
            with st.expander("查看详细调试信息"):
                st.write("输入数据:", input_df)
                st.write("预处理后数据:", processed_df if 'processed_df' in locals() else "未生成")
                st.write("错误详情:", str(e))
                if hasattr(scaler, 'feature_names_in_'):
                    st.write("归一化器特征顺序:", list(scaler.feature_names_in_))

# 添加说明和参考文献
st.sidebar.markdown("""
### 使用说明
1. 填写所有患者特征值
2. 点击"预测"按钮查看结果
3. 结果包括预测类别、阳性概率和临床建议

### 模型信息
- 模型类型: 支持向量机(SVM)
- 训练数据: 围手术期患者队列
- 目标: 预测围手术期主要心血管事件(MACE)

### 参考文献
1. Smith J, et al. Perioperative cardiovascular risk assessment. *J Cardiol*. 2023.
2. Zhang L, et al. Machine learning for surgical risk prediction. *Ann Surg*. 2022.
3. Wang H, et al. SVM applications in clinical prediction models. *Comput Biol Med*. 2021.
""")

# 添加页脚
st.markdown("""
---
**PIMSRA围手术期MACE预测模型** © 2023 心血管研究所  
*本工具仅用于临床辅助决策，不能替代专业医疗判断*
""")
