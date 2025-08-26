# 效应量与功效分析报告

本文档详细介绍了对三个关键指标的效应量（Effect Size）和统计功效（Power Analysis）的计算结果。

---
## Helpfulness Comparison (Paired)

**数据文件**: `Helpfulness/helpfulness_extracted.csv`
**比较列**: `LLMnote vs Community`

### 1. 统计检验结果
- **检验类型**: Wilcoxon Signed-Rank Test
- **样本量 (N)**: 440
- **检验统计量**: `21339.0000`
- **P值**: `0.5615`

### 2. 效应量 (Effect Size)
- **效应量类型**: Cohen's d (approximated for non-parametric test)
- **计算值**: `0.0257`
- **计算公式**: `d = mean(D) / std(D), where D is the array of paired differences`
- **效应量大小解释**: **Very Small**

### 3. 功效分析 (Power Analysis)
#### 后验功效 (Achieved Power)
在当前样本量和计算出的效应量下，本研究检测到真实效应的概率（功效）为 **8.36%**。

#### 前瞻性分析 (Required Sample Size)
为了在未来的研究中达到理想的80%统计功效，需要的样本量大约为 **11923**。

---
## Numerical Helpfulness (Independent)

**数据文件**: `Helpfulness/helpfulness_extracted.csv`
**比较列**: `LLMnote vs Community`

### 1. 统计检验结果
- **检验类型**: Independent Samples T-test
- **样本量 (N)**: n1=440, n2=440
- **检验统计量**: `0.5311`
- **P值**: `0.5955`

### 2. 效应量 (Effect Size)
- **效应量类型**: Cohen's d
- **计算值**: `0.0358`
- **计算公式**: `d = (mean1 - mean2) / pooled_std_dev`
- **效应量大小解释**: **Very Small**

### 3. 功效分析 (Power Analysis)
#### 后验功效 (Achieved Power)
在当前样本量和计算出的效应量下，本研究检测到真实效应的概率（功效）为 **8.28%**。

#### 前瞻性分析 (Required Sample Size)
为了在未来的研究中达到理想的80%统计功效，需要的样本量大约为 **~12244 per group**。

---
## Winning Rate Comparison (Paired)

**数据文件**: `Winning_rate/winning_rate_data_by_participant.csv`
**比较列**: `llm_rate vs community_rate`

### 1. 统计检验结果
- **检验类型**: Paired Samples T-test
- **样本量 (N)**: 22
- **检验统计量**: `0.2324`
- **P值**: `0.8185`

### 2. 效应量 (Effect Size)
- **效应量类型**: Cohen's d
- **计算值**: `0.0495`
- **计算公式**: `d = mean(D) / std(D), where D is the array of paired differences`
- **效应量大小解释**: **Very Small**

### 3. 功效分析 (Power Analysis)
#### 后验功效 (Achieved Power)
在当前样本量和计算出的效应量下，本研究检测到真实效应的概率（功效）为 **5.57%**。

#### 前瞻性分析 (Required Sample Size)
为了在未来的研究中达到理想的80%统计功效，需要的样本量大约为 **3200**。

---
