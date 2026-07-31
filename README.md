# Stock Transformer Forecast

金融时序智能风控系统 —— 第二步：Transformer行情预测推理

## 项目定位
承接上游 stock_daily_crawler 的清洗后数据，
使用 Transformer 时序模型对 A 股 22 只精选个股进行 T+1 收益率预测，
输出标准化预测结果供下游风控决策模块调用。

## 快速开始

### 1. 环境安装
    pip3 install -r requirements.txt

### 2. 下载数据
    python3 data/download_data.py

### 3. 训练模型
    python3 scripts/train.py --config config/model_config.yaml

### 4. 生成预测
    python3 scripts/predict.py --date 20260731

## 项目结构
    config/     配置文件
    data/       数据接入层
    models/     模型定义
    training/   训练模块
    inference/  推理与输出
    evaluation/ 评估与回测
    outputs/    输出结果
    scripts/    可执行脚本
    tests/      单元测试

## 输出给下游
- outputs/forecasts/forecast_YYYYMMDD.parquet
- outputs/reports/model_metrics.json
