# 下游接口文档

## 1. 预测输出文件

位置: `outputs/forecasts/forecast_YYYYMMDD.parquet`

| 字段 | 类型 | 说明 |
|------|------|------|
| trade_date | string | 预测日，格式 YYYY-MM-DD |
| stock_code | string | 股票代码 |
| predicted_return | float | 预测T+1对数收益率 |
| actual_return | float | T+1真实收益率（回测用） |
| confidence | float | 预测置信度 |
| direction_prob | float | 上涨概率 |
| model_version | string | 模型版本 |
| inference_time | string | 推理时间戳 |

## 2. 使用示例

```python
import pandas as pd
df = pd.read_parquet("outputs/forecasts/forecast_20260731.parquet")
top5 = df.groupby("trade_date").apply(lambda x: x.nlargest(5, "predicted_return"))
