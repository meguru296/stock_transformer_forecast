import os
import sys
import json
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 加载残差统计
residual_stats = np.load("outputs/residual_stats.npz")
RESIDUAL_MEAN = float(residual_stats["mean"])      # 0.000566
RESIDUAL_SIGMA = float(residual_stats["sigma"])     # 0.028062
Z_95 = 1.96

def load_features_for_vol(data_dir="./data/raw/data_center/processed"):
    """加载计算波动率所需的历史数据"""
    log_ret = pd.read_parquet(os.path.join(data_dir, "feature_log_return.parquet"))
    return log_ret

def calc_realized_vol(log_ret_df, date, stock, window=20, annualize_days=252):
    """计算历史已实现波动率（年化）"""
    try:
        idx = log_ret_df.index.get_loc(date)
        start = max(0, idx - window + 1)
        series = log_ret_df.iloc[start:idx+1][stock].dropna()
        if len(series) < 5:
            return np.nan
        return series.std() * np.sqrt(annualize_days)
    except:
        return np.nan

def calc_5day_cumulative(log_ret_df, date, stock):
    """计算历史未来5日累计对数收益率（用于回测参考）"""
    try:
        idx = log_ret_df.index.get_loc(date)
        if idx + 5 >= len(log_ret_df):
            return np.nan
        future_5d = log_ret_df.iloc[idx+1:idx+6][stock]
        return future_5d.sum()
    except:
        return np.nan

def main():
    print("=" * 60)
    print("增强版预测输出 - 下游风控指标补全")
    print("=" * 60)
    
    # 1. 加载完整预测（历史回测+最新）
    forecast_path = "outputs/forecasts/forecast_full_2026-07-28.parquet"
    if not os.path.exists(forecast_path):
        print(f"错误: 未找到 {forecast_path}")
        print("请先运行: python3 scripts/predict_full.py")
        return
    
    pred_df = pd.read_parquet(forecast_path)
    print(f"\n1. 加载预测数据: {len(pred_df)} 行")
    
    # 2. 加载历史对数收益率（用于计算波动率）
    print("\n2. 加载历史数据计算波动率...")
    log_ret_df = load_features_for_vol()
    
    # 3. 逐行计算增强指标
    print("\n3. 计算增强指标...")
    
    results = []
    for _, row in pred_df.iterrows():
        date = row["trade_date"]
        stock = row["stock_code"]
        pred_ret = row["predicted_return"]  # 对数收益率
        
        # --- 核心指标 ---
        
        # 3.1 95%置信区间（基于训练集残差分布）
        ci_lower = pred_ret - Z_95 * RESIDUAL_SIGMA
        ci_upper = pred_ret + Z_95 * RESIDUAL_SIGMA
        
        # 3.2 未来1日年化波动率（历史已实现，代理）
        vol_1d = calc_realized_vol(log_ret_df, date, stock, window=20, annualize_days=252)
        
        # 3.3 未来5日年化波动率（历史5日窗口已实现）
        # 计算过去20个5日窗口的标准差
        try:
            idx = log_ret_df.index.get_loc(date)
            start = max(0, idx - 24)
            rolling_5d = []
            for i in range(start, idx - 3):
                if i + 5 < len(log_ret_df):
                    rolling_5d.append(log_ret_df.iloc[i:i+5][stock].sum())
            vol_5d = np.std(rolling_5d) * np.sqrt(252 / 5) if len(rolling_5d) >= 5 else np.nan
        except:
            vol_5d = np.nan
        
        # 3.4 未来5日累计对数收益率（历史真实值，回测参考）
        cum_5d_actual = calc_5day_cumulative(log_ret_df, date, stock)
        
        # 3.5 预测收盘价（需外部当日收盘价，这里基于假设100给出相对值）
        # 实际使用需要: pred_close = today_close * np.exp(pred_ret)
        # 由于上游无close数据，输出相对乘数
        price_multiplier = np.exp(pred_ret)
        price_multiplier_upper = np.exp(ci_upper)
        price_multiplier_lower = np.exp(ci_lower)
        
        results.append({
            # 基础字段
            "trade_date": date,
            "stock_code": stock,
            
            # 预测收益率（核心）
            "pred_log_return_1d": pred_ret,
            "pred_pct_return_1d": (np.exp(pred_ret) - 1) * 100,  # 百分比形式
            
            # 95%置信区间
            "ci_lower_95_log": ci_lower,
            "ci_upper_95_log": ci_upper,
            "ci_lower_95_pct": (np.exp(ci_lower) - 1) * 100,
            "ci_upper_95_pct": (np.exp(ci_upper) - 1) * 100,
            
            # 预测收盘价（相对乘数，需外部当日收盘价）
            "price_multiplier": price_multiplier,
            "price_multiplier_lower": price_multiplier_lower,
            "price_multiplier_upper": price_multiplier_upper,
            # "pred_close": "需外部当日收盘价 * price_multiplier",
            # "pred_close_lower": "需外部当日收盘价 * price_multiplier_lower",
            # "pred_close_upper": "需外部当日收盘价 * price_multiplier_upper",
            
            # 波动率（历史已实现，代理指标）
            "realized_vol_1d_annual": vol_1d,      # 1日年化波动率
            "realized_vol_5d_annual": vol_5d,      # 5日年化波动率
            
            # 5日累计（历史真实，回测参考）
            "actual_cum_5d_log_return": cum_5d_actual,
            "actual_cum_5d_pct_return": (np.exp(cum_5d_actual) - 1) * 100 if not np.isnan(cum_5d_actual) else np.nan,
            
            # 保留原有字段
            "confidence": row["confidence"],
            "direction_prob": row["direction_prob"],
            "actual_return_t1": row.get("actual_return", np.nan),
            "model_version": row["model_version"],
            "inference_time": row["inference_time"],
        })
    
    enhanced_df = pd.DataFrame(results)
    
    # 4. 保存
    os.makedirs("outputs/forecasts", exist_ok=True)
    output_path = "outputs/forecasts/enhanced_forecast_for_risk.parquet"
    enhanced_df.to_parquet(output_path, index=False)
    
    print(f"\n✅ 增强版预测已保存: {output_path}")
    print(f"   总行数: {len(enhanced_df)}")
    print(f"   总列数: {len(enhanced_df.columns)}")
    
    # 5. 输出指标说明
    print("\n" + "=" * 60)
    print("📋 字段说明与数据来源")
    print("=" * 60)
    print("""
【模型直接输出】✅
  pred_log_return_1d      : T+1对数收益率预测（模型核心输出）
  pred_pct_return_1d      : T+1百分比收益率预测
  confidence              : 预测置信度（|预测值|）
  direction_prob          : 上涨概率（1=涨, 0=跌）

【基于残差统计构造】⚠️ 代理指标
  ci_lower_95_log / ci_upper_95_log    : 95%置信区间（对数）
  ci_lower_95_pct / ci_upper_95_pct    : 95%置信区间（百分比）
  依据: 训练集残差σ=0.028062, 假设近似正态分布
  
  price_multiplier / price_multiplier_lower / price_multiplier_upper
  : 收盘价相对乘数（需外部当日收盘价 × 该值）

【历史已实现波动率】⚠️ 非模型预测
  realized_vol_1d_annual  : 过去20日对数收益标准差 × √252
  realized_vol_5d_annual  : 过去20个5日窗口标准差 × √(252/5)
  
【历史回测参考】⚠️ 非未来预测
  actual_cum_5d_log_return : 历史真实未来5日累计对数收益
  actual_cum_5d_pct_return : 历史真实未来5日累计百分比收益

【外部依赖】❌ 上游未提供
  原始收盘价(close)        : 上游数据无此字段
  如需预测收盘价绝对值    : 需外部提供当日收盘价 × price_multiplier
""")

    # 6. 最新一天示例
    latest_date = enhanced_df["trade_date"].max()
    latest = enhanced_df[enhanced_df["trade_date"] == latest_date].nlargest(5, "pred_log_return_1d")
    print(f"\n📅 {latest_date} 增强预测示例 (Top5):")
    cols = ["stock_code", "pred_pct_return_1d", "ci_lower_95_pct", "ci_upper_95_pct", 
            "realized_vol_1d_annual", "actual_cum_5d_pct_return"]
    print(latest[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}" if not np.isnan(x) else "N/A"))

if __name__ == "__main__":
    main()
