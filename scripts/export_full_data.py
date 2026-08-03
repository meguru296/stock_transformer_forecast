import os
import sys
import json
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_all_features(data_dir="./data/raw/data_center/processed"):
    """加载所有原始特征，合并为宽表"""
    with open(os.path.join(data_dir, "feature_names.json"), "r", encoding="utf-8") as f:
        feature_names = json.load(f)
    
    # 1. 个股特征
    stock_features = {}
    for name in feature_names:
        stock_features[name] = pd.read_parquet(os.path.join(data_dir, f"feature_{name}.parquet"))
    
    # 2. 宏观特征
    macro = pd.read_parquet(os.path.join(data_dir, "macro_features.parquet"))
    
    # 3. 行业特征（列是行业编号）
    sector = pd.read_parquet(os.path.join(data_dir, "sector_features.parquet"))
    
    # 4. 真实收益率
    target_y = pd.read_parquet(os.path.join(data_dir, "target_y.parquet"))
    
    # 5. 股票列表和行业映射
    with open(os.path.join(data_dir, "stock_list.json"), "r", encoding="utf-8") as f:
        stocks = json.load(f)
    with open(os.path.join(data_dir, "stock_sw_sector_map.json"), "r", encoding="utf-8") as f:
        stock_sw_sector_map = json.load(f)
    
    # 转为长格式
    all_data = []
    dates = stock_features[feature_names[0]].index
    
    for stock in stocks:
        sector_name = stock_sw_sector_map.get(stock)
        for date in dates:
            row = {"trade_date": date.strftime("%Y-%m-%d"), "stock_code": stock}
            
            # 个股特征
            for name in feature_names:
                row[name] = stock_features[name].loc[date, stock]
            
            # 宏观特征
            for col in macro.columns:
                row[f"macro_{col}"] = macro.loc[date, col]
            
            # 行业特征（通过映射查找）
            if sector_name and sector_name in sector.columns:
                row["sector_ret"] = sector.loc[date, sector_name]
            else:
                row["sector_ret"] = np.nan
            
            # 真实T+1收益率
            row["actual_return_t1"] = target_y.loc[date, stock]
            
            all_data.append(row)
    
    df = pd.DataFrame(all_data)
    return df

def main():
    # 1. 加载原始特征
    print("加载原始特征数据...")
    features_df = load_all_features()
    print(f"原始数据: {len(features_df)} 行, {len(features_df.columns)} 列")
    print(f"日期范围: {features_df['trade_date'].min()} ~ {features_df['trade_date'].max()}")
    
    # 2. 加载预测结果
    forecast_files = sorted([
        f for f in os.listdir("outputs/forecasts") 
        if f.startswith("forecast_") and f.endswith(".parquet") and "full" not in f
    ])
    
    if forecast_files:
        latest_forecast = os.path.join("outputs/forecasts", forecast_files[-1])
        print(f"\n加载预测结果: {latest_forecast}")
        forecast_df = pd.read_parquet(latest_forecast)
        
        forecast_cols = ["trade_date", "stock_code", "predicted_return", 
                        "confidence", "direction_prob", "model_version"]
        forecast_df = forecast_df[[c for c in forecast_cols if c in forecast_df.columns]]
        
        merged = features_df.merge(forecast_df, on=["trade_date", "stock_code"], how="left")
        print(f"合并后: {len(merged)} 行")
        print(f"有预测的数据: {merged['predicted_return'].notna().sum()} 行")
    else:
        print("\n未找到预测结果")
        merged = features_df
    
    # 3. 保存
    os.makedirs("outputs/forecasts", exist_ok=True)
    output_path = "outputs/forecasts/full_data_with_forecast.parquet"
    merged.to_parquet(output_path, index=False)
    
    print(f"\n✅ 完整数据已保存: {output_path}")
    print(f"   总行数: {len(merged)}")
    print(f"   总列数: {len(merged.columns)}")
    print(f"\n列名: {list(merged.columns)}")
    
    # 4. 显示最新一天有预测的数据
    if "predicted_return" in merged.columns:
        latest_date = merged[merged["predicted_return"].notna()]["trade_date"].max()
        latest = merged[merged["trade_date"] == latest_date].nlargest(5, "predicted_return")
        print(f"\n📅 {latest_date} 原始数据 + 预测 Top5:")
        cols = ["stock_code", "predicted_return", "confidence", "volume", "close", "actual_return_t1"]
        cols = [c for c in cols if c in latest.columns]
        print(latest[cols].to_string(index=False))

if __name__ == "__main__":
    main()
