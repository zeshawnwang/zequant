"""
批量计算缺失的技术指标因子列并写入数据库。

计算以下因子并写入 factors_wide：
- ma5, ma10, ma20, ma21, ma60, ma120 (SMA值)
- ma_alignment_score (均线多头排列得分)
- ma60_trend, ma120_trend (均线斜率)
- macd_above_zero (MACD > 0)
- macd_golden_cross (MACD上穿信号线)
- volume_breakout_ratio (量比)
- volume_contraction (量缩程度: 5日均量 / 20日均量)
- ma_convergence (均线粘合度)
- chip_concentration (筹码集中度)
- ma_angle_20 (20日均线角度)
"""
import os,sys,logging,numpy as np,pandas as pd
from datetime import datetime

sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..')))
from core.database import Database

logging.basicConfig(level=logging.INFO,format="%(asctime)s [%(levelname)s] %(message)s")
logger=logging.getLogger("compute_factors")

FACTOR_NAMES=[
    'ma5','ma10','ma20','ma21','ma60','ma120',
    'ma_alignment_score','ma60_trend','ma120_trend',
    'macd_above_zero','macd_golden_cross',
    'volume_breakout_ratio','volume_contraction',
    'ma_convergence','chip_concentration','ma_angle_20',
]

def compute_sma(series,w):
    """滚动SMA."""
    return series.rolling(window=w,min_periods=max(5,w//2)).mean()

def compute_ma_angle(series,w):
    """均线角度: 当前ma值 / w天前ma值 - 1, 正=向上."""
    return series / series.shift(w) - 1.0

def main():
    db=Database()
    logger.info("读取日线数据...")
    bars=db.get_daily_bars(columns=['symbol','date','close','volume'])
    bars['date']=pd.to_datetime(bars['date'])
    logger.info(f"日线: {bars.shape}, {bars['symbol'].nunique()}只")

    # 排序
    bars=bars.sort_values(['symbol','date']).reset_index(drop=True)

    logger.info("计算SMA因子...")
    bars['ma5']=bars.groupby('symbol')['close'].transform(lambda x:compute_sma(x,5))
    bars['ma10']=bars.groupby('symbol')['close'].transform(lambda x:compute_sma(x,10))
    bars['ma20']=bars.groupby('symbol')['close'].transform(lambda x:compute_sma(x,20))
    bars['ma21']=bars.groupby('symbol')['close'].transform(lambda x:compute_sma(x,21))
    bars['ma60']=bars.groupby('symbol')['close'].transform(lambda x:compute_sma(x,60))
    bars['ma120']=bars.groupby('symbol')['close'].transform(lambda x:compute_sma(x,120))

    logger.info("计算衍生因子...")
    # 均线多头排列得分
    bars['ma_alignment_score']=(
        (bars['ma5']>bars['ma20']).astype(float)+
        (bars['ma20']>bars['ma60']).astype(float)+
        (bars['ma60']>bars['ma120']).astype(float)
    )/3.0

    # 均线趋势(斜率)
    bars['ma60_trend']=bars.groupby('symbol')['ma60'].transform(lambda x:compute_ma_angle(x,5))
    bars['ma120_trend']=bars.groupby('symbol')['ma120'].transform(lambda x:compute_ma_angle(x,10))

    # MACD在零轴上方
    macd_df=db.get_factors(factor_names=['macd','macd_signal'],start_date='2018-01-01')
    if macd_df is not None and not macd_df.empty:
        macd_df['date']=pd.to_datetime(macd_df['date'])
        bars=bars.merge(macd_df[['date','symbol','macd','macd_signal']],on=['date','symbol'],how='left')
        bars['macd_above_zero']=(bars['macd']>0).astype(float)
        bars['macd_golden_cross']=((bars['macd']>bars['macd_signal'])&
                                    (bars.groupby('symbol')['macd'].shift(1)<=bars.groupby('symbol')['macd_signal'].shift(1))).astype(float)
    else:
        bars['macd_above_zero']=0.0
        bars['macd_golden_cross']=0.0

    # 量比(当日量/过去20日均量的比值)
    bars['volume_ma20']=bars.groupby('symbol')['volume'].transform(lambda x:compute_sma(x,20))
    bars['volume_breakout_ratio']=np.where(bars['volume_ma20']>0,
        bars['volume']/(bars['volume_ma20']+1),1.0)

    # 量缩程度
    bars['volume_ma5']=bars.groupby('symbol')['volume'].transform(lambda x:compute_sma(x,5))
    bars['volume_contraction']=np.where(bars['volume_ma20']>0,
        bars['volume_ma5']/(bars['volume_ma20']+1),1.0)

    # 均线粘合度(均线间的标准差/均值)
    ma_cols=['ma5','ma20','ma60']
    bars['ma_mean']=bars[ma_cols].mean(axis=1)
    bars['ma_std']=bars[ma_cols].std(axis=1)
    bars['ma_convergence']=np.where(bars['ma_mean']>0,
        bars['ma_std']/(bars['ma_mean']+1e-10),1.0)

    # 筹码集中度(波动率/价格的倒数,衡量价格稳定性)
    bars['price_volatility']=bars.groupby('symbol')['close'].transform(
        lambda x:x.rolling(20).std())
    bars['chip_concentration']=np.where(bars['price_volatility']>0,
        1.0/(1.0+bars['price_volatility']/bars['close']),0.0)
    # 范围0~1, 值越大筹码越集中

    # 20日均线角度
    bars['ma_angle_20']=bars.groupby('symbol')['ma20'].transform(
        lambda x:compute_ma_angle(x,5))

    # 准备写入: 只保留需要的列
    save_cols=['date','symbol']+FACTOR_NAMES
    save_df=bars[save_cols].copy()

    # 处理inf
    for c in FACTOR_NAMES:
        if c in save_df.columns:
            save_df[c]=save_df[c].replace([np.inf,-np.inf],np.nan)

    # 仅保留有至少一个因子有值的行
    save_df=save_df.dropna(subset=FACTOR_NAMES,how='all')
    logger.info(f"准备写入: {save_df.shape}, 日期范围 {save_df['date'].min()}~{save_df['date'].max()}")

    # 写入数据库
    db.ensure_factor_columns(FACTOR_NAMES)
    db.save_factors(save_df)
    logger.info("写入完成!")

    # 验证
    verify=db.get_factors(factor_names=['ma5','ma20'],start_date='2024-01-02',end_date='2024-01-10')
    if verify is not None and not verify.empty:
        logger.info(f"验证: {verify.shape}, ma5={verify['ma5'].iloc[0]:.2f}, ma20={verify['ma20'].iloc[0]:.2f}")

if __name__=="__main__":
    main()
