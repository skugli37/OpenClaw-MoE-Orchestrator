#!/usr/bin/env python3
"""
Production Orchestrator sa STVARNIM tržišnim podacima
Preuzima BTC/ETH/SOL sa Yahoo Finance i detektuje anomalije
"""

import sys
import asyncio
import torch
import numpy as np
import pandas as pd
import yfinance as yf
import logging
from datetime import datetime, timedelta

sys.path.insert(0, '/home/ubuntu/OPENCLAW_MOE_PROJECT/scripts')

from production_agent_orchestrator import ProductionOrchestrator
from beyond_sota_architecture import BeyondSOTAAnomalyDetector

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def fetch_real_market_data(symbol='BTC-USD', days=30):
    """Preuzmi realne podatke sa Yahoo Finance"""
    logger.info(f"📊 Preuzimanje {days}-dnevnih podataka za {symbol}...")
    
    try:
        df = yf.download(symbol, period=f"{days}d", interval="1h", progress=False)
        logger.info(f"   ✅ Preuzeto {len(df)} svečica")
        return df
    except Exception as e:
        logger.error(f"   ❌ Greška: {e}")
        return None

def prepare_market_tensor(df):
    """Pripremi PyTorch tensor iz OHLCV podataka"""
    
    # Osnovna svojstva
    features_dict = {}
    features_dict['close'] = df['Close'].values.flatten()
    features_dict['volume'] = df['Volume'].values.flatten()
    features_dict['high'] = df['High'].values.flatten()
    features_dict['low'] = df['Low'].values.flatten()
    features_dict['open'] = df['Open'].values.flatten()
    
    # Tehnijski indikatori
    df_copy = df.copy()
    df_copy['returns'] = df_copy['Close'].pct_change().fillna(0)
    df_copy['volatility'] = df_copy['returns'].rolling(window=5).std().fillna(0)
    
    features_dict['returns'] = df_copy['returns'].values.flatten()
    features_dict['volatility'] = df_copy['volatility'].values.flatten()
    
    features = pd.DataFrame(features_dict)
    
    # Normalizuj na [-1, 1]
    normalized = features.copy()
    for col in normalized.columns:
        min_val = normalized[col].min()
        max_val = normalized[col].max()
        if max_val > min_val:
            normalized[col] = 2 * (normalized[col] - min_val) / (max_val - min_val) - 1
        else:
            normalized[col] = 0
    
    # Konvertuj u tensor - uzmi poslednje 64 vremenske korake
    data_array = normalized.values
    if len(data_array) >= 64:
        data_array = data_array[-64:]
    
    tensor = torch.from_numpy(data_array).float().unsqueeze(0)  # (1, 64, 7)
    
    logger.info(f"   ✅ Tensor shape: {tensor.shape}")
    return tensor

async def main():
    """Pokreni production orchestrator sa stvarnim podacima"""
    
    print("\n" + "="*70)
    print("🚀 PRODUCTION ORCHESTRATOR - STVARNI TRŽIŠNI PODACI")
    print("="*70)
    
    # Preuzmi podatke
    btc_data = fetch_real_market_data('BTC-USD', days=30)
    if btc_data is None:
        logger.error("Nije moguće preuzeti podatke!")
        return
    
    # Pripremi tensor
    logger.info("🔧 Priprema market tensora...")
    market_tensor = prepare_market_tensor(btc_data)
    
    # Inicijalizuj detektor
    logger.info("🤖 Inicijalizacija Beyond-SOTA detektora...")
    detector = BeyondSOTAAnomalyDetector(
        input_dim=7,
        hidden_dim=256,
        num_heads=8,
        device='cpu'
    )
    
    # Inicijalizuj orchestrator
    logger.info("🎯 Inicijalizacija production orchestrator-a...")
    orchestrator = ProductionOrchestrator(detector)
    
    # Pokreni orchestration
    logger.info("⚡ Pokretanje paralelne orkestracije sa 3 agenta...")
    print()
    
    result = await orchestrator.orchestrate(
        market_data=market_tensor,
        assets=['BTC', 'ETH', 'SOL']
    )
    
    # Prikaži rezultate
    print("\n" + "="*70)
    print("📊 REZULTATI ORKESTRACIJE")
    print("="*70)
    
    print(f"\n🔹 Market Expert Agent:")
    print(f"   Status: {result.market_expert.status}")
    print(f"   Anomaly Score: {result.market_expert.data.get('anomaly_score', 'N/A'):.4f}")
    print(f"   Anomaly Detected: {result.market_expert.data.get('is_anomaly', False)}")
    print(f"   Execution Time: {result.market_expert.execution_time:.3f}s")
    
    print(f"\n🔹 News Oracle Agent:")
    print(f"   Status: {result.news_oracle.status}")
    news_items = result.news_oracle.data.get('news_items', [])
    print(f"   News Items Found: {len(news_items)}")
    print(f"   Execution Time: {result.news_oracle.execution_time:.3f}s")
    
    print(f"\n🔹 Risk Manager Agent:")
    print(f"   Status: {result.risk_manager.status}")
    print(f"   Risk Score: {result.risk_manager.data.get('final_risk_score', 'N/A'):.2f}/10")
    print(f"   Execution Time: {result.risk_manager.execution_time:.3f}s")
    
    print(f"\n🎯 FINALNI REZULTATI:")
    print(f"   Final Risk Score: {result.final_risk_score:.2f}/10")
    print(f"   Anomaly Detected: {result.final_anomaly_detected}")
    print(f"   Total Execution Time: {result.synthesis['total_execution_time']:.2f}s")
    
    # Interpretacija
    print(f"\n💡 INTERPRETACIJA:")
    if result.final_risk_score < 3:
        print(f"   ✅ NIZAK RIZIK - Tržište je stabilno")
    elif result.final_risk_score < 6:
        print(f"   ⚠️  UMEREN RIZIK - Normalna tržišna aktivnost")
    else:
        print(f"   🚨 VISOK RIZIK - Moguća anomalija ili volatilnost")
    
    if result.final_anomaly_detected:
        print(f"   🔴 ANOMALIJA DETEKTOVANA - Preporučuje se opreznost")
    else:
        print(f"   🟢 NEMA ANOMALIJE - Sve je u redu")
    
    print("\n" + "="*70)
    print("✅ ORCHESTRATION COMPLETED")
    print("="*70 + "\n")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Greška: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
