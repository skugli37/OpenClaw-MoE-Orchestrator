#!/usr/bin/env python3
"""
STVARNI Production Orchestrator - Bez mock podataka
Koristi CoinGecko API za news + yfinance za market data
"""

import asyncio
import sys
import os
import torch
import numpy as np
import pandas as pd
import yfinance as yf
import logging
from datetime import datetime
from typing import Dict, List
import aiohttp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from beyond_sota_architecture import BeyondSOTAAnomalyDetector

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class RealNewsOracle:
    """STVARNI news oracle sa CoinGecko API"""
    
    def __init__(self):
        self.coingecko_url = "https://api.coingecko.com/api/v3"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    async def fetch_news(self, assets: List[str]) -> List[Dict]:
        """Preuzmi STVARNE news podatke"""
        logger.info(f"[NewsOracle] Preuzimanje STVARNIH news podataka za {assets}...")
        news = []
        
        try:
            async with aiohttp.ClientSession() as session:
                # CoinGecko market data
                url = f"{self.coingecko_url}/simple/price?ids=bitcoin,ethereum,solana&vs_currencies=usd&include_market_cap=true&include_24hr_vol=true&include_24hr_change=true"
                
                async with session.get(url, headers=self.headers, timeout=10) as response:
                    if response.status != 200:
                        raise Exception(f"CoinGecko API vraća {response.status}")
                    
                    data = await response.json()
                    
                    coin_map = {'bitcoin': 'BTC', 'ethereum': 'ETH', 'solana': 'SOL'}
                    
                    for coin_id, symbol in coin_map.items():
                        if coin_id in data:
                            coin_data = data[coin_id]
                            price_change = coin_data.get('usd_24h_change', 0)
                            
                            news.append({
                                'source': 'CoinGecko',
                                'asset': symbol,
                                'title': f'{symbol} - 24h Change: {price_change:+.2f}%',
                                'sentiment': 0.7 if price_change > 0 else 0.3,
                                'price_usd': coin_data.get('usd', 0),
                                'price_change_24h': price_change
                            })
                    
                    logger.info(f"[NewsOracle] ✅ Preuzeto {len(news)} STVARNIH news stavki")
        
        except Exception as e:
            logger.error(f"[NewsOracle] ❌ GREŠKA: {e}")
            raise
        
        return news


class RealMarketDataLoader:
    """STVARNI market data loader sa yfinance"""
    
    def fetch_market_data(self, symbol: str = 'BTC-USD', days: int = 30) -> torch.Tensor:
        """Preuzmi STVARNE market podatke sa yfinance"""
        logger.info(f"[MarketData] Preuzimanje STVARNIH {days}-dnevnih podataka za {symbol}...")
        
        try:
            df = yf.download(symbol, period=f"{days}d", interval="1h", progress=False)
            
            if len(df) < 64:
                raise Exception(f"Nedovoljno podataka: {len(df)} < 64")
            
            logger.info(f"[MarketData] ✅ Preuzeto {len(df)} svečica")
            
            # Pripremi features
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
            
            # Normalizuj
            normalized = features.copy()
            for col in normalized.columns:
                min_val = normalized[col].min()
                max_val = normalized[col].max()
                if max_val > min_val:
                    normalized[col] = 2 * (normalized[col] - min_val) / (max_val - min_val) - 1
                else:
                    normalized[col] = 0
            
            # Konvertuj u tensor - poslednje 64 koraka
            data_array = normalized.values
            if len(data_array) >= 64:
                data_array = data_array[-64:]
            
            tensor = torch.from_numpy(data_array).float().unsqueeze(0)
            
            logger.info(f"[MarketData] ✅ Tensor shape: {tensor.shape}")
            return tensor
        
        except Exception as e:
            logger.error(f"[MarketData] ❌ GREŠKA: {e}")
            raise


async def main():
    """Pokreni STVARNI production orchestrator"""
    
    print("\n" + "="*70)
    print("🚀 PRODUCTION ORCHESTRATOR - SAMO STVARNI PODACI")
    print("="*70)
    
    try:
        # Preuzmi STVARNE market podatke
        market_loader = RealMarketDataLoader()
        market_tensor = market_loader.fetch_market_data('BTC-USD', days=30)
        
        # Preuzmi STVARNE news podatke
        news_oracle = RealNewsOracle()
        news = await news_oracle.fetch_news(['BTC', 'ETH', 'SOL'])
        
        # Inicijalizuj detektor
        logger.info("[Detector] Inicijalizacija Beyond-SOTA detektora...")
        detector = BeyondSOTAAnomalyDetector(
            input_dim=7,
            hidden_dim=256,
            num_heads=8,
            device='cpu'
        )
        
        # Pokreni detektor
        logger.info("[Detector] Pokretanje anomaly detection...")
        result = detector.detect(market_tensor)
        
        # Kalkuliši risk score od STVARNIH podataka
        market_anomaly_score = result.anomaly_score
        news_sentiment = np.mean([n['sentiment'] for n in news])
        risk_score = (market_anomaly_score * 10) * (1 - news_sentiment)
        
        print("\n" + "="*70)
        print("📊 REZULTATI - SAMO STVARNI PODACI")
        print("="*70)
        
        print(f"\n🔹 MARKET DATA:")
        print(f"   Source: Yahoo Finance (BTC-USD, 30 dana)")
        print(f"   Svečice: 64 (poslednje)")
        print(f"   Anomaly Score: {market_anomaly_score:.4f}")
        print(f"   Is Anomaly: {result.is_anomaly}")
        
        print(f"\n🔹 NEWS DATA:")
        print(f"   Source: CoinGecko API")
        print(f"   News Items: {len(news)}")
        print(f"   Average Sentiment: {news_sentiment:.2f}")
        for i, n in enumerate(news, 1):
            print(f"   {i}. {n['asset']}: {n['title']} (Sentiment: {n['sentiment']:.2f})")
        
        print(f"\n🎯 FINALNI REZULTATI:")
        print(f"   Risk Score: {risk_score:.2f}/10")
        print(f"   Anomaly Detected: {result.is_anomaly}")
        print(f"   Timestamp: {datetime.now().isoformat()}")
        
        print("\n" + "="*70)
        print("✅ ORCHESTRATION COMPLETED - SAMO STVARNI PODACI")
        print("="*70 + "\n")
    
    except Exception as e:
        logger.error(f"❌ KRITIČNA GREŠKA: {e}")
        print(f"\n❌ NEUSPEŠNO: {e}")
        exit(1)


if __name__ == "__main__":
    asyncio.run(main())
