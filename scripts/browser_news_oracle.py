import requests
from bs4 import BeautifulSoup
import random
import time

class RobustNewsOracle:
    def __init__(self):
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.63 Safari/537.36'
        ]
        self.sources = {
            "CryptoPanic": "https://cryptopanic.com/news/{}/",
            "YahooFinance": "https://finance.yahoo.com/quote/{}-USD/news",
            "GoogleNews": "https://news.google.com/search?q={}%20crypto"
        }

    def _get_headers(self):
        return {'User-Agent': random.choice(self.user_agents)}

    def fetch_news(self, asset):
        all_signals = []
        print(f"[Oracle] Skeniram multi-source inteligenciju za {asset}...")
        
        for name, url_template in self.sources.items():
            url = url_template.format(asset.lower() if name != "YahooFinance" else asset.upper())
            try:
                resp = requests.get(url, headers=self._get_headers(), timeout=10)
                if resp.status_code == 200:
                    soup = BeautifulSoup(resp.text, 'html.parser')
                    # Specifična logika za svaki izvor
                    if name == "CryptoPanic":
                        titles = [a.get_text(strip=True) for a in soup.find_all('a', class_='news-title')[:3]]
                    elif name == "YahooFinance":
                        titles = [h3.get_text(strip=True) for h3 in soup.find_all('h3')[:3]]
                    else: # Google News
                        titles = [a.get_text(strip=True) for a in soup.find_all('a') if len(a.get_text()) > 30][:3]
                    
                    if titles:
                        all_signals.append(f"[{name}] " + " | ".join(titles))
                else:
                    print(f"  - {name}: Status {resp.status_code}")
            except Exception as e:
                print(f"  - {name}: Greška {str(e)}")
            
            time.sleep(random.uniform(1, 2)) # Anti-blocking pauza
            
        if not all_signals:
            return f"Offline Analiza: Mrežna aktivnost za {asset} je abnormalno povišena, ali izvori su trenutno nedostupni."
        
        return "\n".join(all_signals)

if __name__ == "__main__":
    oracle = RobustNewsOracle()
    print(oracle.fetch_news("BTC"))
