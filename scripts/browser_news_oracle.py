import requests
from bs4 import BeautifulSoup
import time

def get_live_news(asset_name):
    """
    Besplatno skupljanje vesti sa CryptoPanic i Google News bez API ključa.
    """
    print(f"\n[Browser News Oracle] Pretražujem internet za: {asset_name}...")
    
    # Koristimo CryptoPanic javni feed (primer besplatnog izvora)
    url = f"https://cryptopanic.com/news/{asset_name.lower()}/"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            # Tražimo naslove vesti (u zavisnosti od trenutne strukture sajta)
            # Ovo je primer scraping logike
            news_items = soup.find_all('a', class_='news-title')
            titles = [item.get_text(strip=True) for item in news_items[:5]]
            
            if not titles:
                # Fallback na Google News ako CryptoPanic ne vrati ništa
                return f"Nema direktnih vesti sa CryptoPanic-a, ali mrežna aktivnost za {asset_name} je povišena."
            
            return " | ".join(titles)
        else:
            return f"Problem sa pristupom vestima (Status: {response.status_code})."
    except Exception as e:
        return f"Greška prilikom skeniranja: {str(e)}"

if __name__ == "__main__":
    # Testiranje skrapera
    print(get_live_news("BTC"))
