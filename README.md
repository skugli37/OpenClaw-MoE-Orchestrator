# 🦞 OpenClaw MoE Orchestrator Project

Ovaj repozitorijum sadrži kompletan sistem za autonomnu analizu tržišta i detekciju anomalija, razvijen u saradnji sa Manus AI agentom.

## Sadržaj
- **scripts/**: Python skripte za DeepSpeed MoE trening, prikupljanje podataka i vizuelizaciju.
- **configs/**: DeepSpeed ZeRO-2/3 konfiguracije optimizovane za CPU/GPU.
- **skills/**: Re-usable Manus veština za orkestraciju OpenClaw agenata.
- **docs/**: Izveštaji o misijama i tehnička dokumentacija.

## Kako koristiti
1. Instalirajte zavisnosti koristeći `uv`:
   ```bash
   sudo pip3 install uv && sudo uv pip install --system torch deepspeed transformers yfinance pandas matplotlib
   ```
2. Pokrenite autonomnu misiju:
   ```bash
   python3 scripts/autonomous_mission.py
   ```

---
*Developed in UNRESTRICTED Research Mode.*
