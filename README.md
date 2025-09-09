# SF Crime Aware Navigator

Streamlit app that predicts **route risk** in San Francisco and shows **live incident streams**.

---

## 🚀 How It Works
- Enter start & end addresses → app fetches route and predicts risk score.
- Risk model (pre-trained) estimates safety based on time, day, and location.
- Live incidents (Kafka stream) appear in sidebar in real time.
- Includes demo producers to simulate crime events.

---

## 📂 Files
- `app2.py` — Streamlit UI  
- `utils.py` — routing & plotting helpers  
- `modelling.py` — preprocessing/model code (not required at runtime)  
- `producer.py` / `randomiser.py` — Kafka event generators  

---

## ▶️ Run Locally
```bash
pip install -r requirements.txt
streamlit run app2.py
