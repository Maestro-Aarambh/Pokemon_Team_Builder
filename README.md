# 🧩 Pokémon_Team_Builder  
**Machine Learning–powered, data-driven Pokémon Team Builder** that uses **PCA**, **K-Means clustering**, **PokéAPI integration**, and **type-effectiveness analysis** to create balanced, stat-based, and intelligent Pokémon teams.

---

## 📖 Overview  
This project combines **data science**, **machine learning**, and **Pokémon type logic** to automatically build balanced Pokémon teams.  
Using combat stats such as **HP**, **Attack**, **Defense**, and **Speed**, the system performs **dimensionality reduction (PCA)** and **unsupervised clustering (K-Means)** to identify Pokémon with similar strengths.  

When you enter a Pokémon type (like *Fire*, *Water*, or *Electric*), the program:  
- Selects Pokémon of that type from **different clusters** for stat diversity  
- Calculates their **true weaknesses** based on both types  
- Fetches their **official sprites** from the PokéAPI  
- Displays your **custom team visually**

---

## ✨ Features  
- ✅ Scales and preprocesses Pokémon stats  
- ✅ Reduces dimensions using **PCA (Principal Component Analysis)**  
- ✅ Automatically finds the **optimal number of clusters** with silhouette score  
- ✅ Groups Pokémon using **K-Means clustering**  
- ✅ Integrates **PokéAPI** to display live Pokémon sprites  
- ✅ Computes **dual-type weaknesses and resistances**  
- ✅ Generates **balanced, stat-diverse Pokémon teams**

---

## 🧠 Tech Stack  

| Category | Libraries / Tools |
|-----------|-------------------|
| **Data Handling** | pandas, numpy |
| **Machine Learning** | scikit-learn (StandardScaler, PCA, KMeans, silhouette_score) |
| **Visualization** | matplotlib, Pillow |
| **API** | requests, PokéAPI |
| **Others** | io.BytesIO |

---

## 🚀 How It Works  
1. Load Pokémon dataset containing base stats and types.  
2. Standardize data using **StandardScaler**.  
3. Apply **PCA** to reduce dimensionality for better visualization and clustering.  
4. Use **K-Means** to cluster Pokémon with similar attributes.  
5. Automatically determine the **best cluster count** using silhouette scores.  
6. When a Pokémon type is entered, pick Pokémon from **different clusters** for balance.  
7. Fetch and display Pokémon sprites from **PokéAPI**.  

---


## 🧩 Future Improvements  
- Add **Gen 9 Pokémon data**  
- Implement **team synergy scoring**  
- Introduce **move-set recommendations**  
- Build a **web-based interface** using Streamlit or Flask  

---

## 📜 License  
This project is licensed under the **MIT License** — free to use and modify with credit.

---
