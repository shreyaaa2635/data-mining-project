# 🌍 Global Weather Pattern Analysis using STING

## 📌 Overview

This project implements a **grid-based clustering approach (STING - Statistical Information Grid)** to analyze large-scale climate data efficiently. The system divides geographical space into hierarchical grid cells and computes statistical summaries to identify regional weather patterns without scanning the entire dataset repeatedly.

## 🎯 Objective

* To perform efficient spatial data analysis using STING
* To identify climate patterns using hierarchical grid structures
* To reduce computational cost using aggregated statistical information

## 📂 Dataset

* **Name:** Global Historical Climatology Network (GHCN)
* **Source:** NOAA
* **Format:** Fixed-width text file (`.txt`)
* **Attributes Used:** Station ID, Latitude, Longitude

Dataset Link:
https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt

## ⚙️ Methodology

1. Load dataset using fixed-width parsing
2. Simulate temperature based on latitude
3. Create hierarchical grid structure
4. Compute statistical summaries
5. Store results in grid hierarchy
6. Visualize using scatter plots

## 🧠 Algorithm

* STING (Statistical Information Grid)

## 🛠️ Tools

* Google Colab
* Python
* Pandas, NumPy
* Matplotlib

## 📊 Results
- Generated multi-resolution grid-based visualizations (30° to 5°) showing temperature distribution  
- Observed that higher resolutions provide finer regional climate patterns  
- Identified temperature variation with latitude (higher near equator, lower towards poles)  
- Demonstrated efficient spatial clustering using STING without scanning raw data repeatedly  

## 📸 Output

* sting_30deg.png
* sting_25deg.png
* sting_20deg.png
* sting_10deg.png
* sting_5deg.png

## 🚀 How to Run

1. Download dataset
2. Run the Python script
3. View generated plots

## 📘 Learning

* Grid-based clustering
* Spatial data analysis
* Data visualization

## 👩‍💻 Authors

* Suhani Verma
* Bhargavi Mahajan
* Arya Jha
* Shreya Srivastava
