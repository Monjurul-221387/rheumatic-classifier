# Data Directory

## Dataset

**Name:** Rheumatic and Autoimmune Disease Dataset  
**Source:** Mahdi et al. (2025), *Data in Brief*, Elsevier  
**DOI:** [10.1016/j.dib.2025.111623](https://doi.org/10.1016/j.dib.2025.111623)  
**Repository:** [Harvard Dataverse](https://doi.org/10.7910/DVN/VM4OR3)

---

## Download Instructions

1. Go to: https://doi.org/10.7910/DVN/VM4OR3
2. Click **"Access Dataset"**
3. Download the file: `Rheumatic and Autoimmune Disease Dataset.csv`
4. Place it in this `data/` folder

Final path should be:
```
data/Rheumatic and Autoimmune Disease Dataset.csv
```

---

## Dataset Summary

| Property | Value |
|---|---|
| Patients | 12,085 |
| Features | 15 (2 demographic + 13 serological) |
| Classes | 7 disease categories |
| Collection period | 2019–2024 |
| Collection sites | 4 medical institutions, Iraq |

## Class Distribution

| Disease | Count | % |
|---|---|---|
| Rheumatoid Arthritis | 2,848 | 23.6% |
| Ankylosing Spondylitis | 2,127 | 17.6% |
| Sjögren's Syndrome | 1,852 | 15.3% |
| Psoriatic Arthritis | 1,783 | 14.8% |
| Normal | 1,604 | 13.3% |
| Systemic Lupus Erythematosus | 1,355 | 11.2% |
| Reactive Arthritis | 516 | 4.3% |

## Features

| Feature | Type | Missing |
|---|---|---|
| Age | Numeric | 0% |
| Gender | Categorical | 0% |
| ESR | Numeric | 9.0% |
| CRP | Numeric | 20.0% |
| RF | Numeric | 11.0% |
| Anti-CCP | Numeric | 27.0% |
| HLA-B27 | Binary | 16.0% |
| ANA | Binary | 31.0% |
| Anti-Ro | Binary | 24.0% |
| Anti-La | Binary | 25.0% |
| Anti-dsDNA | Binary | 39.0% |
| Anti-Sm | Binary | 43.0% |
| C3 | Numeric | 14.0% |
| C4 | Numeric | 17.0% |
| Disease | Target | 0% |

---

## Citation

```bibtex
@article{mahdi2025dataset,
  title   = {Diagnosis of rheumatic and autoimmune diseases dataset},
  author  = {Mahdi, Mohammed Fadhil and Jahani, Arezoo and Abd, Dhafar Hamed},
  journal = {Data in Brief},
  volume  = {60},
  pages   = {111623},
  year    = {2025},
  doi     = {10.1016/j.dib.2025.111623}
}
```

> **Note:** The dataset is not included in this repository.  
> Download it directly from Harvard Dataverse using the link above.
