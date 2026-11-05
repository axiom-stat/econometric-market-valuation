# Econometric Modeling & OLS Regression Suite
### Applied Market Valuation, Multicollinearity Mitigation & Causal Inference

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](#)
[![Statsmodels](https://img.shields.io/badge/Statsmodels-OLS%20%26%20WLS-green)](#)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Wrangling-darkblue?logo=pandas&logoColor=white)](#)
[![License](https://img.shields.io/badge/License-MIT-purple)](#)

[**English Version**](#-english-version) • [**Русская версия**](#-русская-версия)

</div>

---

<details open>
<summary><b>🇺🇸 English Version (Click to collapse)</b></summary>

## Executive Summary
This repository contains end-to-end econometric research focused on explainable price modeling (Hedonic Pricing Models) across two distinct high-stakes domains:
1. **Secondary Real Estate Automated Valuation (AVM)** on a custom 12,500-record dataset, highlighting structural multicollinearity diagnosis via Variance Inflation Factor (VIF) and ratio-based feature engineering
2. **Automotive Pricing & Brand Equity Evaluation** investigating whether car brands command genuine price premiums after conditioning on engine and chassis specifications

Unlike black-box tree ensembles, these Ordinary Least Squares (OLS) specifications deliver exact marginal effects, standard errors, 95% confidence intervals, and hypothesis tests ($p$-values) essential for fintech risk scoring, automated collateral valuation, and pricing strategy

---

## Core Projects

### 1. Real Estate Automated Valuation Model (AVM)
- **Notebook:** [`real_estate_valuation_ols.ipynb`](./real_estate_valuation_ols.ipynb)
- **Dataset:** [`real_estate_dataset.csv`](./real_estate_dataset.csv) (12,500 observations, 13 features)

#### Analytical Challenges & Engineering
- **Outlier Curation:** Screened extreme luxury penthouses ($>28.5\text{M}$ RUB, top 5% quantile) to shield OLS estimators from excessive leverage points and variance inflation
- **VIF Diagnostic & Collinearity Resolution:** Identified severe collinearity between `total_area`, `living_area`, and `kitchen_area` ($\text{VIF} > 28$). Deconstructed spatial constraints by engineering scale-invariant ratios (`living_percent`, `kitchen_percent`), bringing all VIF scores below critical thresholds
- **Baseline vs. Multivariate Evolution:**
  - Simple Area OLS ($R^2 = 0.436$, $\text{AIC} = 62,370$): established baseline marginal price of $174,700$ RUB per square meter
  - Full Production Model ($R^2_{\text{adj}} = 0.886$, $\text{AIC} = 45,110$): AIC dropped by over $17,000$ points, confirming superior model parsimony

#### Key Econometric Insights
- **Transit Elasticity:** Each walking minute to the metro reduces apartment valuation by $94,900$ RUB ($p < 0.001$)
- **Floor Penalties:** Ground floor apartments suffer a statistically significant discount of $731,000$ RUB ($p < 0.001$), while top floor units see a milder $417,000$ RUB penalty
- **Location Premium:** Relative to the baseline Eastern district, Central district properties carry an average premium of $+8.95\text{M}$ RUB, followed by the Western district at $+4.04\text{M}$ RUB
- **Renovation Impact:** Designer finishes add an average of $+4.73\text{M}$ RUB, while European-standard renovation yields $+2.96\text{M}$ RUB over unrenovated baseline units

---

### 2. Automotive Hedonic Pricing & Brand Equity Audit
- **Notebook:** [`car_pricing_ols.ipynb`](./car_pricing_ols.ipynb)
- **Dataset:** [`cars.csv`](./cars.csv) (205 observations, 26 specifications)

#### Analytical Findings
- **Single-Factor Baseline:** Horsepower alone accounts for $65.3\%$ of price variance ($R^2 = 0.653$), with each horsepower yielding $+\$163.26$ ($p < 0.001$)
- **The Brand Equity Question:** Comparing a 48-feature model (with brand dummies) against a 26-feature specification (pure engineering metrics):
  - Excluding brand dummies reduced adjusted $R^2$ marginally from $0.948$ to $0.901$ ($< 5\%$ performance loss)
  - Over $85\%$ of brand indicator coefficients were statistically indistinguishable from zero ($p > 0.05$)
  - **Verdict:** Selected the brand-agnostic model under Occam's Razor, avoiding rank deficiency and multi-collinear noise while preserving robust physical interpretability

---

## Technical Stack & Methods
- **Languages & Frameworks:** Python 3.10+, Statsmodels (`sm.OLS`, `smf.ols`), Pandas, NumPy, Seaborn, Matplotlib
- **Econometric Tooling:**
  - One-Hot Encoding with Dummy Variable Trap mitigation (`drop_first=True`)
  - Variance Inflation Factor (`multicollinearity.vif`)
  - Information Criteria comparison ($\Delta\text{AIC}$, $\Delta\text{BIC}$)
  - Patsy syntax with categorical grouping `C(...)` and string quotation `Q(...)`

---

## Project Structure
```text
├── README.md                          # Interactive bilingual project documentation
├── real_estate_valuation_ols.ipynb    # Real estate AVM, VIF auditing & OLS pipeline
├── real_estate_dataset.csv            # 12,500 real estate listings
├── car_pricing_ols.ipynb              # Automotive hedonic pricing & brand premium model
└── cars.csv                           # Automotive technical and pricing dataset
```

---

## How to Run
```bash
git clone https://github.com/axiom-stat/Linear-model.git
cd Linear-model
pip install pandas numpy statsmodels seaborn matplotlib
jupyter lab
```

</details>

---

<details open>
<summary><b>🇷🇺 Русская версия (Нажмите, чтобы свернуть)</b></summary>

## О проекте
В репозитории собраны сквозные эконометрические исследования, посвященные прикладному моделированию рыночной стоимости на основе метода наименьших квадратов (OLS):
1. **Автоматическая оценка вторичной недвижимости (AVM)** на реалистичной выборке из 12 500 квартир: глубокий аудит мультиколлинеарности через VIF, переход к относительным долям площадей и расчет чистых рыночных надбавок
2. **Моделирование стоимости автомобилей и аудит силы бренда (Brand Equity)**: проверка гипотезы о наличии реальной ценовой премии за марку после контроля технических параметров мотора и кузова

В отличие от алгоритмов машинного обучения со сложными ансамблями, классический эконометрический подход дает полную прозрачность: расчет предельных эффектов, стандартных ошибок, доверительных интервалов и проверку статистической значимости факторов ($p$-value)

---

## Исследуемые кейсы

### 1. Автоматическая оценка недвижимости (AVM)
- **Ноутбук:** [`real_estate_valuation_ols.ipynb`](./real_estate_valuation_ols.ipynb)
- **Данные:** [`real_estate_dataset.csv`](./real_estate_dataset.csv) (12 500 строк, 13 признаков)

#### Этапы работы и аналитические решения
- **Фильтрация выбросов:** Срез верхних 5% выборки (пентхаусы дороже 28.5 млн руб) позволил стабилизировать оценки OLS и сосредоточиться на массовом сегменте и комфорт-классе
- **Диагностика мультиколлинеарности (VIF):** Выявлена критическая зависимость между общей, жилой площадью и кухней ($\text{VIF} > 28$). Перевод площадей в относительные доли (`living_percent`, `kitchen_percent`) вернул показатели VIF в норму
- **Сравнение спецификаций:**
  - Базовая модель площади ($R^2 = 0.436$, $\text{AIC} = 62 370$): базовая оценка одного квадратного метра составила $174 700$ рублей
  - Полнофакторная модель ($R^2_{\text{adj}} = 0.886$, $\text{AIC} = 45 110$): падение критерия Акаике более чем на $17 000$ пунктов подтвердило высокую точность и устойчивость модели

#### Главные продуктовые инсайты
- **Транспортная доступность:** Каждая минута пешком до метро снижает стоимость квартиры в среднем на $95 000$ рублей ($p < 0.001$)
- **Фактор крайних этажей:** Первый этаж дисконтируется рынком на $731 000$ рублей, последний этаж продается со скидкой $417 000$ рублей относительно средних этажей
- **Локационная премия:** Относительно базового Восточного района квартиры в Центре дороже в среднем на $+8.95$ млн рублей, а в Западном районе на $+4.04$ млн рублей
- **Вклад ремонта:** Дизайнерская отделка добавляет к стоимости квартиры $+4.73$ млн рублей, а евроремонт $+2.96$ млн рублей по сравнению с жильем без ремонта

---

### 2. Ценообразование автомобилей и оценка фактора бренда
- **Ноутбук:** [`car_pricing_ols.ipynb`](./car_pricing_ols.ipynb)
- **Данные:** [`cars.csv`](./cars.csv) (205 строк, 26 характеристик)

#### Результаты моделирования
- **Однофакторный базис:** Мощность мотора объясняет $65.3\%$ разброса цен ($R^2 = 0.653$), прибавляя в среднем по $+\$163.26$ за каждую лошадиную силу
- **Премия за бренд:** Сравнение полной модели с марками машин (48 признаков) и модели на чистых физических спецификациях (26 признаков):
  - Исключение марок снизило скорректированный $R^2$ всего с $0.948$ до $0.901$ (потеря качества менее 5%)
  - Более $85\%$ коэффициентов марок оказались статистически незначимыми ($p > 0.05$) на фоне мощности, габаритов и веса
  - **Вердикт:** По принципу парсимонии (Бритва Оккама) выбрана модель без брендов, так как она проще, стабильнее и не подвержена рискам переобучения

---

## Стек технологий
- **Инструменты:** Python 3.10+, Statsmodels (`sm.OLS`, `smf.ols`), Pandas, NumPy, Seaborn, Matplotlib
- **Методы анализа:**
  - One-Hot Encoding с исключением базовой категории (`drop_first=True`) во избежание ловушки фиктивных переменных
  - Расчет коэффициента вздутия дисперсии (`VIF`)
  - Сравнение информационных критериев ($\text{AIC}$, $\text{BIC}$)
  - Формульный синтаксис Patsy с изоляцией русских символов `Q(...)` и категоризацией `C(...)`

---

## Инструкция по запуску
```bash
git clone https://github.com/axiom-stat/Linear-model.git
cd Linear-model
pip install pandas numpy statsmodels seaborn matplotlib
jupyter lab
```

</details>
