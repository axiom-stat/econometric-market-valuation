# Real Estate Automated Valuation Model (AVM)
### Hedonic Pricing, Multicollinearity Mitigation & Econometric Inference

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

## Project Overview
This repository contains an end-to-end econometric research and production-oriented Automated Valuation Model (AVM) for urban secondary real estate, trained on 12,500 property listings
The model quantifies fundamental market drivers, isolates the price elasticity of transit proximity and floor placement, and guarantees statistical stability through rigorous Variance Inflation Factor (VIF) collinearity resolution

Unlike black-box machine learning algorithms, this econometric specification delivers exact marginal unit economics, standard errors, 95% confidence intervals, and hypothesis tests ($p$-values) necessary for transparent mortgage underwriting, collateral assessment, and automated algorithmic property acquisitions

---

## Technical Highlights & Engineering Decisions

### 1. Robust Outlier Filtering
Raw transaction listings frequently suffer from extreme luxury penthouses and high-leverage data anomalies
Evaluating the empirical price distribution revealed a heavy right-tail leverage effect
By capping the model space at the 95th percentile ($28.5\text{M}$ RUB), the training pipeline isolates mass-market and comfort-class urban housing, preventing OLS variance inflation and coefficient instability

### 2. Resolving Structural Multicollinearity via VIF
A naive regression including total area, living area, and kitchen area simultaneously causes extreme multicollinearity ($\text{VIF} > 28$) due to internal spatial accounting constraints
Rather than dropping sub-areas and losing valuable layout signals, the feature space was restructured into scale-invariant proportions:
$$\text{living\_percent} = \frac{\text{living\_area}}{\text{total\_area}} \times 100, \quad \text{kitchen\_percent} = \frac{\text{kitchen\_area}}{\text{total\_area}} \times 100$$
This structural reformulation brought all VIF scores below critical thresholds, eliminating rank deficiency while preserving complete spatial layout information

### 3. Model Progression & Parsimony Metrics
- **Baseline Simple OLS ($R^2 = 0.436$, $\text{AIC} = 62,370$):** Single-variable model establishing the unadjusted baseline marginal cost of $174,700$ RUB per square meter
- **Full Production Model ($R^2_{\text{adj}} = 0.886$, $\text{AIC} = 45,110$):** Multi-factor specification incorporating location, floor rank, transit, parking, and finish tiers
AIC dropped by over $17,000$ points, demonstrating massive explanatory gain without parameter bloat

---

## Core Valuation Insights

| Factor / Characteristic | Marginal Impact (RUB) | Statistical Significance | Economic Interpretation |
| :--- | :--- | :--- | :--- |
| **Square Meter (Base Area)** | $+185,700\text{ RUB / m}^2$ | $p < 0.001$ | Clean marginal valuation per square meter in the baseline Eastern district |
| **Walking Time to Metro** | $-94,900\text{ RUB / min}$ | $p < 0.001$ | Robust transit discount per additional minute of pedestrian access to subway |
| **Ground Floor Placement** | $-731,000\text{ RUB}$ | $p < 0.001$ | Statistically significant discount for ground floor privacy and security penalties |
| **Top Floor Placement** | $-417,000\text{ RUB}$ | $p < 0.001$ | Mild discount for top floor units relative to middle floors |
| **Central District Premium** | $+8,954,600\text{ RUB}$ | $p < 0.001$ | Pure location equity premium relative to the baseline Eastern district |
| **Western District Premium** | $+4,035,400\text{ RUB}$ | $p < 0.001$ | Prestigious suburban / western urban corridor valuation premium |
| **Suburban Discount** | $-2,275,200\text{ RUB}$ | $p < 0.001$ | Systematic discount for outer-ring metropolitan housing |
| **Designer Renovation** | $+4,725,300\text{ RUB}$ | $p < 0.001$ | Premium for luxury renovation over unrenovated condition |
| **Euro-standard Renovation**| $+2,960,100\text{ RUB}$ | $p < 0.001$ | Quality finishes premium over baseline shell state |
| **Dedicated Parking** | $+360,600\text{ RUB}$ | $p < 0.001$ | Added liquidity and value for dedicated parking access |

---

## Repository Structure
```text
├── README.md                          # Interactive bilingual documentation
├── real_estate_valuation_ols.ipynb    # Full analytical notebook (VIF audit, EDA, OLS)
└── real_estate_dataset.csv            # 12,500 real estate listings with urban features
```

---

## Quickstart
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
В репозитории представлено комплексное эконометрическое исследование и алгоритм автоматической оценки вторичной недвижимости (AVM - Automated Valuation Model), обученный на 12 500 объявлениях городского жилья
Модель количественно измеряет ключевые рыночные драйверы ценообразования, оценивает чувствительность цены к удаленности от метро и этажности, а также гарантирует статистическую устойчивость коэффициентов за счет устранения мультиколлинеарности через VIF

В отличие от непрозрачных моделей градиентного бустинга («черных ящиков»), эконометрический OLS-подход обеспечивает полную интерпретируемость: расчет чистых надбавок и скидок, стандартных ошибок, доверительных интервалов и проверку статистической значимости факторов ($p$-value) для скоринга ипотеки, андеррайтинга и срочного выкупа квартир

---

## Ключевые аналитические и инженерные решения

### 1. Очистка распределения от элитных выбросов
Реальные базы объявлений содержат единичные сверхдорогие пентхаусы, которые смещают оценки МНК
Анализ правого хвоста распределения показал необходимость ограничения выборки 95-м квантилем ($28.5$ млн рублей)
Это позволило сосредоточить модель на массовом сегменте и комфорт-классе, защитив регрессию от вздутия дисперсии ошибок

### 2. Устранение мультиколлинеарности площадей через VIF
Включение в модель общей, жилой площади и кухни одновременно приводило к критическому значению VIF ($> 28$), так как сумма помещений жестко ограничена общим метражом
Вместо выбрасывания данных архитектура признаков была реорганизована в относительные доли:
$$\text{living\_percent} = \frac{\text{living\_area}}{\text{total\_area}} \times 100, \quad \text{kitchen\_percent} = \frac{\text{kitchen\_area}}{\text{total\_area}} \times 100$$
Такая параметризация вернула все показатели VIF в безопасную зону и сохранила ценную информацию о планировке без риска вырождения матриц

### 3. Эволюция спецификаций и информационные критерии
- **Базовый OLS ($R^2 = 0.436$, $\text{AIC} = 62 370$):** Однофакторная модель площади зафиксировала средний рыночный ориентир в $174 700$ рублей за квадратный метр
- **Полнофакторный OLS ($R^2_{\text{adj}} = 0.886$, $\text{AIC} = 45 110$):** Множественная модель с учетом района, ремонта, этажности и транспорта
Падение критерия Акаике более чем на $17 000$ пунктов подтвердило высокую точность и устойчивость модели без избыточного усложнения

---

## Таблица коэффициентов и продуктовые инсайты

| Фактор / Характеристика | Предельный вклад (руб) | Статистическая значимость | Экономический смысл |
| :--- | :--- | :--- | :--- |
| **Квадратный метр (База)** | $+185 700\text{ руб / м}^2$ | $p < 0.001$ | Чистая стоимость метра при прочих равных в базовом Восточном районе |
| **Минута пешком до метро** | $-94 900\text{ руб / мин}$ | $p < 0.001$ | Снижение стоимости квартиры за каждую минуту удаления от станции метро |
| **Первый этаж** | $-731 000\text{ руб}$ | $p < 0.001$ | Рыночный дисконт за недостатки приватности и безопасность первого этажа |
| **Последний этаж** | $-417 000\text{ руб}$ | $p < 0.001$ | Дисконт за крайний верхний этаж относительно средних этажей дома |
| **Центральный район** | $+8 954 600\text{ руб}$ | $p < 0.001$ | Чистая премия за центральную локацию относительно базового Восточного района |
| **Западный район** | $+4 035 400\text{ руб}$ | $p < 0.001$ | Престижная надбавка за западное направление города |
| **Пригород** | $-2 275 200\text{ руб}$ | $p < 0.001$ | Дисконт за удаленность жилья за пределами основной черты города |
| **Дизайнерский ремонт** | $+4 725 300\text{ руб}$ | $p < 0.001$ | Добавленная стоимость эксклюзивной отделки относительно «без ремонта» |
| **Евроремонт** | $+2 960 100\text{ руб}$ | $p < 0.001$ | Добавленная стоимость качественного современного ремонта |
| **Подземный паркинг** | $+360 600\text{ руб}$ | $p < 0.001$ | Ликвидная прибавка за закрепленное машиноместо |

---

## Структура репозитория
```text
├── README.md                          # Двуязычная интерактивная документация
├── real_estate_valuation_ols.ipynb    # Полный аналитический ноутбук (EDA, VIF, OLS)
└── real_estate_dataset.csv            # 12 500 реальных объявлений о продаже недвижимости
```

---

## Инструкция по запуску
```bash
git clone https://github.com/axiom-stat/Linear-model.git
cd Linear-model
pip install pandas numpy statsmodels seaborn matplotlib
jupyter lab
```

</details>
