# -*- coding: utf-8 -*-
"""Складно MVP — генератор демо-данных.

Имитирует то, что в проде придёт из API Wildberries:
  demo_data/sku.csv           — каталог: баркод, цена, себестоимость, объём (л)
  demo_data/sales.csv         — 60 дней продаж (SKU x дата x регион), как /orders
  demo_data/stocks.csv        — текущие остатки FBO по складам (warehouseRemains)
  demo_data/coefficients.csv  — календарь коэффициентов приёмки на 14 дней
  demo_data/tariffs.csv       — склады: тариф хранения и приёмки, домашний регион
"""
import csv
import os
from datetime import date, timedelta

import numpy as np

rng = np.random.default_rng(7)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_data")
os.makedirs(OUT, exist_ok=True)

TODAY = date(2026, 7, 7)

# ---- Склады WB и их домашние регионы (кластеры) ----
WAREHOUSES = [
    # (склад, регион-кластер, доля спроса, хранение руб/л/день, приёмка руб/л при x1)
    ("Коледино",      "Центр",    0.42, 0.28, 9.0),
    ("Шушары",        "СЗ",       0.11, 0.22, 8.0),
    ("Краснодар",     "Юг",       0.12, 0.20, 8.0),
    ("Казань",        "Поволжье", 0.12, 0.20, 8.0),
    ("Екатеринбург",  "Урал",     0.09, 0.18, 7.5),
    ("Новосибирск",   "Сибирь",   0.08, 0.18, 7.5),
    ("Хабаровск",     "ДВ",       0.03, 0.16, 7.0),
    ("Тула",          "Центр-2",  0.03, 0.24, 8.5),
]
SHARES = np.array([w[2] for w in WAREHOUSES])

# ---- Каталог: 40 SKU, оборот ~5 млн руб/мес ----
N_SKU = 40
price = np.round(rng.uniform(700, 4200, N_SKU), -1)
cost = np.round(price * rng.uniform(0.40, 0.55, N_SKU), 0)
vol = np.round(rng.uniform(0.4, 6.0, N_SKU), 1)          # литры
raw = rng.lognormal(0.0, 1.0, N_SKU)
target_units_day = 5_000_000 / price.mean() / 30.0
sku_rate = raw / raw.sum() * target_units_day             # ед/день на SKU

with open(f"{OUT}/sku.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["sku_id", "barcode", "name", "price_rub", "cost_rub", "volume_l"])
    for i in range(N_SKU):
        w.writerow([f"SKU{i+1:03d}", f"46{rng.integers(10**10, 10**11)}",
                    f"Товар {i+1}", price[i], cost[i], vol[i]])

# ---- Продажи за 60 дней (как выгрузка /orders: дата, SKU, регион, шт) ----
sku_region_rate = sku_rate[:, None] * rng.dirichlet(SHARES * 70, size=N_SKU)
rows = []
for d in range(60):
    day = TODAY - timedelta(days=60 - d)
    season = 1.0 + 0.15 * np.sin(2 * np.pi * d / 30.0)   # лёгкая сезонность
    qty = rng.poisson(sku_region_rate * season)
    for i in range(N_SKU):
        for j, wh in enumerate(WAREHOUSES):
            if qty[i, j] > 0:
                rows.append([day.isoformat(), f"SKU{i+1:03d}", wh[1], qty[i, j]])
with open(f"{OUT}/sales.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["date", "sku_id", "region", "units"])
    w.writerows(rows)

# ---- Текущие остатки FBO (частично распределены, в основном Коледино) ----
with open(f"{OUT}/stocks.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["sku_id", "warehouse", "units"])
    for i in range(N_SKU):
        total_stock = int(sku_rate[i] * rng.uniform(3, 12))  # 3-12 дней запаса
        kol = int(total_stock * rng.uniform(0.6, 0.95))
        if kol:
            w.writerow([f"SKU{i+1:03d}", "Коледино", kol])
        rest = total_stock - kol
        if rest:
            j = rng.integers(1, len(WAREHOUSES))
            w.writerow([f"SKU{i+1:03d}", WAREHOUSES[j][0], rest])

# ---- Календарь коэффициентов приёмки на 14 дней (как /acceptance/coefficients)
# Коэффициент: -1 = приёмка закрыта; 0 = бесплатно; 1..20 = множитель.
BASE_PROFILE = {   # (вероятность закрытия, типичный диапазон коэффициента)
    "Коледино":     (0.25, (3, 8)),
    "Шушары":       (0.15, (1, 5)),
    "Краснодар":    (0.10, (0, 4)),
    "Казань":       (0.10, (0, 3)),
    "Екатеринбург": (0.08, (0, 3)),
    "Новосибирск":  (0.08, (0, 2)),
    "Хабаровск":    (0.05, (0, 2)),
    "Тула":         (0.15, (1, 6)),
}
with open(f"{OUT}/coefficients.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["date", "warehouse", "coefficient"])
    for wh, _, _, _, _ in WAREHOUSES:
        p_closed, (lo, hi) = BASE_PROFILE[wh]
        level = rng.uniform(lo, hi)
        for d in range(1, 15):
            level = np.clip(level + rng.normal(0, 1.0), lo, hi)
            if rng.random() < p_closed:
                coef = -1
            else:
                coef = int(round(level))
            w.writerow([(TODAY + timedelta(days=d)).isoformat(), wh, coef])

with open(f"{OUT}/tariffs.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["warehouse", "home_region", "demand_share",
                "storage_rub_per_l_day", "acceptance_rub_per_l_at_x1"])
    for wh in WAREHOUSES:
        w.writerow([wh[0], wh[1], wh[2], wh[3], wh[4]])

print(f"Демо-данные записаны в {OUT}/")
print(f"  SKU: {N_SKU}, строк продаж: {len(rows)}, "
      f"оборот ~{(price * sku_rate * 30).sum()/1e6:.1f} млн руб/мес")
