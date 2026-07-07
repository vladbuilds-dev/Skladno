# -*- coding: utf-8 -*-
"""Складно MVP — конвейер: данные → прогноз → MILP → план поставки (xlsx).

Запуск:  python3 skladno_mvp.py [--budget 1500000] [--truck 4000]

Конвейер (в проде на входе — API WB, здесь — CSV из demo_data/):
  1. Прогноз спроса на 30 дней: иерархический (ставка SKU x доли регионов
     портфеля, эмпирико-байесовский бленд) + 3 сценария (P25/P50/P75).
  2. MILP (PuLP/CBC): что, сколько, на какой склад и на какую дату везти,
     максимизируя ожидаемую прибыль при ограничениях:
       - бюджет закупки, вместимость фуры на направление,
       - календарь коэффициентов приёмки (закрытые дни исключены),
       - тарифы хранения, надбавка за нелокальность 2.5%.
  3. Выгрузка: план в формате загрузки WB (xlsx) + экономика против
     baseline «всё на Коледино в ближайший открытый день».
"""
import argparse
import os
import sys
from collections import defaultdict
from datetime import date

import numpy as np
import pandas as pd
import pulp
from scipy.stats import poisson

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "demo_data")

PLAN_DAYS = 30
MARGIN_RATE = 0.30          # маржа от цены после комиссий/логистики WB
NONLOCAL_FEE = 0.025        # надбавка за нелокальную продажу, доля цены
NONLOCAL_LOST = 0.20        # доля продаж, теряемая при обслуживании издалека
CAPITAL_RATE = 0.24 / 365   # заморозка оборотного капитала, в день
DELAY_RUB_L_DAY = 0.6       # штраф за поздний завоз, руб/л за день ожидания
SCENARIOS = [(0.25, 0.30), (0.50, 0.40), (0.75, 0.30)]  # (квантиль, вес)


def load():
    sku = pd.read_csv(f"{DATA}/sku.csv").set_index("sku_id")
    sales = pd.read_csv(f"{DATA}/sales.csv", parse_dates=["date"])
    stocks = pd.read_csv(f"{DATA}/stocks.csv")
    coeff = pd.read_csv(f"{DATA}/coefficients.csv", parse_dates=["date"])
    tariffs = pd.read_csv(f"{DATA}/tariffs.csv").set_index("warehouse")
    return sku, sales, stocks, coeff, tariffs


def forecast(sku, sales, tariffs):
    """Иерархический прогноз: ставка SKU x доли регионов, EB-бленд по ячейкам."""
    days = (sales["date"].max() - sales["date"].min()).days + 1
    regions = list(tariffs["home_region"])
    cell = (sales.groupby(["sku_id", "region"])["units"].sum()
            .unstack(fill_value=0).reindex(index=sku.index, columns=regions,
                                           fill_value=0))
    sku_rate = cell.sum(axis=1) / days                       # ед/день на SKU
    pooled_share = cell.sum(axis=0) / cell.values.sum()      # доли регионов
    est_pooled = np.outer(sku_rate, pooled_share)
    est_cell = cell.values / days
    k = 8.0                                                   # сила пула
    w = cell.values / (cell.values + k)
    rate = w * est_cell + (1 - w) * est_pooled                # ед/день, SKU x регион
    return pd.DataFrame(rate, index=sku.index, columns=regions)


def build_and_solve(sku, rate, stocks, coeff, tariffs, budget, truck_l):
    skus = list(sku.index)
    whs = list(tariffs.index)
    region_of = dict(zip(tariffs.index, tariffs["home_region"]))
    dates = sorted(coeff["date"].dt.date.unique())
    cmap = {(r.warehouse, r.date.date()): r.coefficient
            for r in coeff.itertuples()}

    stock0 = defaultdict(int)
    for r in stocks.itertuples():
        stock0[(r.sku_id, r.warehouse)] += r.units

    # Сценарии спроса на 30 дней по SKU x регион
    mu = {(s, region_of[w]): rate.loc[s, region_of[w]] * PLAN_DAYS
          for s in skus for w in whs}
    dem = {(s, rg, i): int(poisson.ppf(q, m)) if m > 0 else 0
           for (s, rg), m in mu.items()
           for i, (q, _) in enumerate(SCENARIOS)}

    m = pulp.LpProblem("skladno", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("ship", [(s, w) for s in skus for w in whs],
                              lowBound=0, cat="Integer")
    y = pulp.LpVariable.dicts("day", [(w, d) for w in whs for d in dates
                                      if cmap.get((w, d), -1) >= 0], cat="Binary")
    z = pulp.LpVariable.dicts("vol", list(y.keys()), lowBound=0)   # литры на (склад, дата)
    sl = pulp.LpVariable.dicts("sold_loc",
                               [(s, w, i) for s in skus for w in whs
                                for i in range(len(SCENARIOS))], lowBound=0)
    snl = pulp.LpVariable.dicts("sold_nonloc",
                                [(s, w, i) for s in skus for w in whs if w != "Коледино"
                                 for i in range(len(SCENARIOS))], lowBound=0)

    open_days = {w: [d for d in dates if (w, d) in y] for w in whs}

    for w in whs:
        if not open_days[w]:   # склад закрыт весь горизонт
            for s in skus:
                m += x[(s, w)] == 0
            continue
        m += pulp.lpSum(y[(w, d)] for d in open_days[w]) <= 1
        ship_vol = pulp.lpSum(x[(s, w)] * sku.loc[s, "volume_l"] for s in skus)
        m += pulp.lpSum(z[(w, d)] for d in open_days[w]) == ship_vol
        for d in open_days[w]:
            m += z[(w, d)] <= truck_l * y[(w, d)]              # фура + связь с датой

    m += pulp.lpSum(x[(s, w)] * sku.loc[s, "cost_rub"]
                    for s in skus for w in whs) <= budget       # бюджет закупки

    for s in skus:
        for i in range(len(SCENARIOS)):
            for w in whs:
                d_i = dem[(s, region_of[w], i)]
                m += sl[(s, w, i)] <= d_i
                m += sl[(s, w, i)] <= stock0[(s, w)] + x[(s, w)]
                if w != "Коледино":
                    m += snl[(s, w, i)] <= d_i - sl[(s, w, i)]  # добор нелокально
            # нелокальные продажи обслуживает Коледино из своего остатка
            m += (pulp.lpSum(snl[(s, w, i)] for w in whs if w != "Коледино")
                  + sl[(s, "Коледино", i)]
                  <= stock0[(s, "Коледино")] + x[(s, "Коледино")])

    margin = {s: sku.loc[s, "price_rub"] * MARGIN_RATE for s in skus}
    margin_nl = {s: (1 - NONLOCAL_LOST) * (margin[s]
                 - sku.loc[s, "price_rub"] * NONLOCAL_FEE) for s in skus}

    revenue = pulp.lpSum(
        wgt * (pulp.lpSum(margin[s] * sl[(s, w, i)] for s in skus for w in whs)
               + pulp.lpSum(margin_nl[s] * snl[(s, w, i)]
                            for s in skus for w in whs if w != "Коледино"))
        for i, (_, wgt) in enumerate(SCENARIOS))

    accept = pulp.lpSum(z[(w, d)] * tariffs.loc[w, "acceptance_rub_per_l_at_x1"]
                        * cmap[(w, d)] for (w, d) in z)
    delay = pulp.lpSum(z[(w, d)] * DELAY_RUB_L_DAY * (d - dates[0]).days
                       for (w, d) in z)
    # хранение и капитал: ожидаемый остаток на конец периода, лежит ~полсрока
    leftover = pulp.lpSum(
        wgt * (stock0[(s, w)] + x[(s, w)] - sl[(s, w, i)]
               - (snl[(s, w, i)] if (w != "Коледино" and (s, w, i) in snl) else 0))
        * (tariffs.loc[w, "storage_rub_per_l_day"] * sku.loc[s, "volume_l"]
           + sku.loc[s, "cost_rub"] * CAPITAL_RATE) * PLAN_DAYS / 2
        for s in skus for w in whs for i, (_, wgt) in enumerate(SCENARIOS))

    m += revenue - accept - delay - leftover
    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=120)
    m.solve(solver)
    status = pulp.LpStatus[m.status]

    plan = {(s, w): int(round(x[(s, w)].value() or 0)) for s in skus for w in whs}
    ship_dates = {w: d for (w, d) in y if (y[(w, d)].value() or 0) > 0.5}
    return status, plan, ship_dates, pulp.value(m.objective)


def evaluate(plan, ship_dates, sku, rate, stocks, coeff, tariffs, label):
    """Считает ожидаемую экономику плана (та же модель, по сценариям)."""
    whs = list(tariffs.index)
    region_of = dict(zip(tariffs.index, tariffs["home_region"]))
    cmap = {(r.warehouse, r.date.date()): r.coefficient for r in coeff.itertuples()}
    dates = sorted(coeff["date"].dt.date.unique())
    stock0 = defaultdict(int)
    for r in stocks.itertuples():
        stock0[(r.sku_id, r.warehouse)] += r.units

    tot = dict(margin=0.0, accept=0.0, storage=0.0, nonlocal_fee=0.0, delay=0.0)
    for w, d in ship_dates.items():
        v = sum(plan.get((s, w), 0) * sku.loc[s, "volume_l"] for s in sku.index)
        tot["accept"] += v * tariffs.loc[w, "acceptance_rub_per_l_at_x1"] * max(cmap[(w, d)], 0)
        tot["delay"] += v * DELAY_RUB_L_DAY * (d - dates[0]).days

    for i, (q, wgt) in enumerate(SCENARIOS):
        for s in sku.index:
            price, cost_ = sku.loc[s, "price_rub"], sku.loc[s, "cost_rub"]
            mrg = price * MARGIN_RATE
            center_stock = stock0[(s, "Коледино")] + plan.get((s, "Коледино"), 0)
            d_center = int(poisson.ppf(q, rate.loc[s, region_of["Коледино"]] * PLAN_DAYS))
            sold_c = min(center_stock, d_center)
            center_left = center_stock - sold_c
            tot["margin"] += wgt * mrg * sold_c
            for w in whs:
                if w == "Коледино":
                    continue
                mu_ = rate.loc[s, region_of[w]] * PLAN_DAYS
                d_i = int(poisson.ppf(q, mu_)) if mu_ > 0 else 0
                have = stock0[(s, w)] + plan.get((s, w), 0)
                sold = min(have, d_i)
                short = d_i - sold
                nl = min(short, center_left)
                center_left -= nl
                tot["margin"] += wgt * (mrg * sold
                                        + (1 - NONLOCAL_LOST) * (mrg - price * NONLOCAL_FEE) * nl)
                tot["nonlocal_fee"] += wgt * (1 - NONLOCAL_LOST) * price * NONLOCAL_FEE * nl
                left = have - sold
                tot["storage"] += wgt * left * (tariffs.loc[w, "storage_rub_per_l_day"]
                                                * sku.loc[s, "volume_l"]
                                                + cost_ * CAPITAL_RATE) * PLAN_DAYS / 2
            tot["storage"] += wgt * center_left * (
                tariffs.loc["Коледино", "storage_rub_per_l_day"] * sku.loc[s, "volume_l"]
                + cost_ * CAPITAL_RATE) * PLAN_DAYS / 2
    tot["net"] = tot["margin"] - tot["accept"] - tot["storage"] - tot["delay"]
    tot["label"] = label
    return tot


def baseline_plan(plan_opt, sku, coeff, tariffs, truck_l):
    """Baseline: тот же суммарный объём закупки, но всё на Коледино,
    в ближайший открытый день (как делает продавец без инструмента)."""
    totals = defaultdict(int)
    for (s, w), q in plan_opt.items():
        totals[s] += q
    plan = {(s, "Коледино"): q for s, q in totals.items()}
    cmap = {(r.warehouse, r.date.date()): r.coefficient for r in coeff.itertuples()}
    dates = sorted(coeff["date"].dt.date.unique())
    open_d = [d for d in dates if cmap.get(("Коледино", d), -1) >= 0]
    return plan, {"Коледино": open_d[0] if open_d else dates[0]}


def export_xlsx(plan, ship_dates, sku, tariffs, econ_opt, econ_base, path):
    rows = []
    for (s, w), q in sorted(plan.items()):
        if q > 0:
            rows.append({
                "Склад": w, "Дата поставки": ship_dates.get(w, ""),
                "Артикул": s, "Баркод": sku.loc[s, "barcode"],
                "Количество, шт": q,
                "Объём, л": round(q * sku.loc[s, "volume_l"], 1),
                "Закупка, руб": int(q * sku.loc[s, "cost_rub"]),
            })
    plan_df = pd.DataFrame(rows)

    def row(name, key, sign=1):
        a, b = econ_opt[key] * sign, econ_base[key] * sign
        return {"Показатель": name, "Складно, руб": round(a),
                "Как обычно (всё на Коледино), руб": round(b),
                "Разница, руб": round(a - b)}

    econ_df = pd.DataFrame([
        row("Ожидаемая маржа за 30 дней", "margin"),
        row("Приёмка (коэффициенты)", "accept"),
        row("Хранение + заморозка капитала", "storage"),
        row("Надбавка за нелокальность", "nonlocal_fee"),
        row("Штраф за поздний завоз", "delay"),
        row("Итого: ожидаемая прибыль", "net"),
    ])
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        plan_df.to_excel(xw, sheet_name="План поставки", index=False)
        econ_df.to_excel(xw, sheet_name="Экономика", index=False)
        for ws in xw.sheets.values():
            for col in ws.columns:
                width = max(len(str(c.value or "")) for c in col) + 2
                ws.column_dimensions[col[0].column_letter].width = min(width, 38)
    return plan_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=1_500_000)
    ap.add_argument("--truck", type=float, default=4_000, help="литров на направление")
    args = ap.parse_args()

    if not os.path.exists(f"{DATA}/sku.csv"):
        sys.exit("Нет demo_data/ — сначала запустите generate_demo_data.py")

    sku, sales, stocks, coeff, tariffs = load()
    print(f"Данные: {len(sku)} SKU, {len(sales)} строк продаж, "
          f"{len(coeff)} записей календаря коэффициентов")

    rate = forecast(sku, sales, tariffs)
    print(f"Прогноз: суммарно {rate.values.sum()*PLAN_DAYS:,.0f} ед "
          f"на {PLAN_DAYS} дней")

    print("Решаю MILP (CBC)...")
    status, plan, ship_dates, obj = build_and_solve(
        sku, rate, stocks, coeff, tariffs, args.budget, args.truck)
    print(f"Статус: {status}")

    econ_opt = evaluate(plan, ship_dates, sku, rate, stocks, coeff, tariffs, "Складно")
    plan_b, dates_b = baseline_plan(plan, sku, coeff, tariffs, args.truck)
    econ_base = evaluate(plan_b, dates_b, sku, rate, stocks, coeff, tariffs, "Baseline")

    out = os.path.join(HERE, "план_поставки.xlsx")
    plan_df = export_xlsx(plan, ship_dates, sku, tariffs, econ_opt, econ_base, out)

    print(f"\n=== ПЛАН ПОСТАВКИ (склады и даты) ===")
    for w, d in sorted(ship_dates.items()):
        units = sum(q for (s, w2), q in plan.items() if w2 == w and q > 0)
        volume = sum(q * sku.loc[s, "volume_l"] for (s, w2), q in plan.items() if w2 == w)
        if units:
            print(f"  {w:<14} {d}  {units:>5} шт  {volume:>7.0f} л")
    print(f"\n=== ЭКОНОМИКА за {PLAN_DAYS} дней (ожидание по сценариям) ===")
    for k, name in [("margin", "Маржа"), ("accept", "Приёмка"),
                    ("storage", "Хранение+капитал"), ("nonlocal_fee", "Нелокальность"),
                    ("net", "Прибыль ИТОГО")]:
        print(f"  {name:<18} Складно: {econ_opt[k]:>10,.0f}  |  "
              f"Как обычно: {econ_base[k]:>10,.0f}")
    diff = econ_opt["net"] - econ_base["net"]
    logi_base = econ_base["accept"] + econ_base["storage"] + econ_base["nonlocal_fee"]
    logi_opt = econ_opt["accept"] + econ_opt["storage"] + econ_opt["nonlocal_fee"]
    print(f"\n  Выгода Складно: +{diff:,.0f} руб/мес "
          f"(логистические издержки: {logi_opt:,.0f} против {logi_base:,.0f}, "
          f"−{(1 - logi_opt/logi_base)*100:.0f}%)")
    print(f"  Файл плана: {out}")


if __name__ == "__main__":
    main()
