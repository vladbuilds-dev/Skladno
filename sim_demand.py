# Проверка риска "данные важнее математики" для Складно.
# Вопрос: насколько шумен прогноз спроса на уровне SKU x регион для типичного
# FBO-продавца (200 SKU, оборот ~5 млн руб/мес), и убивает ли этот шум ценность
# оптимизационного плана по сравнению с (а) оракулом и (б) простой эвристикой.

import numpy as np

rng = np.random.default_rng(42)

# ---------- Калибровка под ICP ----------
N_SKU = 200
PRICE = 1500.0            # руб, средний чек
MARGIN = 0.30             # маржа продавца
TURNOVER_TARGET = 5_000_000  # руб/мес

# 8 кластеров складов/регионов, доли спроса ~ реальная география WB
REGIONS = ["Центр", "СЗ", "Юг", "Поволжье", "Урал", "Сибирь", "ДВ", "Казань"]
REGION_SHARE = np.array([0.42, 0.11, 0.12, 0.12, 0.09, 0.08, 0.03, 0.03])

# Ставки спроса по SKU: логнормальное распределение (типичный длинный хвост)
total_units_per_day = TURNOVER_TARGET / PRICE / 30.0  # ~111 ед/день на все SKU
raw = rng.lognormal(mean=0.0, sigma=1.1, size=N_SKU)
sku_rate = raw / raw.sum() * total_units_per_day      # ед/день на SKU

# Истинные ставки SKU x регион: доля региона слегка варьируется по SKU
noise = rng.dirichlet(REGION_SHARE * 60, size=N_SKU)  # небольшая SKU-специфика
true_rate = sku_rate[:, None] * noise                 # (SKU, регион), ед/день

HIST_DAYS = 60     # истории для прогноза
PLAN_DAYS = 30     # горизонт поставки
N_REP = 30         # репликаций Монте-Карло

# ---------- Экономика ошибки ----------
# Недопоставка в регион: часть спроса уходит с нелокального склада
# (надбавка 2.5% от цены + падение выдачи => часть продаж теряется),
# часть теряется совсем.
LOST_FRAC = 0.25                                  # доля потерянных продаж при дефиците
COST_SHORT = LOST_FRAC * PRICE * MARGIN + (1 - LOST_FRAC) * PRICE * 0.025
# Перепоставка: хранение + заморозка капитала
STOR_PER_UNIT_DAY = 0.45                          # руб/ед/день (усредн. тариф)
CAPITAL_RATE = 0.24 / 365                         # 24% годовых
COST_OVER_DAY = STOR_PER_UNIT_DAY + PRICE * 0.7 * CAPITAL_RATE
COST_OVER = COST_OVER_DAY * PLAN_DAYS / 2         # излишек лежит в среднем полсрока

CRIT = COST_SHORT / (COST_SHORT + COST_OVER)      # ньюсвендор-квантиль

def stock_from_rate(rate_hat):
    """Запас на PLAN_DAYS по прогнозной ставке (ньюсвендор, Пуассон ~ нормаль)."""
    mu = rate_hat * PLAN_DAYS
    from scipy.stats import norm
    z = norm.ppf(CRIT)
    return np.maximum(0, np.round(mu + z * np.sqrt(np.maximum(mu, 1e-9))))

def cost(stock, demand):
    short = np.maximum(demand - stock, 0)
    over = np.maximum(stock - demand, 0)
    return (short * COST_SHORT + over * COST_OVER).sum()

results = {k: [] for k in ["oracle", "naive_cell", "pooled", "one_wh"]}
mape_by_bucket = []

for rep in range(N_REP):
    r = np.random.default_rng(1000 + rep)
    # История: Пуассон по дням
    hist = r.poisson(true_rate[None, :, :] * 1.0, size=(HIST_DAYS, N_SKU, len(REGIONS)))
    hist_sum = hist.sum(axis=0)                        # (SKU, регион)

    # Прогнозы
    est_cell = hist_sum / HIST_DAYS                    # наивный per-cell
    sku_tot = hist_sum.sum(axis=1) / HIST_DAYS         # ставка SKU (все регионы)
    pooled_share = hist_sum.sum(axis=0) / hist_sum.sum()  # доли регионов по портфелю
    est_pooled = sku_tot[:, None] * pooled_share[None, :]

    # Ошибка прогноза по бакетам скорости спроса (per-cell)
    with np.errstate(divide="ignore", invalid="ignore"):
        ape = np.abs(est_cell - true_rate) / true_rate
    mape_by_bucket.append((true_rate.flatten(), ape.flatten()))

    # Фактический спрос планового периода
    demand = r.poisson(true_rate * PLAN_DAYS)

    # Стратегии распределения
    results["oracle"].append(cost(stock_from_rate(true_rate), demand))
    results["naive_cell"].append(cost(stock_from_rate(est_cell), demand))
    results["pooled"].append(cost(stock_from_rate(est_pooled), demand))
    # "Всё на один склад" (Коледино): весь спрос вне Центра — нелокальный
    stock_one = stock_from_rate(sku_tot)               # весь запас в Центре
    d_center = demand[:, 0]
    d_rest = demand[:, 1:].sum(axis=1)
    short_c = np.maximum(d_center + d_rest * (1 - 0) - stock_one, 0)  # placeholder
    # проще: локальный спрос = Центр; остальной обслуживается нелокально с надбавкой,
    # плюс часть теряется из-за ранжирования
    served_nonlocal = d_rest
    over_c = np.maximum(stock_one - d_center - d_rest, 0)
    short_all = np.maximum(d_center + d_rest - stock_one, 0)
    cost_one = (short_all * COST_SHORT + over_c * COST_OVER).sum() \
               + (served_nonlocal * PRICE * 0.025 + served_nonlocal * PRICE * MARGIN * 0.10).sum()
    results["one_wh"].append(cost_one)

# ---------- Отчёт ----------
print(f"Калибровка: {N_SKU} SKU, {total_units_per_day:.0f} ед/день всего, "
      f"медианная ставка SKU = {np.median(sku_rate):.2f} ед/день")
print(f"Медианная ставка SKU x регион = {np.median(true_rate):.3f} ед/день")
print(f"Доля ячеек SKU x регион со ставкой < 0.1 ед/день: "
      f"{(true_rate < 0.1).mean()*100:.0f}%")
print(f"Критический квантиль (ньюсвендор): {CRIT:.2f}, "
      f"C_short={COST_SHORT:.0f}р, C_over={COST_OVER:.0f}р/ед\n")

rates = np.concatenate([m[0] for m in mape_by_bucket])
apes = np.concatenate([m[1] for m in mape_by_bucket])
print("Ошибка наивного per-cell прогноза (медианная APE) по скорости спроса:")
for lo, hi in [(0, 0.05), (0.05, 0.2), (0.2, 1.0), (1.0, 5.0), (5.0, 1e9)]:
    m = (rates >= lo) & (rates < hi)
    if m.sum():
        print(f"  ставка {lo:>4}-{hi if hi<1e9 else 'inf':>4} ед/день: "
              f"APE={np.median(apes[m])*100:>5.0f}%  (ячеек: {m.mean()*100:.0f}%)")

print("\nСтоимость ошибок плана за 30 дней (медиана по репликациям), руб:")
base = np.median(results["oracle"])
for k, label in [("oracle", "Оракул (знает истинный спрос)"),
                 ("naive_cell", "Наивный прогноз SKU x регион"),
                 ("pooled", "Пул: ставка SKU x доли регионов портфеля"),
                 ("one_wh", "Всё на один склад (Коледино)")]:
    v = np.median(results[k])
    print(f"  {label:<42} {v:>10,.0f}  (+{(v/base-1)*100:>5.1f}% к оракулу)")

print(f"\nОборот: {TURNOVER_TARGET/1e6:.0f} млн руб/мес; "
      f"маржинальная прибыль ~{TURNOVER_TARGET*MARGIN/1e6:.1f} млн руб/мес")
