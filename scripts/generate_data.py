"""Generate labelled historical card transactions and load them into the database.

Legitimate behaviour includes the things that make fraud rules hard:
travellers spending abroad, night-time gamblers, new customers buying
electronics, occasional big-ticket purchases and new phones. Five fraud
patterns are injected on top, each labelled with its fraud_type:

  card_testing        burst of sub-$5 online charges, then a cash-out purchase
  account_takeover    new device, often foreign IP, spend far above normal
  bust_out            brand-new accounts maxing out on electronics / gift cards
  cloned_card         card-present use abroad shortly after a home transaction
  late_night_risk     high-value crypto / gambling between 1am and 4am

Velocity features (txn_count_1h, amount_sum_24h, ...) are materialised per
card with trailing time windows, the way a feature store would.

    python scripts/generate_data.py --customers 3000 --days 90 --seed 42
"""
import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import delete, insert

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import backtests, engine, init_db, rules, transactions  # noqa: E402
from app.features import COUNTRIES, FEATURES, MERCHANT_CATEGORIES  # noqa: E402

START = datetime(2026, 1, 1)

HOME_COUNTRY_P = dict(US=0.60, GB=0.12, IN=0.12, DE=0.06, AE=0.03, SG=0.03, BR=0.02, NG=0.01, RU=0.005, CN=0.005)
CATEGORY_P = dict(grocery=0.22, restaurant=0.18, fuel=0.11, online_retail=0.19, electronics=0.05, travel=0.04,
                  gift_cards=0.03, gambling=0.02, crypto=0.02, cash_withdrawal=0.14)
MEDIAN_AMOUNT = dict(grocery=45, restaurant=35, fuel=50, online_retail=60, electronics=320, travel=420,
                     gift_cards=50, gambling=80, crypto=250, cash_withdrawal=120)
ONLINE_SHARE = dict(grocery=0.05, restaurant=0.05, fuel=0.0, online_retail=1.0, electronics=0.5, travel=0.9,
                    gift_cards=0.9, gambling=0.95, crypto=1.0, cash_withdrawal=0.0)
# Daytime-heavy hour profile for genuine activity.
HOUR_W = np.array([1, .5, .3, .2, .2, .4, 1, 2, 3.5, 4.5, 5, 5.5, 6, 5.5, 5, 5, 5.5, 6, 6.5, 6, 5, 4, 3, 2])
HOUR_P = HOUR_W / HOUR_W.sum()
RISKY_FOREIGN = ["BR", "NG", "RU", "CN", "AE", "SG", "DE", "GB"]
MERCHANTS_PER_CATEGORY = 150


class Gen:
    def __init__(self, n_customers: int, days: int, seed: int):
        self.rng = np.random.default_rng(seed)
        self.days = days
        self.rows: list[dict] = []
        self.device_seq = 0
        self.customers = self._make_customers(n_customers)

    # ── helpers ──────────────────────────────────────────────────────────────
    def pick(self, mapping: dict):
        keys = list(mapping)
        p = np.array(list(mapping.values()), dtype=float)
        return keys[self.rng.choice(len(keys), p=p / p.sum())]

    def new_device(self) -> str:
        self.device_seq += 1
        return f"D{self.device_seq:07d}"

    def merchant(self, category: str) -> str:
        return f"M{MERCHANT_CATEGORIES.index(category):02d}{self.rng.integers(MERCHANTS_PER_CATEGORY):04d}"

    def add(self, cust: dict, ts: datetime, category: str, amount: float, channel: str, country: str,
            device: str | None = None, ip_country: str | None = None, fraud_type: str | None = None):
        self.rows.append(dict(
            ts=ts, customer_id=cust["id"], card_id=cust["card"], merchant_id=self.merchant(category),
            merchant_category=category, channel=channel, amount=round(max(amount, 0.5), 2),
            country=country, home_country=cust["home"], device_id=device,
            ip_country=ip_country or cust["home"], opened=cust["opened"],
            is_fraud=fraud_type is not None, fraud_type=fraud_type,
        ))

    def _make_customers(self, n: int) -> list[dict]:
        out = []
        for i in range(n):
            new_account = self.rng.random() < 0.08
            age_at_start = self.rng.integers(0, 25) if new_account else int(self.rng.uniform(60, 3000))
            out.append(dict(
                id=f"U{i:05d}", card=f"C{i:05d}", home=self.pick(HOME_COUNTRY_P),
                opened=START - timedelta(days=int(age_at_start)),
                rate=self.rng.gamma(2.0, 0.1),  # mean ~0.2 txns/day
                scale=float(np.exp(self.rng.normal(0, 0.4))),
                devices=[self.new_device() for _ in range(self.rng.integers(1, 3))],
                traveller=self.rng.random() < 0.07,
                night_gambler=self.rng.random() < 0.015,
            ))
        return out

    def rand_ts(self, cust: dict, hour: int | None = None) -> datetime:
        start = max(START, cust["opened"])
        span_days = max((START + timedelta(days=self.days) - start).days, 1)
        day = start + timedelta(days=int(self.rng.integers(span_days)))
        hour = int(self.rng.choice(24, p=HOUR_P)) if hour is None else hour
        return day.replace(hour=0, minute=0, second=0) + timedelta(
            hours=hour, minutes=int(self.rng.integers(60)), seconds=int(self.rng.integers(60)))

    # ── genuine activity ─────────────────────────────────────────────────────
    def legit(self):
        for c in self.customers:
            trips = []
            if c["traveller"]:
                for _ in range(self.rng.integers(1, 3)):
                    t0 = START + timedelta(days=int(self.rng.integers(self.days - 10)))
                    dest = self.rng.choice([x for x in COUNTRIES if x != c["home"]])
                    trips.append((t0, t0 + timedelta(days=int(self.rng.integers(3, 11))), str(dest)))

            n = self.rng.poisson(c["rate"] * self.days)
            for _ in range(n):
                cat = self.pick(CATEGORY_P)
                ts = self.rand_ts(c)
                self._legit_txn(c, ts, cat, trips)

            if c["night_gambler"]:
                for _ in range(self.rng.integers(5, 20)):
                    ts = self.rand_ts(c, hour=int(self.rng.integers(0, 5)))
                    cat = "gambling" if self.rng.random() < 0.7 else "crypto"
                    amt = MEDIAN_AMOUNT[cat] * np.exp(self.rng.normal(0.8, 0.9))
                    self.add(c, ts, cat, amt, "ecommerce", c["home"], device=c["devices"][0])

    def _legit_txn(self, c: dict, ts: datetime, cat: str, trips: list):
        online = self.rng.random() < ONLINE_SHARE[cat]
        channel = "ecommerce" if online else ("atm" if cat == "cash_withdrawal" else "pos")
        amount = MEDIAN_AMOUNT[cat] * c["scale"] * np.exp(self.rng.normal(0, 0.7))
        if self.rng.random() < 0.01:  # occasional big-ticket purchase
            amount *= self.rng.uniform(5, 15)
        if online and cat == "online_retail" and self.rng.random() < 0.04:  # app-store style micro charge
            amount = self.rng.uniform(0.99, 4.99)

        country, ip = c["home"], c["home"]
        for t0, t1, dest in trips:
            if t0 <= ts <= t1:
                country = ip = dest
        if online:
            country = c["home"]  # online purchases are booked to the home-country merchant
            if self.rng.random() < 0.01:
                ip = str(self.rng.choice(RISKY_FOREIGN))  # VPN users
        device = None
        if online:
            if self.rng.random() < 0.02:
                c["devices"].append(self.new_device())  # new phone / laptop
                device = c["devices"][-1]
            else:
                device = c["devices"][int(self.rng.integers(len(c["devices"])))]
        self.add(c, ts, cat, amount, channel, country, device=device, ip_country=ip)

    # ── fraud patterns ───────────────────────────────────────────────────────
    def victims(self, k: int) -> list[dict]:
        return [self.customers[i] for i in self.rng.choice(len(self.customers), size=k, replace=False)]

    def card_testing(self, episodes: int):
        for c in self.victims(episodes):
            t = self.rand_ts(c)
            dev = self.new_device()
            ip = str(self.rng.choice(RISKY_FOREIGN)) if self.rng.random() < 0.6 else c["home"]
            for _ in range(self.rng.integers(6, 16)):
                t += timedelta(seconds=int(self.rng.integers(30, 240)))
                cat = "online_retail" if self.rng.random() < 0.7 else "gift_cards"
                self.add(c, t, cat, self.rng.uniform(0.5, 4.99), "ecommerce", c["home"], dev, ip, "card_testing")
            for _ in range(self.rng.integers(1, 3)):
                t += timedelta(minutes=int(self.rng.integers(20, 180)))
                cat = "electronics" if self.rng.random() < 0.5 else "gift_cards"
                self.add(c, t, cat, self.rng.uniform(300, 1500), "ecommerce", c["home"], dev, ip, "card_testing")

    def account_takeover(self, episodes: int):
        for c in self.victims(episodes):
            t = self.rand_ts(c)
            dev = self.new_device()
            ip = str(self.rng.choice(RISKY_FOREIGN)) if self.rng.random() < 0.7 else c["home"]
            for _ in range(self.rng.integers(1, 4)):
                cat = str(self.rng.choice(["electronics", "gift_cards", "crypto", "online_retail"]))
                amt = max(MEDIAN_AMOUNT[cat] * c["scale"] * self.rng.uniform(6, 25), self.rng.uniform(500, 4000))
                self.add(c, t, cat, amt, "ecommerce", c["home"], dev, ip, "account_takeover")
                t += timedelta(minutes=int(self.rng.integers(5, 60)))

    def bust_out(self, n_accounts: int):
        base = len(self.customers)
        for i in range(n_accounts):
            opened = START + timedelta(days=int(self.rng.integers(0, self.days - 30)))
            c = dict(id=f"U{base + i:05d}", card=f"C{base + i:05d}", home=self.pick(HOME_COUNTRY_P),
                     opened=opened, scale=1.0, devices=[self.new_device()])
            self.customers.append(c)
            t = opened + timedelta(days=int(self.rng.integers(1, 5)), hours=int(self.rng.integers(9, 20)))
            for _ in range(self.rng.integers(1, 4)):  # look normal first
                cat = self.pick(dict(grocery=1, restaurant=1, fuel=1))
                self.add(c, t, cat, MEDIAN_AMOUNT[cat] * np.exp(self.rng.normal(0, 0.5)), "pos", c["home"])
                t += timedelta(days=int(self.rng.integers(1, 6)))
            t += timedelta(days=int(self.rng.integers(3, 18)))
            for _ in range(self.rng.integers(3, 9)):
                cat = "electronics" if self.rng.random() < 0.6 else "gift_cards"
                online = self.rng.random() < 0.5
                self.add(c, t, cat, self.rng.uniform(300, 2500), "ecommerce" if online else "pos", c["home"],
                         c["devices"][0] if online else None, None, "bust_out")
                t += timedelta(minutes=int(self.rng.integers(10, 180)))

    def cloned_card(self, episodes: int):
        for c in self.victims(episodes):
            t = self.rand_ts(c)
            self.add(c, t, "grocery", MEDIAN_AMOUNT["grocery"] * c["scale"], "pos", c["home"])
            dest = str(self.rng.choice([x for x in RISKY_FOREIGN if x != c["home"]]))
            t += timedelta(hours=float(self.rng.uniform(1, 8)))
            for _ in range(self.rng.integers(2, 7)):
                if self.rng.random() < 0.6:
                    self.add(c, t, "cash_withdrawal", self.rng.uniform(200, 800), "atm", dest, fraud_type="cloned_card")
                else:
                    self.add(c, t, "electronics", self.rng.uniform(250, 2000), "pos", dest, fraud_type="cloned_card")
                t += timedelta(minutes=int(self.rng.integers(10, 150)))

    def late_night_risk(self, episodes: int):
        for c in self.victims(episodes):
            t = self.rand_ts(c, hour=int(self.rng.integers(1, 4)))
            dev = self.new_device() if self.rng.random() < 0.5 else c["devices"][0]
            for _ in range(self.rng.integers(1, 3)):
                cat = "crypto" if self.rng.random() < 0.5 else "gambling"
                self.add(c, t, cat, self.rng.uniform(800, 5000), "ecommerce", c["home"], dev, None, "late_night_risk")
                t += timedelta(minutes=int(self.rng.integers(3, 30)))


def materialise_features(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.sort_values(["card_id", "ts"], kind="stable").reset_index(drop=True)
    n = len(df)
    ts = df["ts"].values.astype("datetime64[ns]").astype(np.int64)
    amt = df["amount"].to_numpy()
    merch = df["merchant_id"].to_numpy()
    ctry = df["country"].to_numpy()
    H, D = 3600 * 10**9, 86400 * 10**9

    cnt1h = np.empty(n, int); cnt24 = np.empty(n, int); sum24 = np.empty(n)
    dmer1h = np.empty(n, int); dctry24 = np.empty(n, int)
    avg30 = np.full(n, np.nan); gap = np.full(n, 43200.0)

    for idx in df.groupby("card_id", sort=False).indices.values():
        t, a = ts[idx], amt[idx]
        lo1h = np.searchsorted(t, t - H, side="right")
        lo24 = np.searchsorted(t, t - D, side="right")
        lo30 = np.searchsorted(t, t - 30 * D, side="left")
        cs = np.concatenate([[0.0], np.cumsum(a)])
        pos = np.arange(len(t))
        cnt1h[idx] = pos - lo1h + 1
        cnt24[idx] = pos - lo24 + 1
        sum24[idx] = cs[pos + 1] - cs[lo24]
        prev_n = pos - lo30
        with np.errstate(invalid="ignore", divide="ignore"):
            avg30[idx] = np.where(prev_n > 0, (cs[pos] - cs[lo30]) / np.maximum(prev_n, 1), np.nan)
        gap[idx[1:]] = np.minimum(np.diff(t) / 60e9, 43200.0)
        m, c = merch[idx], ctry[idx]
        dmer1h[idx] = [len(set(m[lo1h[i]:i + 1])) for i in pos]
        dctry24[idx] = [len(set(c[lo24[i]:i + 1])) for i in pos]

    df["txn_count_1h"], df["txn_count_24h"], df["amount_sum_24h"] = cnt1h, cnt24, sum24.round(2)
    df["distinct_merchants_1h"], df["distinct_countries_24h"] = dmer1h, dctry24
    df["minutes_since_last_txn"] = gap.round(1)
    df["amount_to_avg_ratio"] = np.where(np.isnan(avg30), 1.0, df["amount"] / avg30).round(3)
    df["hour_of_day"] = df["ts"].dt.hour
    df["account_age_days"] = (df["ts"] - df["opened"]).dt.days.clip(lower=0)
    df["is_international"] = df["country"] != df["home_country"]
    df["ip_country_mismatch"] = (df["channel"] == "ecommerce") & (df["ip_country"] != df["home_country"])
    first_seen = ~df.duplicated(["customer_id", "device_id"]) & df["device_id"].notna()
    df["is_new_device"] = first_seen

    df = df.sort_values("ts", kind="stable").reset_index(drop=True)
    df["txn_id"] = [f"T{i:08d}" for i in range(len(df))]
    cols = ["txn_id", "ts", "customer_id", "card_id", "merchant_id", *FEATURES, "is_fraud", "fraud_type"]
    return df[cols]


def build(customers: int, days: int, seed: int) -> pd.DataFrame:
    g = Gen(customers, days, seed)
    g.legit()
    scale = customers / 3000
    g.card_testing(int(35 * scale))
    g.account_takeover(int(80 * scale))
    g.bust_out(int(40 * scale))
    g.cloned_card(int(60 * scale))
    g.late_night_risk(int(50 * scale))
    raw = pd.DataFrame(g.rows)
    raw = raw[raw["ts"] < START + timedelta(days=days)]
    return materialise_features(raw)


def load(df: pd.DataFrame, reset_rules: bool) -> None:
    init_db()
    records = df.astype(object).where(df.notna(), None).to_dict(orient="records")
    with engine.begin() as conn:
        if reset_rules:
            conn.execute(delete(backtests))
            conn.execute(delete(rules))
        conn.execute(delete(transactions))
        for i in range(0, len(records), 5000):
            conn.execute(insert(transactions), records[i:i + 5000])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--customers", type=int, default=3000)
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reset-rules", action="store_true", help="also delete stored rules and back-tests")
    ap.add_argument("--csv", help="additionally write the dataset to this CSV path")
    args = ap.parse_args()

    df = build(args.customers, args.days, args.seed)
    if args.csv:
        df.to_csv(args.csv, index=False)
    load(df, args.reset_rules)

    print(f"Loaded {len(df):,} transactions into {engine.dialect.name} database '{engine.url.database}'")
    print(f"Fraud rate: {df['is_fraud'].mean():.2%}")
    print(df[df["is_fraud"]]["fraud_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
