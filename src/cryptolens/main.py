#!/usr/bin/env python
# coding: utf-8

# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maoki1980/CryptoLens/blob/main/src/cryptolens/main.ipynb)

# In[ ]:


import glob
import os
import re
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm


# In[ ]:


# ByBitの通貨名から余計な倍率の数値の除去する関数
def remove_specific_numbers(text):
    # パターンを作成 (10, 100, 1000, ...)
    pattern = r"1(0{1,})"
    return re.sub(pattern, "", text)


# In[ ]:


# dfの列のリスト内の空要素を除去する関数
def remove_empty_elements(lst):
    return [x for x in lst if x]


# In[ ]:


# リストの各要素の前後スペースを削除する関数
def strip_space(lst):
    return [s.strip() for s in lst]


# In[ ]:


# 仮想通貨のfeatherデータの中の更新日時を確認してデータを再取得するかのフラグを出力する関数
def need_update_coin_data(df, days):
    now = datetime.now().astimezone(ZoneInfo("Asia/Tokyo"))
    latest_ratio_update = df["ratioUpdateTime"].max()
    latest_coin_update = df["coinUpdateTime"].max()
    latest_update = max(latest_ratio_update, latest_coin_update)
    return (now - latest_update) > timedelta(days=days)


# In[ ]:


# カテゴリのfeatherデータの中の更新日時を確認してデータを再取得するかのフラグを出力する関数
def need_update_category_data(df, days):
    now = datetime.now().astimezone(ZoneInfo("Asia/Tokyo"))
    latest_update = df["categoryUpdateTime"].max()
    return (now - latest_update) > timedelta(days=days)


# In[ ]:


# JSONのキーから値を取得する関数 (キーがない場合デフォルト値を返す)
def get_nested_value(json, keys, default=None):
    for key in keys:
        json = json.get(key, default)
        if json is None:
            return default
    return json


# In[ ]:


# CoinGeckoから仮想通貨のカテゴリ一覧を取得する関数
def get_coingecko_categories_list(api_key):
    url = "https://api.coingecko.com/api/v3/coins/categories?order=market_cap_desc"
    headers = {"accept": "application/json", "x-cg-demo-api-key": api_key}
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(
            f"Error fetching CoinGecko categories list: {response.status_code}, {response.text}"
        )
        return pd.DataFrame()
    df = pd.DataFrame(response.json())
    # 必要な列だけ抽出
    df = df[
        [
            "id",
            "name",
            "market_cap",
            "market_cap_change_24h",
            "volume_24h",
            "updated_at",
        ]
    ]
    # 更新日時の形式をJSTのdatetime形式に変更
    df["updated_at"] = pd.to_datetime(df["updated_at"], utc=True)
    df["updated_at"] = df["updated_at"].dt.tz_convert("Asia/Tokyo")
    df["updated_at"] = df["updated_at"].dt.floor("s")
    # 列名変更
    columns = [
        "categoryId",
        "categoryName",
        "categoryCap",
        "categoryCapChg24h",
        "categoryVol24h",
        "categoryUpdateTime",
    ]
    df.columns = columns
    return df


# In[ ]:


# CoinGeckoから仮想通貨一覧を取得する関数
def get_coingecko_coins_list(api_key):
    url = "https://api.coingecko.com/api/v3/coins/list"
    headers = {"accept": "application/json", "x-cg-demo-api-key": api_key}
    params = {
        "include_platform": "true",
    }
    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        print(
            f"Error fetching CoinGecko coins list: {response.status_code}, {response.text}"
        )
        return pd.DataFrame()
    json = response.json()
    for entry in json:
        entry["platforms"] = list(entry["platforms"].keys())
    df = pd.DataFrame(json)
    df["platforms"] = df["platforms"].apply(remove_empty_elements)
    df["platforms"] = df["platforms"].apply(strip_space)
    df["id"] = df["id"].str.strip()
    df["symbol"] = df["symbol"].str.strip()
    df["name"] = df["name"].str.strip()
    columns = ["coinId", "coin", "coinName", "coinPlatforms"]
    df.columns = columns
    return df


# In[ ]:


# CoinGeckoから仮想通貨の詳細情報を取得する関数
def get_coingecko_coin_info(id, api_key):
    url = f"https://api.coingecko.com/api/v3/coins/{id}"
    headers = {"accept": "application/json", "x-cg-demo-api-key": api_key}
    params = {"sparkline": "true"}
    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        print(
            f"Error fetching CoinGecko coins info: {response.status_code}, {response.text}"
        )
        return pd.DataFrame()
    json = response.json()

    json_info = {
        "coinId": json.get("id"),
        "coin": json.get("symbol"),
        "coinName": json.get("name"),
        "coinSlug": json.get("web_slug"),
        "categories": json.get("categories"),
        "assetPlatformId": json.get("asset_platform_id"),
        "platforms": json.get("platforms"),
        "facebookLikes": get_nested_value(json, ["community_data", "facebook_likes"]),
        "redditSubscribers": get_nested_value(
            json, ["community_data", "reddit_subscribers"]
        ),
        "telegramUserCount": get_nested_value(
            json, ["community_data", "telegram_channel_user_count"]
        ),
        "xFollowers": get_nested_value(json, ["community_data", "twitter_followers"]),
        "coinCap": get_nested_value(json, ["market_data", "market_cap", "usd"]),
        "coinCapRank": json.get("market_cap_rank"),
        "coinCapFdvRatio": json.get("market_cap_fdv_ratio"),
        "coinCapChg%24h": get_nested_value(
            json, ["market_data", "market_cap_change_percentage_24h_in_currency", "usd"]
        ),
        "ath": get_nested_value(json, ["market_data", "ath", "usd"]),
        "athChg%": get_nested_value(
            json, ["market_data", "ath_change_percentage", "usd"]
        ),
        "athDate": get_nested_value(json, ["market_data", "ath_date", "usd"]),
        "atl": get_nested_value(json, ["market_data", "atl", "usd"]),
        "atlChg%": get_nested_value(
            json, ["market_data", "atl_change_percentage", "usd"]
        ),
        "atlDate": get_nested_value(json, ["market_data", "atl_date", "usd"]),
        "currentPrice": get_nested_value(json, ["market_data", "current_price", "usd"]),
        "priceChg%1h": get_nested_value(
            json, ["market_data", "price_change_percentage_1h_in_currency", "usd"]
        ),
        "priceChg%24h": get_nested_value(
            json, ["market_data", "price_change_percentage_24h_in_currency", "usd"]
        ),
        "priceChg%7d": get_nested_value(
            json, ["market_data", "price_change_percentage_7d_in_currency", "usd"]
        ),
        "priceChg%14d": get_nested_value(
            json, ["market_data", "price_change_percentage_14d_in_currency", "usd"]
        ),
        "priceChg%30d": get_nested_value(
            json, ["market_data", "price_change_percentage_30d_in_currency", "usd"]
        ),
        "priceChg%60d": get_nested_value(
            json, ["market_data", "price_change_percentage_60d_in_currency", "usd"]
        ),
        "priceChg%200d": get_nested_value(
            json, ["market_data", "price_change_percentage_200d_in_currency", "usd"]
        ),
        "sentimentVotesUp%": json.get("sentiment_votes_up_percentage"),
        "watchlistUsers": json.get("watchlist_portfolio_users"),
        "coinUpdateTime": json.get("last_updated"),
    }
    json_info["platforms"] = list(json_info["platforms"].keys())
    df = pd.DataFrame([json_info])
    return df


# In[ ]:


# CoinGeckoから渡したidリストのすべての詳細情報を取得する関数
def get_all_info(l_ids, api_key, wait_time=2.05):
    df = pd.DataFrame()
    errors = []

    for id in tqdm(l_ids, desc="Fetching coin info from CoinGecko"):
        try:
            df_info = get_coingecko_coin_info(id, api_key)
            df_info = df_info.dropna(axis=1, how="all")
            df = pd.concat([df, df_info], ignore_index=True)
        except Exception as e:
            errors.append((id, str(e)))
            print(f"Error fetching info data for id {id}: {e}")
        time.sleep(wait_time)

    if errors:
        print("Errors occurred for the following ids:")
        for error in errors:
            print(f"ID: {error[0]}, Error: {error[1]}")

    # 日時の形式をJSTのdatetime形式に変更
    df["athDate"] = pd.to_datetime(df["athDate"], utc=True)
    df["athDate"] = df["athDate"].dt.tz_convert("Asia/Tokyo")
    df["athDate"] = df["athDate"].dt.floor("s")
    df["atlDate"] = pd.to_datetime(df["atlDate"], utc=True)
    df["atlDate"] = df["atlDate"].dt.tz_convert("Asia/Tokyo")
    df["atlDate"] = df["atlDate"].dt.floor("s")
    df["coinUpdateTime"] = pd.to_datetime(df["coinUpdateTime"], utc=True)
    df["coinUpdateTime"] = df["coinUpdateTime"].dt.tz_convert("Asia/Tokyo")
    df["coinUpdateTime"] = df["coinUpdateTime"].dt.floor("s")

    df["categories"] = df["categories"].apply(remove_empty_elements)
    df["platforms"] = df["platforms"].apply(remove_empty_elements)

    # strip
    df["categories"] = df["categories"].apply(strip_space)
    df["platforms"] = df["platforms"].apply(strip_space)
    df["coinId"] = df["coinId"].str.strip()
    df["coin"] = df["coin"].str.strip()
    df["coinName"] = df["coinName"].str.strip()
    df["coinSlug"] = df["coinSlug"].str.strip()
    df["assetPlatformId"] = df["assetPlatformId"].str.strip()

    df["coinCapChg%24h"] = df["coinCapChg%24h"] / 100
    df["athChg%"] = df["athChg%"] / 100
    df["atlChg%"] = df["atlChg%"] / 100
    df["priceChg%1h"] = df["priceChg%1h"] / 100
    df["priceChg%24h"] = df["priceChg%24h"] / 100
    df["priceChg%7d"] = df["priceChg%7d"] / 100
    df["priceChg%14d"] = df["priceChg%14d"] / 100
    df["priceChg%30d"] = df["priceChg%30d"] / 100
    df["priceChg%60d"] = df["priceChg%60d"] / 100
    df["priceChg%200d"] = df["priceChg%200d"] / 100
    df["sentimentVotesUp%"] = df["sentimentVotesUp%"] / 100

    return df


# In[ ]:


# ByBitから仮想通貨一覧を取得する関数
def get_bybit_coins_list(category):
    url = "https://api.bybit.com/v5/market/instruments-info"
    headers = {}
    params = {
        "category": category,
        "limit": 1000,
    }
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        json = response.json()
        if json["retCode"] == 0:
            json = json["result"]["list"]
            df = pd.DataFrame(json)
            # 対象を絞る
            df = df[df["quoteCoin"] == "USDT"].reset_index(drop=True)
            df = df[df["contractType"] == "LinearPerpetual"].reset_index(drop=True)
            df = df[df["status"] == "Trading"].reset_index(drop=True)
            # 必要な列だけ取得
            df = df[["symbol", "baseCoin", "launchTime"]]
            # strip
            df["symbol"] = df["symbol"].str.strip()
            df["baseCoin"] = df["baseCoin"].str.strip()
            # 仮想通貨名から倍率の数値を除去し、小文字にする
            df["baseCoin"] = df["baseCoin"].apply(remove_specific_numbers)
            df["baseCoin"] = df["baseCoin"].str.lower()
            # launchTimeをUNIXタイムスタンプからJSTのdatetime形式にする
            df["launchTime"] = pd.to_numeric(df["launchTime"])
            df["launchTime"] = pd.to_datetime(df["launchTime"], unit="ms", utc=True)
            df["launchTime"] = df["launchTime"].dt.tz_convert("Asia/Tokyo")
            df.columns = ["symbol", "coin", "coinLaunchTime"]
        else:
            print(f'Error fetching Bybit coins list: {json["retMsg"]}')
            return pd.DataFrame()
    else:
        print(
            f"Error fetching Bybit coins list: {response.status_code}, {response.text}"
        )
        return pd.DataFrame()

    return df


# In[ ]:


# ByBitから仮想通貨の売買比率を取得する関数
def get_bybit_long_short_ratio(category, symbol, period, limit):
    url = "https://api.bybit.com/v5/market/account-ratio"
    headers = {}
    params = {
        "category": category,
        "symbol": symbol,
        "period": period,
        "limit": limit,
    }
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        json = response.json()
        if json["retCode"] == 0:
            json = json["result"]["list"]
            df = pd.DataFrame(json)
            df = df[["symbol", "buyRatio", "timestamp"]]
        else:
            print(f'Error fetching Bybit long short ratio: {json["retMsg"]}')
            return pd.DataFrame()
    else:
        print(
            f"Error fetching Bybit long short ratio: {response.status_code}, {response.text}"
        )
        return pd.DataFrame()
    return df


# In[ ]:


# ByBitから渡したsymbolリストのすべての売買比率を取得する関数
def get_all_ratios(l_symbols, wait_time=0.05):
    df = pd.DataFrame()
    errors = []

    for symbol in tqdm(l_symbols, desc="Fetching long-short ratios from ByBit"):
        try:
            df_ratio = get_bybit_long_short_ratio("linear", symbol, "1d", 1)
            df = pd.concat([df, df_ratio], ignore_index=True)
        except Exception as e:
            errors.append((symbol, str(e)))
            print(f"Error fetching ratio data for symbol {symbol}: {e}")
        time.sleep(wait_time)

    if errors:
        print("Errors occurred for the following symbols:")
        for error in errors:
            print(f"Symbol: {error[0]}, Error: {error[1]}")

    # 売買比率を数値に変換
    df["buyRatio"] = pd.to_numeric(df["buyRatio"])
    # timestampをUNIXタイムスタンプからJSTの日時形式にする
    df["timestamp"] = pd.to_numeric(df["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["timestamp"] = df["timestamp"].dt.tz_convert("Asia/Tokyo")
    df = df.rename(columns={"timestamp": "ratioUpdateTime"})

    return df


# In[ ]:


# .envファイルからAPIキーを読み込む
load_dotenv("../../.env")
coingecko_api_key = os.getenv("COINGECKO_API_KEY")

data_dir = "./data"
elapsed_days = 5


# In[ ]:


# featherファイルを読み込んでデータの更新日時を確認する
l_feather_coins_files = glob.glob(os.path.join(data_dir, "df_coins_*.feather"))
l_feather_categories_files = glob.glob(
    os.path.join(data_dir, "df_categories_*.feather")
)
if l_feather_coins_files and l_feather_categories_files:
    feather_coins_file_path = max(l_feather_coins_files, key=os.path.basename)
    feather_coins_file_name = os.path.basename(feather_coins_file_path)
    feather_categories_file_path = max(l_feather_categories_files, key=os.path.basename)
    feather_categories_file_name = os.path.basename(feather_categories_file_path)
    df_coins = pd.read_feather(feather_coins_file_path)
    df_categories = pd.read_feather(feather_categories_file_path)
    if need_update_coin_data(df_coins, elapsed_days) and need_update_category_data(
        df_coins, elapsed_days
    ):
        print(
            f"The data in feather files is older than {elapsed_days} days. Re-fetching data."
        )
        df_coins = pd.DataFrame()
        df_categories = pd.DataFrame()
    else:
        latest_ratio_update = df_coins["ratioUpdateTime"].max()
        latest_coin_update = df_coins["coinUpdateTime"].max()
        latest_category_update = df_categories["categoryUpdateTime"].max()
        latest_update = max(
            latest_ratio_update, latest_coin_update, latest_category_update
        )
        print(f"Loaded data from feather files, latest update at {latest_update}")
else:
    df_coins = pd.DataFrame()
    df_categories = pd.DataFrame()


# In[ ]:


# 仮想通貨データを作成する
if df_coins.empty or df_categories.empty:
    # ByBitの仮想通貨リストを取得 (無期限先物)
    df_bybit_coins = get_bybit_coins_list("linear")
    print(f"ByBit coins df size: {df_bybit_coins.shape}")

    # CoinGeckoの仮想通貨リストを取得
    df_coingecko_coins = get_coingecko_coins_list(coingecko_api_key)
    print(f"CoinGecko coins df size: {df_coingecko_coins.shape}")

    # ByBitとCoinGeckoでそれぞれ取得した仮想通貨リストをマージする
    df_coins = pd.merge(
        df_bybit_coins, df_coingecko_coins, on="coin", how="inner"
    ).reset_index(drop=True)

    # symbolのリストを作成する
    l_coin_symbols = list(df_coins["symbol"].unique())
    l_coin_symbols = l_coin_symbols[-10:]
    # idのリストを作成する
    l_coin_ids = list(df_coins["coinId"].unique())
    l_coin_ids = l_coin_ids[-10:]

    # すべてのsymbolについて売買比率を取得する
    df_bybit_long_short_ratio = get_all_ratios(l_coin_symbols)

    # 仮想通貨リストに売買比率をマージする
    df_coins = pd.merge(
        df_coins, df_bybit_long_short_ratio, on="symbol", how="inner"
    ).reset_index(drop=True)

    # すべてのidについてCoinGeckoから仮想通貨の詳細情報を取得する
    df_coingecko_coin_info = get_all_info(l_coin_ids, coingecko_api_key)

    # 仮想通貨リストに詳細情報をマージする
    df_coins = pd.merge(
        df_coins, df_coingecko_coin_info, on=["coinId", "coinName", "coin"], how="inner"
    ).reset_index(drop=True)

    # Noneを欠損値に置換する
    df_coins = df_coins.replace({None: np.nan})

    now = datetime.now().astimezone(ZoneInfo("Asia/Tokyo")).strftime("%Y%m%d%H%M")

    # 仮想通貨リストをfeatherファイルに保存
    feather_coins_file_name = f"df_coins_{now}.feather"
    feather_coins_file_path = os.path.join(data_dir, feather_coins_file_name)
    df_coins.to_feather(feather_coins_file_path)
    print(f'Data saved to "{feather_coins_file_name}"')

    # CoinGeckoのカテゴリリストを取得
    df_categories = get_coingecko_categories_list(coingecko_api_key)
    df_categories["categoryId"] = df_categories[
        "categoryId"
    ].str.strip()
    df_categories["categoryName"] = df_categories[
        "categoryName"
    ].str.strip()
    print(f"CoinGecko categories df size: {df_categories.shape}")

    # カテゴリリストをfeatherファイルに保存
    feather_categories_file_name = f"df_categories_{now}.feather"
    feather_categories_file_path = os.path.join(data_dir, feather_categories_file_name)
    df_categories.to_feather(feather_categories_file_path)
    print(f'Data saved to "{feather_categories_file_name}"')

    # featherファイルから再読み込み
    df_coins = pd.read_feather(feather_coins_file_path)
    df_categories = pd.read_feather(feather_categories_file_path)


# In[ ]:


# 2つのplatforms列が等しくなければ警告を出す
df_mismatch_platforms = df_coins[
    df_coins.apply(lambda x: list(x["coinPlatforms"]) != list(x["platforms"]), axis=1)
]
if not df_mismatch_platforms.empty:
    print(
        'Warning: There are rows with unequal lists in columns "coinPlatforms" and "platforms"'
    )


# In[ ]:


# assetPlatformIdがcoinPlatformsの最初の要素と等しくなければ警告を出す
df_coins["1stCoinPlatform"] = df_coins["coinPlatforms"].apply(
    lambda x: x[0] if len(x) > 0 else np.nan
)
df_coins_filtered = df_coins.dropna(subset=["assetPlatformId"])
df_mismatch_platform = df_coins_filtered[
    df_coins_filtered["assetPlatformId"] != df_coins_filtered["1stCoinPlatform"]
]
if not df_mismatch_platform.empty:
    print(
        'Warning: There are rows where "assetPlatformId" is not equal to the first element of "coinPlatforms"'
    )


# In[ ]:


# 時価総額のある仮想通貨を抽出
df_coins = df_coins.dropna(subset=["coinCap"])
df_coins = df_coins[df_coins["coinCap"] > 0]
df_coins = df_coins.sort_values(by=["coinCap"], ascending=[False]).reset_index(
    drop=True
)


# In[ ]:


# 仮想通貨データをExcelに出力する
df_xlsx = df_coins.copy()
for col in df_xlsx.select_dtypes(include=["datetimetz"]).columns:
    df_xlsx[col] = df_xlsx[col].dt.tz_localize(None)
xlsx_file_name = "df_coins.xlsx"
xlsx_file_path = os.path.join(data_dir, xlsx_file_name)
df_xlsx.to_excel(xlsx_file_path, sheet_name="df_coins", index=False)


# In[ ]:


# 時価総額のあるカテゴリを抽出
df_categories = df_categories.dropna(subset=["categoryCap"])
df_categories = df_categories[df_categories["categoryCap"] > 0]
df_categories = df_categories.sort_values(
    by=["categoryCap"], ascending=[False]
).reset_index(drop=True)


# In[ ]:


# カテゴリデータをExcelに出力する
df_xlsx = df_categories.copy()
for col in df_xlsx.select_dtypes(include=["datetimetz"]).columns:
    df_xlsx[col] = df_xlsx[col].dt.tz_localize(None)
xlsx_file_name = "df_categories.xlsx"
xlsx_file_path = os.path.join(data_dir, xlsx_file_name)
df_xlsx.to_excel(xlsx_file_path, sheet_name="df_categories", index=False)


# In[ ]:


df_coins_exploded = df_coins.explode("categories")
df_coins_exploded = df_coins_exploded.dropna(subset=["categories"]).reset_index(
    drop=True
)
df_coins_exploded = df_coins_exploded.rename(columns={"categories": "categoryName"})
df_coins_exploded = pd.merge(
    df_coins_exploded, df_categories, on=["categoryName"], how="outer", indicator=True
)

diff = df_coins_exploded[df_coins_exploded["_merge"] == "left_only"]
l_only_coin_categories = diff["categoryName"].unique()

diff = df_coins_exploded[df_coins_exploded["_merge"] == "right_only"]
l_only_category_list = diff["categoryName"].unique()

