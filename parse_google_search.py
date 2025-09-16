import csv
import time
import pandas as pd
import json
from apify_client import ApifyClient
import numpy as np
import requests

def create_responses() -> list:
    response = pd.read_csv('response.csv', header=None).iloc[:, 0].values
    cities = pd.read_csv('cities.csv', header=None).iloc[:, 0].values
    responses = np.array([[x, y] for x in response for y in cities])
    return responses

def create_dataset(response:str) -> str:
    with open("token.txt", "r") as file:
        token = file.read().strip()
    client = ApifyClient(token)
    run_input = {
    "focusOnPaidAds": False,
    "countryCode": "uz",
    "languageCode": "ru",
    "forceExactMatch": False,
    "includeIcons": False,
    "includeUnfilteredResults": False,
    "maxPagesPerQuery": 3,
    "mobileResults": False,
    "queries": response,
    "resultsPerPage": 100,
    "saveHtml": False,
    "saveHtmlToKeyValueStore": False
    }
    run = client.actor("apify/google-search-scraper").call(run_input=run_input)
    dataset_id = run['defaultDatasetId']
    url = f"https://api.apify.com/v2/datasets/{dataset_id}/items?format=json&fields=searchQuery,organicResults&unwind=organicResults"
    return url

def parse_result(url: str) -> pd.DataFrame:
    with requests.get(url) as response:
        json_data = response.json()

    field_mapings = {
        "position": lambda item: item.get("position", None),
        "title": lambda item: item.get("title", None),
        "description": lambda item: item.get("description", None),
        "url": lambda item: item.get("url", None),
        "keywords": lambda item: " ".join(item.get("emphasizedKeywords", [])) if isinstance(
            item.get("emphasizedKeywords"), list) else None

    }

    df_dict = {field: [maping(item) for item in json_data] for field, maping in field_mapings.items()}
    return pd.DataFrame(df_dict)

def make_response(response: str) -> pd.DataFrame:
    query, city = response
    link_result = create_dataset(f"{query} {city}")
    df = parse_result(link_result)
    df['query'] = query
    df['city'] = city
    return df


def main():
    responses = create_responses()
    print(f"Total responses: {len(responses)}")
    dataframe = pd.DataFrame()
    failed_responses = []
    for i, response in enumerate(responses):
        try:
            df = make_response(response)
            dataframe = pd.concat([dataframe, df], ignore_index=True)
            print(f"Response {i} has been ended successfully: {response}")
            if i % 10 == 0 and i != 0:
                dataframe.to_csv("data_frame_csv/partial_output.csv", index=False, sep="|", encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
        except requests.exceptions.HTTPError as e:
            failed_responses.append(response)
            print(f"HTTP error occurred: {e}")
    dataframe.to_csv(f"data_frame_csv/crude_base_uz.csv", index=False, sep="|", encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
    print(failed_responses)


if __name__ == "__main__":
    main()