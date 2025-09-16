import requests, time, json
import pandas as pd

with open("token.txt", "r") as file:
    abstract_api = file.readline()


def validate(phone_number: str) -> tuple:
    response = requests.get(f"https://phonevalidation.abstractapi.com/v1/?api_key=48297372e15c45818930d5d70fb4b83f&phone={phone_number}")
    content =  json.loads(response.content)
    valid = content["valid"] or False
    country = content["country"]["code"] or None
    return valid, country

def check_phones(phone_numbers: str, country: str) -> str:
    phones = phone_numbers.split(", ")
    result = ""
    for phone in phones:
        cur_valid, cur_country = validate(phone)
        if not cur_valid or country != cur_country:
            if len(result) == 0:
                result += phone
            else:
                result += ", " + phone
    return result

def main():
    companies_base = pd.read_csv("database.csv").loc[:, ["base", "contact"]]


if __name__ == "__main__":
    main()
