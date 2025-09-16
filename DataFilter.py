import csv
import re
import pandas as pd
import urllib.parse
import logging

class DataFilter:
    def __init__(self, df: pd.DataFrame = None, red_flags: object = {}, csv_path: str = None, locale: str = None, green_flags: object = {}):
        self.df = pd.DataFrame()
        if df is not None:
            self.df = df
        elif csv_path is not None:
            self.df = pd.read_csv(csv_path, sep="|")

        self.removed = pd.DataFrame(columns=self.df.columns)
        self.locale = locale

        self.bad_domains = [
            r"\b(?:google|yandex|bing|mail|yahoo|facebook|instagram|threads)\b",
            r"\b(?:2gis|ozon|vk|youtube|tiktok|twitter|whatsapp|telegram|aliexpress|avito|wildberries)\b",
            r"\b(?:ebay|amazon|etsy|alibaba|craigslist|craiglist|gumtree|yandex\.market|kaspi|otvertka|all\.biz)\b",
            r"\b(?:kolesa|olx|drom|zoon|flagma|cdek|prom|rabota|hh|satu|yellowpages|goldpages|allbiz|optoviki|factories)\b",
            r"\b(?:ostin|maag|zara|lamoda|asos|ikea|mvideo|eldorado|technopark|dns)\b",
            r"b(?:media|news|travel|blog|wiki|forum|gov)",
            r"\b(?:sber(?:bank)?|tinkoff|vtb|gazprombank|alfa|rosbank|raiffeisen|unicredit|homecredit|pochta|kurs)\b"
            r"\b(?:(files|docs|images|media|static|cdn|assets|content|download|archive|backup|storage)\.(?:ru|ua|com))\b",
        ]

        self.bad_title = [
            r"\b(страница\s+не\s+найдена|404|ошибка|доступ\s+запрещён|not\s+found)\b",
            r"\b(карта\s+сайта|site\s+map)\b",
            r"\b(каталог|список|товаров|услуг|предприятий|компаний)\b",
            r"\b(курс[а]?\s+валют|свежие\s+данные|топ-?\w+)\b"
            r"\b(официальн(ый|ые|ая))"
        ]

        self.bad_description = [
            r"^\s*\d+\s*(дн|день|дней|час|часов|минут|мин|секунд)"
        ]

        self._add_filters_from_red_flags(red_flags)
        self._compile_regex_patterns()
        self.logger = self._setup_logger()
        self.green_flags = green_flags

    def _add_filters_from_red_flags(self, red_flags):
        mapping = {
            "url": self.bad_domains,
            "title": self.bad_title,
            "description": self.bad_description
        }

        for key, target_list in mapping.items():
            value = red_flags.get(key)
            if value:
                if isinstance(value, list):
                    target_list.extend(value)
                elif isinstance(value, str):
                    target_list.append(value)

    def _compile_regex_patterns(self):
        self.domain_pattern = re.compile('|'.join(self.bad_domains), re.IGNORECASE)
        self.title_pattern = re.compile('|'.join(self.bad_title), re.IGNORECASE)
        self.description_pattern = re.compile('|'.join(self.bad_description), re.IGNORECASE)

    def _setup_logger(self):
        logger = logging.getLogger('DataFilter')
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _log_removed(self, original_df):
        removed = pd.concat([original_df, self.df]).drop_duplicates(keep=False)
        self.removed = pd.concat([self.removed, removed])

    def __clean_url(self, url: str) -> str:
        if not isinstance(url, str):
            return ""
        url = urllib.parse.unquote(url.strip().lower())
        parsed = urllib.parse.urlparse(url)

        # Пропускаем невалидные схемы
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            return ""

        cleaned_url = urllib.parse.urlunparse((
            parsed.scheme,
            parsed.netloc.replace("www.", ""),
            parsed.path.rstrip("/"),
            '', '', ''
        ))
        cleaned_url = re.sub(r"\s+", "", cleaned_url)
        cleaned_url = re.sub(r"[\u200B-\u200D\uFEFF]", "", cleaned_url)
        return cleaned_url

    def drop_duplicates(self):
        initial_count = len(self.df)
        self.df.drop_duplicates(subset=["url"], inplace=True)
        self.logger.info(f"Duplicate removal: removed {initial_count - len(self.df)} records")
        return self

    def filter_url(self):
        original_df = self.df.copy()
        self.df["url"] = self.df["url"].astype(str).apply(self.__clean_url)
        self.df.dropna(subset=["url"], inplace=True)
        self.df = self.df[self.df["url"] != ""]

        if self.locale:
            self.df = self.df[self.df["url"].str.contains(self.locale, na=False)]

        self.df = self.df[~self.df["url"].str.contains(self.domain_pattern, na=False)]
        self._log_removed(original_df)
        self.logger.info(f"URL filtering: removed {len(original_df) - len(self.df)} records")
        return self

    def filter_title(self):
        original_df = self.df.copy()
        self.df["title"] = self.df["title"].fillna("")
        self.df = self.df[~self.df["title"].str.contains(self.title_pattern, na=False)]
        self._log_removed(original_df)
        self.logger.info(f"Title filtering: removed {len(original_df) - len(self.df)} records")
        return self

    def filter_description(self):
        original_df = self.df.copy()
        self.df["description"] = self.df["description"].fillna("")
        self.df = self.df[~self.df["description"].str.contains(self.description_pattern, na=False)]
        self._log_removed(original_df)
        self.logger.info(f"Description filtering: removed {len(original_df) - len(self.df)} records")
        return self

    def filter_with_custom_regex(self, column: str, pattern: str):
        if column not in self.df.columns:
            self.logger.warning(f"Column '{column}' not found.")
            return self
        original_df = self.df.copy()
        self.df[column] = self.df[column].fillna("")
        compiled = re.compile(pattern, re.IGNORECASE)
        self.df = self.df[~self.df[column].astype(str).str.contains(compiled, na=False)]
        self._log_removed(original_df)
        self.logger.info(f"Custom regex filtering on '{column}': removed {len(original_df) - len(self.df)} records")
        return self


    def get_green_flags_count(self):
        if not self.green_flags:
            self.logger.warning("No green flags defined.")
            return 0

        self.df["green_flags_count"] = 0
        total_count = 0

        for column, patterns in self.green_flags.items():
            if column not in self.df.columns:
                self.logger.warning(f"Column '{column}' not found.")
                continue
            compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for pattern in compiled_patterns:
                matches = self.df[column].astype(str).str.contains(pattern, na=False)
                self.df["green_flags_count"] += matches.astype(int)
                pattern_count = matches.sum()
                total_count += pattern_count
                self.logger.debug(f"Patter '{pattern.pattern}' found {pattern_count} matches in column '{column}'")
        self.df = self.df.sort_values(by="green_flags_count", ascending=False)
        self.logger.info(f"Total green flags found: {total_count}")

    def apply_all(self):
        self.logger.info(f"Starting filtering with {len(self.df)} records")
        return (
            self.drop_duplicates()
                .filter_url()
                .filter_title()
                .filter_description()
                .df
        )

    def save_to_csv(self, path: str):
        self.df.to_csv(path, sep="|", index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
        self.logger.info(f"Saved filtered data to {path} ({len(self.df)} records)")

    def save_removed_to_csv(self, path: str):
        if not self.removed.empty:
            self.removed.to_csv(path, sep="|", index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
            self.logger.info(f"Saved removed records to {path} ({len(self.removed)} records)")

def main():
    df = pd.read_csv("data_frame_csv/filtered_output.csv", sep="|")
    print(f"Initial data loaded with {df.shape[0]} records.")

    red_flags = {
        "url": [
            r"\.(?:ru|ua)\b",  # объединил .ru и .ua
            r"\/\/top\.",
            r"\/\/gigal\.",
            r"\/\/market.sello\.",
            r"\/\/yopt\.",
            r"report\.kg",
            r"uzbekistan\.mfa\.gov",
            r"wikimapia",
            r"uzfoodexpo",
            r"sprav\.uz",
            r"(?:com|jp)/ru\b",
            r"bizkim\.uz",
            r"\/\/mover\.uz",
            r"\/\/xitoydan\.uz",
            r"\/\/lenta\.com",
            r"\/\/evu\.uz",
            r"uz\.bizorg\.su",
            r"\/\/tovar",
            r"news",
            r"\/\/orzon",
            r"\/\/uzdaily",
            r"\/\/tutto",
            r"\/\/agrobaza",
            r"\/\/uzum",
            r"\/\/biotus",
            r"\/\/glotr",
            r"\/\/lochin",
        ],
        "title": [
            r"(?i)\bсеть кондитерских\w*",
            r"(?i)\bдетск\w*",
            r"(?i)госкомстат",
            r"(?i)омское",
            r"(?i)линия по производству",
            r"(?i)\b(?:охот|рыбал)\w*",
            r"(?i)\bигр\w*",
            r"(?i)\bваканси\w*",
            r"(?i)\b(?:спецодежд|спецобув)\w*",
            r"(?i)\bхим\w*",
            r"(?i)\b(?:секонд[ -]?хенд|second[ -]?hand)\b",
            r"(?i)\bмебел\w*",
            r"(?i)\bвыставк\w*",
            r"(?i)\bматериал\w*",
            r"(?i)Поиск по запросу",
            r"(?i)\bпрокат\w*",
            r"(?i)\bстроител\w*",
            r"(?i)оборудование для произво",
            r"(?i)\bновост[ейя]\w*",
            r"(?i)\bновостро[йек]\w*",
            r"(?i)\bсеть супермаркетов"

        ],
        "description": [
            r"(?i)\bсеть кондитерских",
            r"(?i)\bоплат\w*",
            r"(?i)\b(?:из|в)\s+росси[яию]\b",
            r"(?i)\bваканси\w*",
            r"(?i)линия по производству",
            r"(?i)\b(?:канцелярия|канцтовар\w*)",
            r"(?i)\bортопедическ(?:ая|их|ое)",
            r"(?i)\bдиллер\w*\b",
            r"(?i)\bсписок\s+(?:поставщик\w*|компани\w*|фирм\w*|магазин\w*|сайт\w*|ресурс\w*)",
            r"(?i)\bсписок организаций",
            r"(?i)\bаптек[ае]\w*",
            r"(?i)\bукраин[аы]\w*",
            r"(?i)\bказахстан\w*",
            r"(?i)\bоборудование для произво",
            r"(?i)\bпроизводители\s*(?:и|,)?\s*поставщики\s+оборудовани[яеё]\w*",
            r"(?i)\bсправочная\s+информация\b",
            r"(?i)\bсеть супермаркетов",
            r"(?i)рф",
            r"^\d{1,2}\s[а-яё]{3,8}\b",  # даты типа "10 июня"
            r"(?i)\bпроизводители\s*(?:и|,|\s+)?поставщики\s+оборудовани[яеё]\w*"
        ]
    }

    green_flags = {
        # "url": [
        #     r"\b(?:optovik|optoviki|opto)\b",
        #     r"\b(?:wholesale|wholesaler|wholesalers)\b",
        #     r"\b(?:distributor|distributors)\b",
        #     r"\b(?:supplier|suppliers)\b"
        # ],
        # "title": [
        #     r"\b(?:оптовый\s+продавец|оптовая\s+компания|оптовая\s+фирма)\b",
        #     r"\b(?:дистрибьютор|дистрибьюторы)\b",
        #     r"\b(?:оптовые\s+поставки|оптовые\s+поставщики)\b"
        #     r"\bмагазин\w*",
        # ],
        # "description": [
        #     r"\b(?:оптовые\s+цены|оптовые\s+условия)\b",
        #     r"\b(?:оптовая\s+продажа|оптовая\s+продажи)\b"
        # ]
    }
    data_filter = DataFilter(df, red_flags=red_flags)
    filtered_df = data_filter.apply_all()
    data_filter_2 = DataFilter(filtered_df, green_flags=green_flags)
    data_filter_2.save_to_csv("data_frame_csv/filtered_output.csv")

if __name__ == "__main__":
    main()







