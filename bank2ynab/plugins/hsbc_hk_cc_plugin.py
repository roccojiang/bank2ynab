import logging
from datetime import date, datetime
from typing import Optional

import bank_handler
import numpy as np
import pandas as pd
import pdfplumber
from bank_handler import BankHandler
from pdfplumber.pdf import PDF

Table = list[list[Optional[str]]]


class HsbcHkCreditCardPlugin(BankHandler):
    """
    Plugin for extracting a CSV table from HSBC Hong Kong credit card statements in PDF format.
    """

    PDFPLUMBER_TABLE_SETTINGS = {
        "horizontal_strategy": "text",
        "explicit_vertical_lines": [260, 330, 350, 380],
        "intersection_x_tolerance": 25,  # right column text is not always flush with vertical line
    }
    COLUMNS = [
        "post_date",
        "trans_date",
        "payee",
        "location",
        "country",
        "original_currency",
        "original_amount",
        "hkd_amount",
    ]

    STATEMENT_DATE_BOX_HEIGHT = 20
    STATEMENT_DATE_BOX_RIGHT_MARGIN = 40
    STATEMENT_DATE_BOX_LEFT_MARGIN = 20

    def __init__(self, config_dict: dict):
        super().__init__(config_dict)
        self.name = "HSBC Hong Kong Credit Card"

    def _preprocess_file(self, file_path: str, plugin_args: list) -> str:
        """
        Combines all tables in a PDF file into one table and writes to CSV.

        :param file_path: path to PDF file
        :type file_path: str
        :param plugin_args: plugin arguments (unused in this plugin)
        :type plugin_args: list
        :return: path to CSV file
        :rtype: str
        """

        logging.info("Converting PDF file...")

        # create dataframe from pdf
        df = self.read_pdf_to_dataframe(
            pdf_path=file_path, table_cols=self.COLUMNS
        )
        # generate output path
        new_path = bank_handler.get_output_path(
            input_path=file_path,
            prefix=f"converted_pdf_{self.config_dict['bank_name']}_",
            ext=".csv",
        )
        # write the dataframe to output file
        df.to_csv(new_path, index=False)
        logging.info("\tFinished converting PDF file.")
        return new_path

    def read_pdf_to_dataframe(
        self, pdf_path: str, table_cols: list[str]
    ) -> pd.DataFrame:
        """
        Reads the main table from each page of the statement PDF,
        processing and combining them into a single dataframe.
        An error is thrown if a table does not have the right number of columns.

        :param pdf_path: filepath for PDF file
        :type pdf_path: str
        :param table_cols: columns to use for dataframe
        :type table_cols: list[str]
        :return: processed dataframe of combined tables
        :rtype: pd.DataFrame
        """
        pdf = pdfplumber.open(pdf_path)

        logging.info("\tExtracting tables and statement date.")
        tables, statement_date = self.extract_data(pdf, table_cols)

        logging.info("\tCleaning and processing table data.")
        flattened_table = self.preprocess_table(tables)
        df = self.convert_to_dataframe(flattened_table, table_cols)
        processed_df = self.process_dataframe(df, statement_date)
        return processed_df

    def extract_data(
        self, pdf: PDF, table_cols: list[str]
    ) -> tuple[list[Table], date]:
        statement_date_str = ""
        tables = []
        for page_num, page in enumerate(pdf.pages, start=1):
            if not statement_date_str:
                statement_date_str = self.extract_statement_date_str(page)

            top = (
                0
                if not page.search("Post date")
                else page.search("Post date")[0]["top"]
            )
            bottom = (
                page.height
                if not page.search("Minimum payment summary")
                else page.search("Minimum payment summary")[0]["top"]
            )
            bbox = (0, top, page.width, bottom)

            extracted_page = page.crop(bbox).extract_table(
                self.PDFPLUMBER_TABLE_SETTINGS
            )
            if extracted_page:
                if len(extracted_page[0]) != len(table_cols):
                    raise TableSizeError(
                        f"Table extracted from page {page_num} has {len(extracted_page[0])} columns, expected {len(table_cols)}."
                    )
                tables.append(extracted_page)

        if not statement_date_str:
            logging.warning(
                "Statement date not found, defaulting to using today's date."
            )
            statement_date = date.today()
        else:
            statement_date = datetime.strptime(
                statement_date_str, "%d %b %Y"
            ).date()

        return tables, statement_date

    def preprocess_table(self, table: list[Table]) -> Table:
        # Flatten table and drop header row from each page
        return [row for page in table for row in page[1:]]

    def convert_to_dataframe(
        self, table: Table, table_cols: list[str]
    ) -> pd.DataFrame:
        df = (
            pd.DataFrame(table, columns=table_cols)
            .replace("", np.nan)
            .dropna(how="all", ignore_index=True)
            .fillna("")
        )

        if df.empty:
            logging.warning("No transactions found in the extracted table.")

        return df

    def process_dataframe(
        self, df: pd.DataFrame, statement_date: date
    ) -> pd.DataFrame:
        # Drop previous balance information - begin with first transaction
        df = drop_until(
            df, lambda df: (df["post_date"] != "") | (df["trans_date"] != "")
        )
        # Drop footer information
        df = take_until(
            df,
            lambda df: df.apply(
                lambda x: "".join(x).replace(" ", ""), axis=1
            ).str.contains(r"\*Forcreditcardtransactionseffectedincurrencies"),
        )

        # Merge rows with same transaction
        df = (
            df.groupby(
                ((df["post_date"] != "") | (df["trans_date"] != "")).cumsum()
            )
            .agg({col: "first" for col in df.columns} | {"payee": list})
            .reset_index(drop=True)
        )

        # Populate memo field
        df["memo"] = df.apply(
            lambda x: self.create_memo(
                x["payee"],
                x["location"],
                x["country"],
                x["original_currency"],
                x["original_amount"],
            ),
            axis=1,
        )
        df["payee"] = df["payee"].apply(lambda x: x[0])

        # Add year to transaction dates, based on statement date
        df[["post_date", "trans_date"]] = df[
            ["post_date", "trans_date"]
        ].applymap(
            self.parse_and_add_year_to_date,
            output_date_format=self.config_dict["date_format"],
            statement_date=statement_date,
        )

        # Convert hkd_amount column to expected inflow format
        df["hkd_amount"] = df["hkd_amount"].apply(self.add_sign_to_transaction)

        return df

    def extract_statement_date_str(self, page) -> str:
        date_box_areas = page.search("Statement date")
        if date_box_areas:
            date_box = date_box_areas[0]
            return (
                page.within_bbox(
                    (
                        date_box["x0"] - self.STATEMENT_DATE_BOX_LEFT_MARGIN,
                        date_box["bottom"],
                        date_box["x1"] + self.STATEMENT_DATE_BOX_RIGHT_MARGIN,
                        date_box["bottom"] + self.STATEMENT_DATE_BOX_HEIGHT,
                    )
                )
                .extract_text()
                .strip()
            )
        return ""

    def create_memo(
        self,
        payee_info: list[str],
        location: str,
        country: str,
        original_currency: str,
        original_amount: str,
    ) -> str:
        memo = []

        if location:
            memo.append(
                f"Location: {location}" + (f", {country}" if country else "")
            )
        elif country:
            memo.append(f"Country: {country}")

        if original_currency:
            assert (
                original_amount
            ), "A foreign currency symbol exists, but no corresponding amount"
            memo.append(
                f"Original amount: {original_amount} {original_currency}"
            )

        if len(payee_info) > 1:
            info = ", ".join(x.strip() for x in payee_info[1:])
            memo.append(f"Details: {info}")

        memo_str = "; ".join(memo)
        import_info_text = "[Imported from PDF statement]"
        return (
            memo_str + " // " + import_info_text
            if memo_str
            else import_info_text
        )

    def parse_and_add_year_to_date(
        self,
        date_string: str,
        statement_date: date,
        input_date_format: str = "%d%b",
        output_date_format: str = "%d/%m/%Y",
    ) -> str:
        parsed_date = datetime.strptime(date_string, input_date_format).date()
        if parsed_date.month == 12 and statement_date.month == 1:
            return parsed_date.replace(year=statement_date.year - 1).strftime(
                output_date_format
            )
        return parsed_date.replace(year=statement_date.year).strftime(
            output_date_format
        )

    def add_sign_to_transaction(self, amount: str) -> str:
        if amount[-2:] == "CR":
            return amount[:-2]  # inflow
        return "-" + amount  # outflow


class TableSizeError(Exception):
    pass


def build_bank(config):
    return HsbcHkCreditCardPlugin(config)


def drop_until(df: pd.DataFrame, predicate, *args, **kwargs) -> pd.DataFrame:
    return df[predicate(df, *args, **kwargs).idxmax() :].reset_index(drop=True)


def take_until(df: pd.DataFrame, predicate, *args, **kwargs) -> pd.DataFrame:
    return df[: predicate(df, *args, **kwargs).idxmax()].reset_index(drop=True)
