import itertools
import logging
from datetime import date, datetime
from typing import Any, Optional

import bank_handler
import pandas as pd
import pdfplumber
from bank_handler import BankHandler
from pdfplumber.pdf import PDF

NullableTable = list[list[Optional[str]]]
Table = list[list[str]]


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

        logging.info("\tExtracting tables and the statement date.")
        tables, statement_date = self.extract_tables_and_statement_date(
            pdf, table_cols
        )

        logging.info("\tCleaning and processing table data.")
        flattened_table = self.flatten_and_clean_table(tables)
        processed_df = self.convert_and_process_dataframe(
            flattened_table, table_cols, statement_date
        )
        return processed_df

    def extract_tables_and_statement_date(
        self, pdf: PDF, table_cols: list[str]
    ) -> tuple[list[NullableTable], date]:
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

    def flatten_and_clean_table(self, table: list[NullableTable]) -> Table:
        # Flatten table and drop header row from each page
        flattened_table = [row for page in table for row in page[1:]]
        # Replace occurrences of None with empty strings
        flattened_table = [
            ["" if cell is None else cell for cell in row]
            for row in flattened_table
        ]
        # Drop rows with all empty cells
        flattened_table = [
            row for row in flattened_table if any(cell != "" for cell in row)
        ]

        # Drop previous balance info at the beginning of the statement, begin with first transaction
        cleaned_table = list(
            itertools.dropwhile(
                lambda row: row[:2] == ["", ""], flattened_table
            )
        )
        # Drop non-transaction info at the end of the statement
        cleaned_table = list(
            itertools.takewhile(
                lambda row: "*Forcreditcardtransactionseffectedincurrencies"
                not in "".join(row).replace(" ", ""),
                cleaned_table,
            )
        )
        return cleaned_table

    def convert_and_process_dataframe(
        self, table: Table, table_cols: list[str], statement_date: date
    ) -> pd.DataFrame:
        df = pd.DataFrame(table, columns=table_cols)

        if df.empty:
            logging.warning("No transactions found in the extracted table.")
            return df

        aggregators: dict[str, Any] = {col: "first" for col in df.columns}
        aggregators["payee"] = list

        # Merge rows with same transaction
        df = (
            df.groupby(
                ((df["post_date"] != "") | (df["trans_date"] != "")).cumsum()
            )
            .agg(aggregators)
            .reset_index(drop=True)
        )

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

        df[["post_date", "trans_date"]] = df[
            ["post_date", "trans_date"]
        ].applymap(
            self.parse_and_add_year_to_date,
            output_date_format=self.config_dict["date_format"],
            statement_date=statement_date,
        )

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
