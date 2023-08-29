import itertools
import logging
from datetime import date, datetime

import bank_handler
import pandas as pd
import pdfplumber
from bank_handler import BankHandler


class HsbcHkCreditCardPlugin(BankHandler):
    PDFPLUMBER_TABLE_SETTINGS = {
        "horizontal_strategy": "text",
        "explicit_vertical_lines": [260, 330, 350, 380],
        "intersection_x_tolerance": 25,  # right column text is not always flush with vertical line
    }
    COLUMNS = [
        "post_date",
        "trans_date",
        "payee",
        "address",
        "country",
        "foreign_currency",
        "foreign_currency_amount",
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
        This is an example of how to preprocess the transaction file
        prior to feeding the data into the main read_data function.
        Any specialised string or format operations can easily
        be done here.
        """
        """
        For every row that doesn't have a valid date field
        strip out separators and append to preceding row.
        Overwrite input file with modified output.
        :param file_path: path to file
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
        pdf = pdfplumber.open(pdf_path)

        statement_date_text = ""
        table = []
        for page in pdf.pages:
            if not statement_date_text:
                statement_date_text_areas = page.search("Statement date")
                if statement_date_text_areas:
                    statement_date_box = statement_date_text_areas[0]
                    statement_date_text = (
                        page.within_bbox(
                            (
                                statement_date_box["x0"]
                                - self.STATEMENT_DATE_BOX_LEFT_MARGIN,
                                statement_date_box["bottom"],
                                statement_date_box["x1"]
                                + self.STATEMENT_DATE_BOX_RIGHT_MARGIN,
                                statement_date_box["bottom"]
                                + self.STATEMENT_DATE_BOX_HEIGHT,
                            )
                        )
                        .extract_text()
                        .strip()
                    )

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
            assert len(extracted_page[0]) == len(table_cols)
            table.append(extracted_page)

        statement_date = datetime.strptime(
            statement_date_text, "%d %b %Y"
        ).date()

        # Drop header row from each page
        flat_table = [row for page in table for row in page[1:]]
        # Replace occurrences of None with empty strings
        flat_table = [
            ["" if cell is None else cell for cell in row]
            for row in flat_table
        ]
        # Drop rows with all empty cells
        flat_table = [
            row for row in flat_table if any(cell != "" for cell in row)
        ]

        # Drop previous balance information, begin with first transaction
        flat_table = list(
            itertools.dropwhile(lambda row: row[:2] == ["", ""], flat_table)
        )
        # Drop footer information
        flat_table = list(
            itertools.takewhile(
                lambda row: "*Forcreditcardtransactionseffectedincurrencies"
                not in "".join(row[2:8]).replace(" ", ""),
                flat_table,
            )
        )

        df = pd.DataFrame(flat_table, columns=table_cols)

        aggregators = {col: "first" for col in df.columns}
        aggregators["payee"] = list

        df = (
            df.groupby(
                ((df["post_date"] != "") | (df["trans_date"] != "")).cumsum()
            )
            .agg(aggregators)
            .reset_index(drop=True)
        )

        df["memo"] = df["payee"].apply(lambda x: "\n".join(x[1:]))
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


def build_bank(config):
    return HsbcHkCreditCardPlugin(config)
