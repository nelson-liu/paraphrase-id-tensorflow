import argparse
import csv
import logging
import re
import os


logger = logging.getLogger(__name__)


def main():
    argparser = argparse.ArgumentParser(description=("Clean the Quora dataset "
                                                     "by removing newlines in "
                                                     "the data."))
    argparser.add_argument("dataset_input_path", type=str,
                           help=("The path to the raw Quora "
                                 "dataset to clean."))
    argparser.add_argument("dataset_output_path", type=str,
                           help=("The *folder* to write the "
                                 "cleaned file to. The name will just have "
                                 "_cleaned appended to it, before the "
                                 "extension"))
    config = argparser.parse_args()

    # Get the data
    logger.info("Reading csv at {}".format(config.dataset_input_path))

    # Iterate through the CSV, removing anomalous whitespace
    # and making a list of lists the clean csv.
    logger.info("Cleaning csv")
    clean_rows = []
    with open(config.dataset_input_path) as f:
        reader = csv.reader(f)
        # skip the header
        reader.__next__()
        for row in reader:
            clean_row = []
            for item in row:
                # normalize whitespace in each string in each row
                item_no_newlines = re.sub(r"\n", " ", item)
                clean_item = re.sub(r"\s+", " ", item_no_newlines)
                clean_row.append(clean_item)
            clean_rows.append(clean_row)

    input_filename_full = os.path.basename(config.dataset_input_path)
    input_filename, input_ext = os.path.splitext(input_filename_full)
    out_path = os.path.join(config.dataset_output_path,
                            input_filename + "_cleaned" + input_ext)

    logger.info("Writing output to {}".format(out_path))
    with open(out_path, "w") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerows(clean_rows)


if __name__ == "__main__":
    logging.basicConfig(format=("%(asctime)s - %(levelname)s - "
                                "%(name)s - %(message)s"),
                        level=logging.INFO)
    main()
