import logging


def test_import():
    """Test data import - this example is completed for you to assist with the other test functions."""
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
        assert 1 == 0
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        print("dupa")
        # assert df.shape[0] > 0
        # assert df.shape[1] > 0
    except AssertionError as err:
        logging.error("Testing import_data: The file doesn't appear to have rows and columns")
        raise err
