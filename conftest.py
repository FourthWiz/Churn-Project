import logging
import os

def pytest_configure(config):
    # Ensure the logs folder exists
    if not os.path.exists('logs'):
        os.makedirs('logs')

    # Configure logging
    logging.basicConfig(
        filename='logs/churn_library.log',
        level=logging.INFO,
        filemode='w',
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )