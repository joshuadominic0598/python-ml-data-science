PYTHON := python3
VENV := .venv
VENV_PYTHON := $(VENV)/bin/python
VENV_PIP := $(VENV)/bin/pip
GDOWN := $(VENV)/bin/gdown

HOUSING_DATASET_DIR := statistics/housing_data_analysis/housing_dataset
HOUSE_PRICES_DATASET_DIR := regression/house_price_prediction/house_prices_dataset
CUSTOMER_PURCHASES_DATASET_DIR := classification/customer_purchases_dataset
CAR_PRICES_DATASET_DIR := regression/car_price_prediction/car_prices_dataset
NYC_311_CSV := statistics/service_requests_analysis/nyc_311_data.csv
MALL_CUSTOMERS_CSV := clustering/mall_customers.csv

HOUSING_DATASET_URL := https://drive.google.com/drive/folders/1a2HWdQdx8kc4mjfwirZMf-pPj9s9QP3r
HOUSE_PRICES_DATASET_URL := https://drive.google.com/drive/folders/1wzy788nUz0ZlCcBnj2Z4udeA06dXYV1a
CUSTOMER_PURCHASES_DATASET_URL := https://drive.google.com/drive/folders/1pmyO_IsuGwYy3Bkcela0TDPTD1q_Bf3z
CAR_PRICES_DATASET_URL := https://drive.google.com/drive/folders/1TCjBTu47g1htMyBeJdtmFcGtvRDsIT_Z
NYC_311_CSV_URL := https://drive.google.com/file/d/1NBhCray_O3H4id3CIkbZw8zfnkS9T_8M/view
MALL_CUSTOMERS_CSV_URL := https://drive.google.com/file/d/1gWsErcY9I4OH_nntV1Rm1hjWsJIj6ro5/view

# Every file expected on disk once all datasets have downloaded successfully.
EXPECTED_FILES := \
	$(HOUSING_DATASET_DIR)/housing.csv \
	$(HOUSE_PRICES_DATASET_DIR)/data_description.txt \
	$(HOUSE_PRICES_DATASET_DIR)/ground_truth/ground_truth.csv \
	$(HOUSE_PRICES_DATASET_DIR)/ground_truth/ground_truth_comparison.csv \
	$(HOUSE_PRICES_DATASET_DIR)/ground_truth/ground_truth_features.csv \
	$(HOUSE_PRICES_DATASET_DIR)/ground_truth/ground_truth_predictions.csv \
	$(HOUSE_PRICES_DATASET_DIR)/train/train.csv \
	$(HOUSE_PRICES_DATASET_DIR)/train/training_data.csv \
	$(HOUSE_PRICES_DATASET_DIR)/train/validation_data.csv \
	$(CUSTOMER_PURCHASES_DATASET_DIR)/cleaned_data.csv \
	$(CUSTOMER_PURCHASES_DATASET_DIR)/customer_details.csv \
	$(CAR_PRICES_DATASET_DIR)/ground_truth/ground_truth.csv \
	$(CAR_PRICES_DATASET_DIR)/ground_truth/ground_truth_comparison.csv \
	$(CAR_PRICES_DATASET_DIR)/ground_truth/ground_truth_features.csv \
	$(CAR_PRICES_DATASET_DIR)/ground_truth/ground_truth_predictions.csv \
	$(CAR_PRICES_DATASET_DIR)/train/car_price.csv \
	$(CAR_PRICES_DATASET_DIR)/train/training_data.csv \
	$(CAR_PRICES_DATASET_DIR)/train/validation_data.csv \
	$(NYC_311_CSV) \
	$(MALL_CUSTOMERS_CSV)

.PHONY: all setup venv install data verify-data clean

all: setup

setup: install data

venv:
	$(PYTHON) -m venv $(VENV)

install: venv
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install -r requirements.txt

data: install
	rm -rf $(HOUSING_DATASET_DIR) $(HOUSE_PRICES_DATASET_DIR) $(CUSTOMER_PURCHASES_DATASET_DIR) $(CAR_PRICES_DATASET_DIR)
	$(GDOWN) --folder -O $(HOUSING_DATASET_DIR) $(HOUSING_DATASET_URL)
	$(GDOWN) --folder -O $(HOUSE_PRICES_DATASET_DIR) $(HOUSE_PRICES_DATASET_URL)
	$(GDOWN) --folder -O $(CUSTOMER_PURCHASES_DATASET_DIR) $(CUSTOMER_PURCHASES_DATASET_URL)
	$(GDOWN) --folder -O $(CAR_PRICES_DATASET_DIR) $(CAR_PRICES_DATASET_URL)
	$(GDOWN) --fuzzy -O $(NYC_311_CSV) $(NYC_311_CSV_URL)
	$(GDOWN) --fuzzy -O $(MALL_CUSTOMERS_CSV) $(MALL_CUSTOMERS_CSV_URL)
	$(MAKE) verify-data

# Fails the build if any expected dataset file is missing, so a broken/partial
# Google Drive download never passes silently.
verify-data:
	@missing=0; \
	for f in $(EXPECTED_FILES); do \
		if [ ! -f "$$f" ]; then echo "MISSING: $$f"; missing=1; fi; \
	done; \
	if [ "$$missing" -eq 1 ]; then \
		echo "One or more dataset files failed to download. See MISSING lines above."; \
		exit 1; \
	fi; \
	echo "All $(words $(EXPECTED_FILES)) expected dataset files are present."

clean:
	rm -rf $(VENV)
