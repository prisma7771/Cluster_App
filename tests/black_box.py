import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import time
import os
import shutil


screenshot_folder = (
    r"E:\Programming\Python\JupyterNotebook\Clustering\Tugas_Akhir\App\tests\ss"
)


def clear_folder(folder_path):
    """Clear out the folder by removing all files and subfolders"""
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)


clear_folder(screenshot_folder)


@pytest.fixture(scope="module")
def setup_teardown():
    chrome_options = Options()
    chrome_options.add_argument("--ignore-certificate-errors")

    driver = webdriver.Chrome(options=chrome_options)
    driver.get("http://localhost:8501")
    driver.maximize_window()
    yield driver

    driver.quit()


def capture_screenshot(driver, step_name):
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    screenshot_name = os.path.join(
        screenshot_folder, f"screenshot_{step_name}_{timestamp}.png"
    )
    driver.save_screenshot(screenshot_name)


def test_components(setup_teardown):
    driver = setup_teardown
    title = driver.title
    assert title == "123190048_ClusterApp"

    driver.implicitly_wait(3)
    time.sleep(2)
    capture_screenshot(driver, "homepage")


def test_homepage(setup_teardown):
    driver = setup_teardown

    radio_option1 = driver.find_element(
        By.XPATH,
        "//div[@id='root']/div/div/div/div/div/section/div/div[3]/div/div/div/div/div/div/div/label[2]/div[2]/div/p",
    )
    radio_option1.click()
    time.sleep(2)
    capture_screenshot(driver, "preview")
    time.sleep(1)
    assert "FAIL To click Preview Button"

    radio_option2 = driver.find_element(
        By.XPATH,
        "//div[@id='root']/div/div/div/div/div/section/div/div[3]/div/div/div/div/div/div/div/label[3]/div[2]/div/p",
    )
    radio_option2.click()
    time.sleep(2)
    capture_screenshot(driver, "Firefly_PCA")
    time.sleep(1)
    assert "FAIL To click Firefly PCA Button"

    radio_option3 = driver.find_element(
        By.XPATH,
        "//div[@id='root']/div/div/div/div/div/section/div/div[3]/div/div/div/div/div/div/div/label[4]/div[2]/div/p",
    )
    radio_option3.click()
    time.sleep(2)
    capture_screenshot(driver, "eda")
    time.sleep(1)
    assert "FAIL To click EDA Button"


def test_pre_process(setup_teardown):
    driver = setup_teardown

    menu_option = driver.find_element(
        By.XPATH,
        "//div[@id='root']/div/div/div/div/div/section/div/div[2]/ul/li[2]/div/a/span ",
    )
    menu_option.click()
    time.sleep(2)
    capture_screenshot(driver, "Pre_process")
    time.sleep(1)
    assert "FAIL To click Pre-Process Button"


def test_comparison(setup_teardown):
    driver = setup_teardown

    menu_option = driver.find_element(
        By.XPATH,
        "//div[@id='root']/div/div/div/div/div/section/div/div[2]/ul/li[3]/div/a/span ",
    )
    menu_option.click()
    time.sleep(2)
    capture_screenshot(driver, "Comparison")
    time.sleep(1)
    assert "FAIL To click Comparison Button"


def test_cluster(setup_teardown):
    driver = setup_teardown

    menu_option = driver.find_element(
        By.XPATH,
        "//div[@id='root']/div/div/div/div/div/section/div/div[2]/ul/li[4]/div/a/span ",
    )
    menu_option.click()
    time.sleep(2)
    capture_screenshot(driver, "Cluster")
    time.sleep(1)
    assert "FAIL To click Cluster Button"


def test_input_file(setup_teardown):
    driver = setup_teardown

    file_input = driver.find_element(By.XPATH, "//input[@type='file']")
    file_path = r"E:\Programming\Python\JupyterNotebook\Clustering\Tugas_Akhir\App\sample\sample10_data.csv"
    file_input.send_keys(file_path)
    time.sleep(3)

    predict_btn = driver.find_element(
        By.XPATH,
        "//div[@id='root']/div/div/div/div/div/section[2]/div/div/div/div/div[5]/div/button/div/p",
    )

    predict_btn.click()
    time.sleep(2)
    capture_screenshot(driver, "Input")
    time.sleep(1)

    assert "FAIL To click Cluster Button"


@pytest.mark.xfail
def test_input_file2(setup_teardown):
    driver = setup_teardown

    file_input2 = driver.find_element(By.XPATH, "//input[@type='file']")
    file_path = r"E:\Programming\Python\JupyterNotebook\Clustering\Tugas_Akhir\App\sample\nonvalid_data.csv"
    file_input2.send_keys(file_path)
    time.sleep(2)
    capture_screenshot(driver, "Nonvalid_Input")
    time.sleep(1)
    assert "NOT Supposed to Success To click Cluster Button"


def test_manual_input(setup_teardown):
    driver = setup_teardown

    change_option = driver.find_element(
        By.XPATH,
        "//div[@id='root']/div/div/div/div/div/section/div/div[3]/div/div/div/div/div[2]/div/div/div/div",
    )
    change_option.click()

    manual_option = WebDriverWait(driver, 10).until(
        EC.visibility_of_element_located(
            (By.XPATH, "(//div[contains(@id, '__anchor')]/div/div)[2]")
        )
    )
    manual_option.click()
    assert manual_option

    select_coffee = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable(
            (
                By.XPATH,
                "//*[normalize-space(text())='Cappucinno']/following::*[name()='svg'][1]",
            )
        )
    )
    time.sleep(2)
    select_coffee.click()

    vietnam_drip_element = WebDriverWait(driver, 10).until(
        EC.visibility_of_element_located(
            (By.XPATH, "//*[normalize-space(text())='Vietnam Drip']")
        )
    )
    vietnam_drip_element.click()

    vietnam_drip_add = driver.find_element(
        By.XPATH,
        "//div[@id='root']/div/div/div/div/div/section[2]/div/div/div/div/div[3]/div/div/div/div/div[2]/div/div/div[2]/button[2]",
    )
    time.sleep(2)
    vietnam_drip_add.click()
    vietnam_drip_add.click()

    vietnam_drip_btn = driver.find_element(
        By.XPATH,
        "//div[@id='root']/div/div/div/div/div/section[2]/div/div/div/div/div[3]/div[2]/div/div/div/div[3]/div/button/div/p",
    )
    vietnam_drip_btn.click()
    time.sleep(2)
    capture_screenshot(driver, "Item_Add")
    time.sleep(1)

    predict_btn = driver.find_element(
        By.XPATH,
        "//div[@id='root']/div/div/div/div/div/section[2]/div/div/div/div/div[10]/div/button/div/p",
    )
    predict_btn.click()
    time.sleep(2)
    capture_screenshot(driver, "Predict_Manual")
    time.sleep(1)


def test_clear(setup_teardown):
    driver = setup_teardown

    clear_btn = driver.find_element(
        By.XPATH,
        "//div[@id='root']/div/div/div/div/div/section[2]/div/div/div/div/div[6]/div/button/div/p",
    )
    clear_btn.click()
    time.sleep(2)
    capture_screenshot(driver, "Clear")
    time.sleep(1)
