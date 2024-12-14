from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import os

# Save Screeshot
screenshot_folder = (
    r"E:\Programming\Python\JupyterNotebook\Clustering\Tugas_Akhir\App\tests\ss"
)

def test_eight_components():
    driver = setup()

    title = driver.title
    assert title == "123190048_ClusterApp"

    driver.implicitly_wait(0.5)
    time.sleep(3)

    teardown(driver)


def setup():
    driver = webdriver.Chrome()
    driver.get("https://localhost:8501")
    driver.maximize_window()
    return driver


def teardown(driver):
    driver.quit()


# Function to capture screenshot
def capture_screenshot(driver, step_name):
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    screenshot_name = os.path.join(screenshot_folder, f"screenshot_{step_name}_{timestamp}.png") 
    driver.save_screenshot(screenshot_name)
