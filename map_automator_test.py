from selenium import webdriver
# from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, WebDriverException
# from webdriver_manager.chrome import ChromeDriverManager
# from webdriver_manager.core.os_manager import ChromeType

from PIL import Image
import piexif
import os
import cv2
import math
import time
import logging


# =========================
# CONFIG
# =========================

OUTPUT_DIR = "maps_screenshots"

ZOOM_LEVEL = 18
WAIT_TIME = 2
PITCH = 60
HEIGHT = 120

PAGE_LOAD_TIMEOUT = 30
MAX_RETRIES = 10
RETRY_DELAY = 5

HEADLESS = True

START = (36.07530634254636, -79.77220753467735)
END = (36.07303311892077, -79.79110358622233)

LINE_SPACING_METERS = 50

failed_screenshots = []
is_first = True


# =========================
# LOGGING
# =========================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# =========================
# PATH HELPERS
# =========================

# =========================
# BRAVE DRIVER SETUP
# =========================

def find_brave_binary():
    possible_paths = [
        r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe",
        r"C:\Program Files (x86)\BraveSoftware\Brave-Browser\Application\brave.exe",
        os.path.expandvars(
            r"%LOCALAPPDATA%\BraveSoftware\Brave-Browser\Application\brave.exe"
        ),
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        "Could not find brave.exe. Open Brave, go to brave://version, "
        "copy the executable path, and put it in possible_paths."
    )


def create_brave_driver():
    brave_path = find_brave_binary()

    chrome_options = Options()
    chrome_options.binary_location = brave_path

    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--force-device-scale-factor=1")
    chrome_options.add_argument("--disable-web-security")
    chrome_options.add_argument("--disable-features=VizDisplayCompositor")
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")

    # IMPORTANT:
    # No Service(...)
    # No ChromeDriverManager(...)
    # Selenium Manager handles the driver automatically.
    driver = webdriver.Chrome(options=chrome_options)

    driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT)
    driver.implicitly_wait(10)

    return driver


driver = create_brave_driver()


def restart_driver():
    global driver, is_first

    logger.info("Restarting Brave WebDriver...")

    try:
        driver.quit()
    except Exception:
        pass

    driver = create_brave_driver()
    is_first = True

    logger.info("Brave WebDriver restarted successfully")


# =========================
# DISTANCE / HEADING HELPERS
# =========================

def haversine_distance_m(start, end):
    lat1, lon1 = map(math.radians, start)
    lat2, lon2 = map(math.radians, end)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    radius_m = 6371000

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )

    return 2 * radius_m * math.asin(math.sqrt(a))


def calculate_bearing_degrees(start, end):
    lat1, lon1 = map(math.radians, start)
    lat2, lon2 = map(math.radians, end)

    dlon = lon2 - lon1

    x = math.sin(dlon) * math.cos(lat2)
    y = (
        math.cos(lat1) * math.sin(lat2)
        - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    )

    bearing = math.degrees(math.atan2(x, y))
    return (bearing + 360) % 360


def generate_line_coordinates(start, end, spacing_meters):
    total_distance = haversine_distance_m(start, end)
    num_segments = max(1, math.ceil(total_distance / spacing_meters))

    heading = calculate_bearing_degrees(start, end)

    coordinates = []

    for i in range(num_segments + 1):
        t = i / num_segments

        lat = start[0] + t * (end[0] - start[0])
        lon = start[1] + t * (end[1] - start[1])

        coordinates.append((lat, lon, heading))

    return coordinates


GPS_COORDINATES = generate_line_coordinates(
    START,
    END,
    LINE_SPACING_METERS
)

print("START:", START)
print("END:", END)
print(f"Generated {len(GPS_COORDINATES)} line points")
print(f"Heading: {GPS_COORDINATES[0][2]:.2f} degrees")
print(f"Distance: {haversine_distance_m(START, END):.2f} meters")


# =========================
# BRAVE DRIVER SETUP
# =========================


driver = create_brave_driver()


# =========================
# IMAGE HELPERS
# =========================

def crop_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not read image for cropping: {image_path}")

    cropped_img = image[90:843, 400:1504]
    cv2.imwrite(image_path, cropped_img)


def decimal_to_dms_rational(value):
    value = abs(value)

    degrees = int(value)
    minutes_float = (value - degrees) * 60
    minutes = int(minutes_float)
    seconds = (minutes_float - minutes) * 60

    return [
        (degrees, 1),
        (minutes, 1),
        (int(seconds * 100), 100),
    ]


def add_gps_metadata(image_path, lat, lon):
    crop_image(image_path)

    img = Image.open(image_path)

    exif_dict = {
        "GPS": {
            piexif.GPSIFD.GPSLatitude: decimal_to_dms_rational(lat),
            piexif.GPSIFD.GPSLatitudeRef: "N" if lat >= 0 else "S",
            piexif.GPSIFD.GPSLongitude: decimal_to_dms_rational(lon),
            piexif.GPSIFD.GPSLongitudeRef: "E" if lon >= 0 else "W",
        }
    }

    exif_bytes = piexif.dump(exif_dict)
    img.save(image_path, exif=exif_bytes)


# =========================
# CAPTURE FUNCTION
# =========================

def capture_bing_map(lat, lon, heading, zoom):
    global is_first

    filename = f"{OUTPUT_DIR}/frame_{lat:.6f}_{lon:.6f}_{heading:.2f}.png"

    if os.path.exists(filename):
        logger.info(f"Screenshot already exists, skipping: {filename}")
        return True

    for attempt in range(MAX_RETRIES):
        try:
            logger.info(
                f"Attempt {attempt + 1}/{MAX_RETRIES}: "
                f"zoom={zoom}, pitch={PITCH}, heading={heading:.2f}, "
                f"lat={lat:.6f}, lon={lon:.6f}"
            )

            url = (
                f"https://www.bing.com/maps/search?"
                f"style=3d"
                f"&cp={lat}~{lon}"
                f"&lvl={zoom}"
                f"&dir={heading:.2f}"
                f"&pi={PITCH}"
                f"&sd=0"
                f"&eh=226.57"
            )

            print(url)

            try:
                driver.get(url)
            except TimeoutException:
                logger.warning(
                    f"Page load timeout for {lat:.6f}, {lon:.6f}, "
                    f"attempt {attempt + 1}"
                )

                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    continue

                raise

            if is_first:
                time.sleep(WAIT_TIME * 3)
                is_first = False
            else:
                time.sleep(WAIT_TIME)

            try:
                driver.save_screenshot(filename)
                logger.info(f"Screenshot saved: {filename}")
            except (TimeoutException, WebDriverException) as e:
                logger.warning(
                    f"Screenshot error for {lat:.6f}, {lon:.6f}, "
                    f"attempt {attempt + 1}: {str(e)[:150]}"
                )

                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    continue

                raise

            add_gps_metadata(filename, lat, lon)
            logger.info(f"GPS metadata added: {filename}")

            return True

        except Exception as e:
            logger.error(
                f"Error on attempt {attempt + 1} for "
                f"{lat:.6f}, {lon:.6f}: {str(e)[:150]}"
            )

            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)

                if (
                    "timeout" in str(e).lower()
                    or "connection" in str(e).lower()
                    or "chrome not reachable" in str(e).lower()
                    or "session deleted" in str(e).lower()
                ):
                    try:
                        restart_driver()
                    except Exception as restart_error:
                        logger.error(
                            f"Failed to restart driver: {restart_error}"
                        )
            else:
                failed_screenshots.append((lat, lon, heading, str(e)))
                logger.error(
                    f"Failed after {MAX_RETRIES} attempts: "
                    f"{lat:.6f}, {lon:.6f}"
                )
                return False

    return False


# =========================
# MAIN LOOP
# =========================

def main_loop():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_coordinates = len(GPS_COORDINATES)
    successful_captures = 0

    logger.info(f"Starting capture of {total_coordinates} coordinates")

    for i, (lat, lon, heading) in enumerate(GPS_COORDINATES):
        try:
            logger.info(
                f"Progress: {i + 1}/{total_coordinates} "
                f"({((i + 1) / total_coordinates) * 100:.1f}%)"
            )

            success = capture_bing_map(
                lat,
                lon,
                heading,
                ZOOM_LEVEL
            )

            if success:
                successful_captures += 1

            time.sleep(0.5)

        except KeyboardInterrupt:
            logger.info("Process interrupted by user")
            driver.quit()
            raise

        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}")
            continue

    driver.quit()

    logger.info("====== CAPTURE SUMMARY ======")
    logger.info(f"Total coordinates: {total_coordinates}")
    logger.info(f"Successful captures: {successful_captures}")
    logger.info(f"Failed captures: {len(failed_screenshots)}")

    if total_coordinates > 0:
        logger.info(
            f"Success rate: "
            f"{(successful_captures / total_coordinates) * 100:.1f}%"
        )

    if failed_screenshots:
        failed_path = f"{OUTPUT_DIR}/failed_coordinates.txt"

        with open(failed_path, "w", encoding="utf-8") as f:
            f.write("# Failed coordinates: lat, lon, heading, error\n")

            for lat, lon, heading, error in failed_screenshots:
                f.write(
                    f"{lat:.6f},{lon:.6f},{heading:.2f},"
                    f"\"{error}\"\n"
                )

        logger.info(f"Failed coordinates saved to {failed_path}")

    logger.info("Screenshot capture process completed!")


if __name__ == "__main__":
    main_loop()