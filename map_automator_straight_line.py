from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.common.exceptions import TimeoutException, WebDriverException
from PIL import Image
import piexif
import os
import cv2
import time
import math
import shutil
import sys
import logging

# === RATE-LIMIT AUTO-RECOVERY ===
# When Bing starts rate-limiting, every frame exhausts MAX_RETRIES (10/10)
# and gets skipped. Detect this by counting CONSECUTIVE failed frames; once
# the threshold is hit, cool down and re-exec the whole Python process.
# The "skip if file exists" logic in capture_bing_map means the script
# resumes seamlessly from where it stopped after the restart.
#
# Override via env vars:
#   RATE_LIMIT_THRESHOLD - consecutive failures that trigger restart (default 3)
#   RATE_LIMIT_COOLDOWN  - seconds to sleep before re-executing (default 600 = 10 min)
RATE_LIMIT_THRESHOLD = int(os.environ.get("RATE_LIMIT_THRESHOLD", 10))
RATE_LIMIT_COOLDOWN  = int(os.environ.get("RATE_LIMIT_COOLDOWN", 600))

# === BROWSER AUTO-DETECTION ===
# Works with whatever Chromium-family or Firefox browser the user has
# installed. Override by setting the BROWSER env var to one of:
#   chrome | edge | brave | chromium | firefox
# Optionally set BROWSER_BINARY to point at a custom executable path.
#
# Detection order (first match wins): Chrome -> Edge -> Brave -> Chromium -> Firefox.
BROWSER_OVERRIDE = os.environ.get("BROWSER", "").strip().lower() or None
BROWSER_BINARY_OVERRIDE = os.environ.get("BROWSER_BINARY", "").strip() or None

# Common install paths per browser (Windows + a couple cross-platform fallbacks).
_BROWSER_PATHS = {
    "chrome": [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
        "/usr/bin/google-chrome",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    ],
    "edge": [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    ],
    "brave": [
        r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe",
        r"C:\Program Files (x86)\BraveSoftware\Brave-Browser\Application\brave.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\BraveSoftware\Brave-Browser\Application\brave.exe"),
        "/usr/bin/brave-browser",
        "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
    ],
    "chromium": [
        r"C:\Program Files\Chromium\Application\chrome.exe",
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
    ],
    "firefox": [
        r"C:\Program Files\Mozilla Firefox\firefox.exe",
        r"C:\Program Files (x86)\Mozilla Firefox\firefox.exe",
        "/usr/bin/firefox",
        "/Applications/Firefox.app/Contents/MacOS/firefox",
    ],
}

def _find_browser_binary(name):
    """Return the first existing path for `name`, or None."""
    # Also check PATH (handy on Linux/macOS)
    path_lookups = {
        "chrome": ["google-chrome", "chrome"],
        "edge": ["microsoft-edge", "msedge"],
        "brave": ["brave", "brave-browser"],
        "chromium": ["chromium", "chromium-browser"],
        "firefox": ["firefox"],
    }
    for cand in _BROWSER_PATHS.get(name, []):
        if cand and os.path.exists(cand):
            return cand
    for exe in path_lookups.get(name, []):
        found = shutil.which(exe)
        if found:
            return found
    return None

def detect_browser():
    """Pick (name, binary_path) honoring BROWSER / BROWSER_BINARY overrides."""
    if BROWSER_OVERRIDE:
        binary = BROWSER_BINARY_OVERRIDE or _find_browser_binary(BROWSER_OVERRIDE)
        if not binary or not os.path.exists(binary):
            raise FileNotFoundError(
                f"BROWSER={BROWSER_OVERRIDE} requested but binary not found. "
                f"Set BROWSER_BINARY to its full path."
            )
        return BROWSER_OVERRIDE, binary
    # Order chosen empirically: Brave produced the cleanest fully-rendered
    # Bing 3D tiles in testing, Edge second, Chrome occasionally returned
    # dark/unrendered frames, Firefox last (slowest 3D tile load).
    for name in ("brave", "edge", "chrome", "chromium", "firefox"):
        binary = _find_browser_binary(name)
        if binary:
            return name, binary
    raise FileNotFoundError(
        "No supported browser found (chrome / edge / brave / chromium / firefox). "
        "Install one, or set BROWSER and BROWSER_BINARY env vars."
    )

# Output directory for the straight START -> END capture line.
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR_LINE = os.path.join(_BASE_DIR, "maps_screenshots_line")

# ZOOM_LEVEL controls how close Bing's 3D camera sits to the ground.
# Higher = closer = sharper buildings and Bing requests higher-LOD tiles.
# 19 gives noticeably sharper textures than 18 at 120 m drone scale.
ZOOM_LEVEL = int(os.environ.get("ZOOM_LEVEL", 18))
# Browser viewport size. Larger = more rendered pixels per frame, which
# gives Bing more screen space to draw high-detail tiles. Output crop is
# normalized so it scales automatically with this size.
VIEWPORT_W = int(os.environ.get("VIEWPORT_W", 2560))
VIEWPORT_H = int(os.environ.get("VIEWPORT_H", 1440))

# === IMAGE FORMAT / COMPRESSION ===
# PNG is lossless but ~5-10x larger than necessary for photographic aerial
# imagery. JPEG at quality 90 is the typical sweet spot for drone datasets:
# imperceptible quality loss vs. PNG, dramatically smaller files. WebP is
# even better (~30% smaller than JPEG at equivalent quality) if downstream
# tools support it.
#
# IMAGE_FORMAT: "jpg" (default) | "webp" | "png"
# IMAGE_QUALITY: 1-100 for jpg/webp (default 90 - high quality). Ignored for png.
IMAGE_FORMAT  = os.environ.get("IMAGE_FORMAT", "jpg").lower().lstrip(".")
IMAGE_QUALITY = int(os.environ.get("IMAGE_QUALITY", 100))
# Per-frame settle time before screenshot. Bing's 3D tile mesh streams in
# progressively; too short and you get dark/unrendered patches. 8s is a
# safer floor; the first frame waits 3x to warm the tile cache.
WAIT_TIME = 8
# Lower pitch = less sky and more ground in frame, which dramatically
# reduces the visible "dark unrendered" area when the 3D mesh is still
# loading. 30 keeps an oblique 3D feel; set to 0 for straight-down aerial.
PITCH = 60
HEIGHT = 120
EAST_HEADING = 90
WEST_HEADING = 270

# Timeout and retry configuration
PAGE_LOAD_TIMEOUT = 30  # seconds
SCREENSHOT_TIMEOUT = 60  # seconds
MAX_RETRIES = 10
RETRY_DELAY = 5  # seconds between retries

is_first = True
failed_screenshots = []  # Track failed captures
driver = None  # created later in create_driver()

# === STRAIGHT-LINE START -> END CONFIGURATION ===
# Captures images along one direct line from START to END. This replaces the
# previous circle/oval spiral coordinate generation while leaving the browser
# driver setup unchanged.
START = (36.07654003335112, -79.79475496658736)
END   = (36.066127362851674, -79.79793149047865)

# Distance between consecutive screenshots along the line.
# Override at runtime, for example:
#   LINE_SPACING_METERS=25 python map_automator_test.py
LINE_SPACING_METERS = float(os.environ.get("LINE_SPACING_METERS", 50))

METERS_PER_DEGREE = 111_320.0


def haversine_distance_m(start, end):
    """Great-circle distance between two (lat, lon) points, in meters."""
    lat1, lon1 = map(math.radians, start)
    lat2, lon2 = map(math.radians, end)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    earth_radius_m = 6_371_000.0

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )

    return 2 * earth_radius_m * math.asin(math.sqrt(a))


def calculate_bearing_degrees(start, end):
    """
    Forward bearing from START to END in degrees clockwise from north.
    This is the Bing Maps camera heading used for every frame on the line.
    """
    lat1, lon1 = map(math.radians, start)
    lat2, lon2 = map(math.radians, end)

    dlon = lon2 - lon1

    x = math.sin(dlon) * math.cos(lat2)
    y = (
        math.cos(lat1) * math.sin(lat2)
        - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    )

    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def interpolate_line_coordinates(start, end, spacing_meters):
    """
    Create evenly spaced capture coordinates from START to END, inclusive.

    The number of segments is ceil(total_distance / spacing_meters), so the
    final END point is always included and the actual spacing is never larger
    than the requested spacing.
    """
    total_distance = haversine_distance_m(start, end)
    num_segments = max(1, math.ceil(total_distance / spacing_meters))
    heading = calculate_bearing_degrees(start, end)

    coords = []
    for i in range(num_segments + 1):
        t = i / num_segments
        lat = start[0] + t * (end[0] - start[0])
        lon = start[1] + t * (end[1] - start[1])
        coords.append((lat, lon, heading))

    return coords


LINE_COORDINATES = interpolate_line_coordinates(
    START,
    END,
    LINE_SPACING_METERS,
)

LINE_DISTANCE_M = haversine_distance_m(START, END)
LINE_HEADING_DEG = calculate_bearing_degrees(START, END)
ACTUAL_LINE_SPACING_M = (
    LINE_DISTANCE_M / max(1, len(LINE_COORDINATES) - 1)
)

print("START:", START)
print("END:", END)
print(f"[line] distance: {LINE_DISTANCE_M:.2f} m")
print(f"[line] requested spacing: {LINE_SPACING_METERS:.2f} m")
print(f"[line] actual spacing: {ACTUAL_LINE_SPACING_M:.2f} m")
print(f"[line] heading: {LINE_HEADING_DEG:.2f} degrees")
print(f"[line] planned frames: {len(LINE_COORDINATES)}")

# === SETUP LOGGING ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

os.makedirs(OUTPUT_DIR_LINE, exist_ok=True)

# === DRIVER FACTORY ===
# Browser is no longer launched at import time. create_driver() is called
# from main_loop() so the script can be imported without spawning Chrome.
def create_driver():
    """
    Build a Selenium WebDriver for whichever browser the host has installed.
    Selenium Manager (built into Selenium 4.6+) auto-fetches the matching
    driver binary, so no webdriver-manager / manual chromedriver setup needed.
    """
    name, binary = detect_browser()
    logger.info(f"Using browser: {name} ({binary})")

    if name == "firefox":
        opts = FirefoxOptions()
        opts.binary_location = binary
        opts.add_argument(f"--width={VIEWPORT_W}")
        opts.add_argument(f"--height={VIEWPORT_H}")
        opts.add_argument("--headless")
        drv = webdriver.Firefox(options=opts)
    elif name == "edge":
        opts = EdgeOptions()
        opts.binary_location = binary
        opts.add_argument(f"--window-size={VIEWPORT_W},{VIEWPORT_H}")
        opts.add_argument("--force-device-scale-factor=1")
        opts.add_argument("--disable-web-security")
        opts.add_argument("--disable-features=VizDisplayCompositor")
        opts.add_argument("--start-maximized")
        opts.add_argument("--headless=new")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        drv = webdriver.Edge(options=opts)
    else:
        # chrome / brave / chromium all use the Chrome driver protocol
        opts = ChromeOptions()
        opts.binary_location = binary
        opts.add_argument(f"--window-size={VIEWPORT_W},{VIEWPORT_H}")
        opts.add_argument("--force-device-scale-factor=1")
        opts.add_argument("--disable-web-security")
        opts.add_argument("--disable-features=VizDisplayCompositor")
        opts.add_argument("--start-maximized")
        opts.add_argument("--headless=new")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        drv = webdriver.Chrome(options=opts)

    drv.set_page_load_timeout(PAGE_LOAD_TIMEOUT)
    drv.implicitly_wait(10)
    return drv

# === GPS Metadata Function ===
def add_gps_metadata(image_path, lat, lon):
    img = Image.open(image_path)
    exif_dict = {"GPS": {
        piexif.GPSIFD.GPSLatitude: [(int(abs(lat)), 1), (int((abs(lat)*60)%60), 1), (int((abs(lat)*3600)%60), 1)],
        piexif.GPSIFD.GPSLatitudeRef: 'N' if lat >= 0 else 'S',
        piexif.GPSIFD.GPSLongitude: [(int(abs(lon)), 1), (int((abs(lon)*60)%60), 1), (int((abs(lon)*3600)%60), 1)],
        piexif.GPSIFD.GPSLongitudeRef: 'E' if lon >= 0 else 'W',
    }}
    exif_bytes = piexif.dump(exif_dict)
    # File is already cropped + compressed by save_screenshot_compressed.
    # piexif.insert works for jpg only - silently skip for png/webp.
    if image_path.lower().endswith((".jpg", ".jpeg")):
        try:
            piexif.insert(exif_bytes, image_path)
        except Exception:
            pass

# === Crop + Compress + Save ===
# Crop fractions are normalized 0-1 against the original 1920x1080 layout
# (rows 90:843, cols 400:1504). Applying them to the actual image size
# means the same UI chrome is removed regardless of VIEWPORT_W/VIEWPORT_H.
_CROP_Y0, _CROP_Y1 = 90 / 1080, 843 / 1080
_CROP_X0, _CROP_X1 = 400 / 1920, 1504 / 1920

def crop_image(image_path):
    """Crop a saved file in place (used when the file is already on disk)."""
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    y0, y1 = int(_CROP_Y0 * h), int(_CROP_Y1 * h)
    x0, x1 = int(_CROP_X0 * w), int(_CROP_X1 * w)
    cv2.imwrite(image_path, image[y0:y1, x0:x1])

def save_screenshot_compressed(png_bytes, output_path):
    """
    Decode a PNG byte string from Selenium, crop the UI chrome, and save
    in the configured format/quality. Avoids ever writing the full-size
    PNG to disk - the only file produced is the final compressed crop.
    """
    import numpy as np
    arr = np.frombuffer(png_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise IOError("Failed to decode screenshot bytes")
    h, w = image.shape[:2]
    y0, y1 = int(_CROP_Y0 * h), int(_CROP_Y1 * h)
    x0, x1 = int(_CROP_X0 * w), int(_CROP_X1 * w)
    cropped = image[y0:y1, x0:x1]

    fmt = IMAGE_FORMAT
    if fmt == "jpg" or fmt == "jpeg":
        params = [int(cv2.IMWRITE_JPEG_QUALITY), IMAGE_QUALITY,
                  int(cv2.IMWRITE_JPEG_OPTIMIZE), 1,
                  int(cv2.IMWRITE_JPEG_PROGRESSIVE), 1]
    elif fmt == "webp":
        params = [int(cv2.IMWRITE_WEBP_QUALITY), IMAGE_QUALITY]
    elif fmt == "png":
        params = [int(cv2.IMWRITE_PNG_COMPRESSION), 9]
    else:
        raise ValueError(f"Unknown IMAGE_FORMAT: {fmt}")

    if not cv2.imwrite(output_path, cropped, params):
        raise IOError(f"cv2.imwrite failed for {output_path}")

# === Anti-Dark-Frame Helpers ===
# A captured frame is "too dark" when Bing's WebGL canvas hasn't finished
# painting opaque tile pixels. We attack this from 3 angles:
#   (A) network-idle gate (wait_for_tiles_loaded, below)
#   (B) render-poke - dispatch resize+mousemove events to nudge the WebGL
#       canvas into redrawing once tiles arrive.
#   (C) pixel-luminance check - read the saved PNG, compute mean grayscale
#       brightness; if too low, wait longer and retry. Browser-agnostic.

# Brightness threshold (0-255). Mean luminance below this is treated as
# "frame mostly dark/unrendered" and triggers a retry. Daylight 3D map
# tiles typically score 90-160; pre-render black is ~0-10.
DARK_FRAME_THRESHOLD = float(os.environ.get("DARK_FRAME_THRESHOLD", 60))
DARK_FRAME_RETRIES   = int(os.environ.get("DARK_FRAME_RETRIES", 5))

def hide_loading_indicators():
    """
    Inject CSS that hides Bing's tile-loading spinner and any other busy
    indicator from the screenshot. The spinner overlay was appearing in
    captures because it sits ON TOP of the WebGL canvas - so even after
    the underlying tiles finished, the spinner element itself was visible
    in the frame. We can't make Bing finish faster, but we CAN remove its
    spinner overlay from the rendered DOM before the screenshot fires.
    """
    try:
        driver.execute_script("""
            const css = `
                [class*="loading" i]:not([class*="loaded" i]):not([class*="done" i]),
                [class*="spinner" i],
                [class*="LoadingIndicator" i],
                [class*="ProgressIndicator" i],
                [class*="busyIndicator" i],
                [role="progressbar"],
                [aria-busy="true"],
                .ms-Spinner, .b_loading,
                #directionsLoadingIndicator,
                .loadingPanel, .mapLoading {
                    display: none !important;
                    visibility: hidden !important;
                    opacity: 0 !important;
                    pointer-events: none !important;
                }
            `;
            let style = document.getElementById('__hide_spinners__');
            if (!style) {
                style = document.createElement('style');
                style.id = '__hide_spinners__';
                document.head.appendChild(style);
            }
            style.textContent = css;
        """)
    except Exception:
        pass

def poke_renderer():
    """Force Bing's WebGL canvas to redraw without changing the camera."""
    try:
        driver.execute_script("""
            window.dispatchEvent(new Event('resize'));
            const c = document.querySelector('canvas');
            if (c) {
                const r = c.getBoundingClientRect();
                ['mousemove','pointermove'].forEach(t => {
                    c.dispatchEvent(new MouseEvent(t, {
                        clientX: r.left + r.width/2,
                        clientY: r.top  + r.height/2,
                        bubbles: true,
                    }));
                });
            }
        """)
    except Exception:
        pass

def wait_for_screen_stable(timeout=18.0, settle_checks=3, settle_delay=1.1):
    """
    Wait until consecutive in-memory screenshots hash-match - i.e. the
    visible page has stopped changing.

    QUALITY NOTE: the comparison is done at 480x270 (NOT 96x54). The lower
    resolution caused the gate to fire while Bing was still streaming
    high-detail texture tiles - low-res tile pixels hashed identical to
    high-res ones at 96x54. Higher hash resolution detects subtle texture
    sharpening so we only screenshot once tiles really are at full LOD.
    """
    import io, hashlib
    last = None
    stable = 0
    start = time.time()
    while time.time() - start < timeout:
        try:
            png = driver.get_screenshot_as_png()
            img = Image.open(io.BytesIO(png)).resize((480, 270)).convert("L")
            h = hashlib.md5(img.tobytes()).hexdigest()
        except Exception:
            h = None
        if h is not None and h == last:
            stable += 1
            if stable >= settle_checks:
                return True
        else:
            stable = 0
            last = h
        time.sleep(settle_delay)
    return False

def measure_brightness(image_path):
    """
    Mean grayscale luminance of the GROUND area of the saved PNG.

    Why a sub-region instead of the whole frame: at PITCH=60 the upper
    portion of the frame is sky/horizon. Unrendered sky is near-black,
    pulling the whole-frame mean below the dark-threshold even when the
    ground tiles are fully rendered. Triggers infinite false retries.
    Sampling only the ground band (lower 65% center) makes the metric
    reflect actual map content quality.
    """
    try:
        img = Image.open(image_path).convert("L")
        w, h = img.size
        crop = img.crop((int(w * 0.10), int(h * 0.30),
                         int(w * 0.90), int(h * 0.95)))
        pixels = list(crop.getdata())
        if not pixels:
            return None
        return sum(pixels) / len(pixels)
    except Exception:
        return None

# === Tile-Load Wait ===
# The previous version polled DOM selectors (.loadingIndicator etc.) that
# don't actually exist in Bing's 3D map -> it always returned True instantly,
# leaving the screenshot at the same instant the fixed sleep ended.
#
# This version uses NETWORK-IDLE detection via the Performance API:
# count the number of resources the page has loaded, and require the count
# to remain unchanged for `quiet_ms` milliseconds before declaring ready.
# Bing's 3D tile fetcher is constantly loading mesh chunks and texture
# atlases - when those stop arriving, the visible map is fully rendered.
def wait_for_tiles_loaded(min_wait=3.0, max_wait=20.0, quiet_ms=2500):
    # Mandatory floor so even a misbehaving Performance API can't return
    # instantly on a blank page.
    time.sleep(min_wait)

    start = time.time()
    last_count = -1
    last_change = time.time()
    while time.time() - start < max_wait:
        try:
            count = driver.execute_script(
                "return performance.getEntriesByType('resource').length;"
            )
        except Exception:
            count = -1
        if count != last_count:
            last_count = count
            last_change = time.time()
        elif (time.time() - last_change) * 1000.0 >= quiet_ms:
            return True
        time.sleep(0.3)
    return False

# === Capture Function ===
def capture_bing_map(lat, lon, heading, zoom, frame_id, output_dir):
    global is_first

    # Sequential frame ID prefix keeps files in capture order. Lat/lon are
    # printed to 6 decimals (~11 cm precision) so the actual coordinate is
    # visible in the filename - the underlying Python float carries the
    # full precision of the user-specified center.
    ext = "jpg" if IMAGE_FORMAT in ("jpg", "jpeg") else IMAGE_FORMAT
    filename = f"{output_dir}/frame_{frame_id:05d}_{lat:.6f}_{lon:.6f}_{heading:.2f}.{ext}"

    # Skip if file already exists
    if os.path.exists(filename):
        logger.info(f"Screenshot already exists, skipping: {filename}")
        return True

    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"Attempt {attempt + 1}/{MAX_RETRIES}: Capturing height={HEIGHT}, zoom={zoom}, pitch={PITCH}, heading={heading}, latitude={lat:.4f}, longitude={lon:.4f}")

            # url = f"https://www.bing.com/maps?cp={lat}~{lon}&lvl={zoom}&style=a&pi={PITCH}"
            # url = f"https://www.bing.com/maps/search?ss=id.ypid%3A521x400073041&cp={lat}~{lon}&eh=270&lvl={zoom}&style=3d&dir=249.4&pi={PITCH}"
            # url = f"https://www.bing.com/maps/search?style=3d&cp={lat}~{lon}&lvl={zoom}&dir={heading}&pi={45}&sd=0"
            url = f"https://www.bing.com/maps/search?style=3d&cp={lat}~{lon}&lvl={zoom}&dir={heading}&pi={PITCH}&sd=0&eh=226.57"
            print(url)
            # Load the page with timeout handling
            try:
                driver.get(url)
            except TimeoutException:
                logger.warning(f"Page load timeout for {lat:.4f}, {lon:.4f}, attempt {attempt + 1}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    continue
                else:
                    raise

            # Wait for page to load
            if is_first:
                time.sleep(WAIT_TIME * 3)  # Extra wait for the first load
                is_first = False
            else:
                time.sleep(WAIT_TIME)

            # Block until Bing's tile loader is network-idle. Longer
            # quiet window catches the SECOND wave of fetches Bing makes
            # for the highest-LOD texture tiles after the initial mesh.
            wait_for_tiles_loaded(min_wait=3.0, max_wait=22.0, quiet_ms=4000)
            # Nudge the WebGL canvas to redraw with the just-arrived tiles.
            poke_renderer()
            # Hide Bing's loading spinner overlay so it doesn't appear in
            # the middle of the screenshot.
            hide_loading_indicators()
            time.sleep(0.5)
            # Final gate: don't shoot until consecutive screenshot hashes
            # match at HIGH resolution - i.e. high-detail textures are
            # also done streaming, not just the coarse mesh.
            wait_for_screen_stable(timeout=18.0, settle_checks=3, settle_delay=1.1)

            # try:
            #     flyover_button = driver.find_element("xpath", "//button[contains(text(),'3D')]")
            #     flyover_button.click()
            #     time.sleep(WAIT_TIME)
            # except:
            #     logger.info("Flyover button not found, continuing with aerial view.")

            # Take screenshot, then verify it isn't mostly dark. If it is,
            # wait longer + poke the renderer + re-shoot. Up to
            # DARK_FRAME_RETRIES attempts before accepting whatever we got.
            screenshot_ok = False
            for dark_try in range(DARK_FRAME_RETRIES + 1):
                try:
                    # Capture to memory, crop, then save in the configured
                    # format/quality. Avoids ever writing the bloated full
                    # PNG to disk - only the final compressed crop hits the
                    # filesystem, dramatically reducing dataset size.
                    png_bytes = driver.get_screenshot_as_png()
                    save_screenshot_compressed(png_bytes, filename)
                except (TimeoutException, WebDriverException) as e:
                    logger.warning(
                        f"Screenshot error for {lat:.6f}, {lon:.6f}, "
                        f"attempt {attempt + 1}: {str(e)[:100]}"
                    )
                    break  # let outer loop handle the retry
                brightness = measure_brightness(filename)
                if brightness is None or brightness >= DARK_FRAME_THRESHOLD:
                    logger.info(
                        f"Screenshot saved (brightness={brightness:.1f}): {filename}"
                    )
                    screenshot_ok = True
                    break
                if dark_try < DARK_FRAME_RETRIES:
                    extra = 3.0 + 2.0 * dark_try  # progressive backoff
                    logger.warning(
                        f"Frame too dark (brightness={brightness:.1f} < "
                        f"{DARK_FRAME_THRESHOLD}); waiting {extra:.1f}s and "
                        f"re-shooting (dark-retry {dark_try + 1}/{DARK_FRAME_RETRIES})"
                    )
                    time.sleep(extra)
                    poke_renderer()
                    hide_loading_indicators()
                    wait_for_tiles_loaded(min_wait=1.0, max_wait=12.0, quiet_ms=2000)
                    wait_for_screen_stable(timeout=8.0, settle_checks=2, settle_delay=0.8)
                else:
                    logger.warning(
                        f"Frame still dark (brightness={brightness:.1f}) after "
                        f"{DARK_FRAME_RETRIES} retries; accepting it."
                    )
                    screenshot_ok = True
            if not screenshot_ok:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    continue
                else:
                    raise WebDriverException("Screenshot failed all retries")

            add_gps_metadata(filename, lat, lon)
            logger.info(f"GPS metadata added: {filename}")
            return True

        except Exception as e:
            logger.error(f"Error on attempt {attempt + 1} for {lat:.4f}, {lon:.4f}: {str(e)[:100]}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
                # Try to restart the driver if it's unresponsive
                if "timeout" in str(e).lower() or "connection" in str(e).lower():
                    try:
                        restart_driver()
                    except Exception as restart_error:
                        logger.error(f"Failed to restart driver: {restart_error}")
            else:
                # Final attempt failed, log the failure
                failed_screenshots.append((lat, lon, heading, str(e)))
                logger.error(f"Failed to capture after {MAX_RETRIES} attempts: {lat:.4f}, {lon:.4f}")
                return False

    return False

# === Driver Restart Function ===
def restart_driver():
    global driver, is_first
    logger.info("Restarting WebDriver...")
    try:
        driver.quit()
    except:
        pass

    driver = create_driver()
    is_first = True  # Reset for extra wait time
    logger.info("WebDriver restarted successfully")

# === Trajectory Preview ===
# Plot the planned START -> END capture line BEFORE any browser activity,
# so the user can sanity-check the route before captures begin.
# Saves a PNG next to the screenshots and also opens an interactive window
# if TRAJECTORY_PREVIEW=block is set and a display is available.
def show_trajectory_preview(coords, output_dir, block=False):
    """Draw the straight-line capture plan."""
    if not coords:
        logger.warning("No coordinates to preview.")
        return None

    try:
        import matplotlib
        if not block:
            matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning(
            "matplotlib not installed - skipping trajectory preview. "
            "Install with: pip install matplotlib"
        )
        return None

    lats = [c[0] for c in coords]
    lons = [c[1] for c in coords]

    fig, ax = plt.subplots(1, 1, figsize=(9, 7))

    ax.plot(lons, lats, linewidth=1.5, label="Capture line")
    ax.scatter(lons, lats, s=10, label="Capture points")

    ax.plot(START[1], START[0], marker="o", markersize=10, label="START")
    ax.plot(END[1], END[0], marker="X", markersize=10, label="END")

    # Keep latitude/longitude visually proportional near this route.
    center_lat = (START[0] + END[0]) / 2.0
    aspect = 1.0 / math.cos(math.radians(center_lat))
    ax.set_aspect(aspect)

    # Add a small margin so the endpoints do not touch the plot border.
    lat_span = max(lats) - min(lats)
    lon_span = max(lons) - min(lons)
    lat_margin = max(lat_span * 0.15, 0.0002)
    lon_margin = max(lon_span * 0.15, 0.0002)

    ax.set_xlim(min(lons) - lon_margin, max(lons) + lon_margin)
    ax.set_ylim(min(lats) - lat_margin, max(lats) + lat_margin)

    ax.set_title(f"Straight START -> END capture plan: {len(coords)} frames")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="best", fontsize=8)

    info = (
        f"Distance: {LINE_DISTANCE_M:.2f} m    "
        f"Heading: {LINE_HEADING_DEG:.2f}°    "
        f"Spacing: {ACTUAL_LINE_SPACING_M:.2f} m"
    )
    fig.suptitle(info, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    os.makedirs(output_dir, exist_ok=True)
    preview_path = os.path.join(output_dir, "trajectory_preview.png")
    fig.savefig(preview_path, dpi=140, bbox_inches="tight")
    logger.info(f"Trajectory preview saved: {preview_path}")

    if block:
        plt.show()

    plt.close(fig)
    return preview_path

# === Rate-Limit Self-Restart ===
def trigger_self_restart(reason):
    """
    Cool down and re-execute this script in place.

    Used when Bing rate-limits us hard enough that consecutive frames all
    burn through MAX_RETRIES. The browser is closed cleanly, the process
    sleeps for RATE_LIMIT_COOLDOWN seconds (long enough for the rate limit
    to age out), and then re-execs itself with the same arguments. Existing
    captured files are detected and skipped on resume, so no work is lost
    and no duplicate frames are produced.
    """
    logger.warning("=" * 60)
    logger.warning(f"RATE LIMIT DETECTED: {reason}")
    logger.warning(f"Cooling down for {RATE_LIMIT_COOLDOWN} s, then restarting...")
    logger.warning("=" * 60)
    try:
        if driver is not None:
            driver.quit()
    except Exception:
        pass
    time.sleep(RATE_LIMIT_COOLDOWN)
    logger.warning("Re-executing script - already-captured frames will be skipped.")
    try:
        os.execv(sys.executable, [sys.executable] + sys.argv)
    except Exception as e:
        # If execv fails for any reason, fall back to a hard exit so a
        # supervising shell loop (e.g. `while ($true) { python ...}`) can
        # restart us. Keeps the script unattended-friendly.
        logger.error(f"os.execv failed ({e}); exiting with code 75 for external restart.")
        os._exit(75)

# === Per-Shape Capture Pass ===
# One pass = one shape (circle or oval) -> one folder. Each pass owns its
# own frame-ID counter and its own failure list, so circle and oval data
# never get mixed even though they share the browser session.
def run_capture_pass(shape, coords, output_dir):
    total = len(coords)
    successful = 0
    failures = []
    # Counts only frames that ACTUALLY ran the browser (i.e. the file did
    # not pre-exist). After a self-restart the resume sweep over already-
    # captured frames returns success without ever hitting Bing, and must
    # not be counted as "successful frames that broke a rate-limit streak."
    consecutive_failures = 0

    logger.info(f"=== {shape.upper()} pass: {total} planned captures -> {output_dir} ===")

    for i, (lat, lon, heading) in enumerate(coords):
        try:
            frame_id = i + 1  # 1-based, restarts at 1 for each shape's folder

            # Detect resume-skip BEFORE counting so it doesn't reset the
            # consecutive-failure streak (the file existing means we never
            # talked to Bing on this iteration).
            ext = "jpg" if IMAGE_FORMAT in ("jpg", "jpeg") else IMAGE_FORMAT
            preexisting = os.path.exists(
                f"{output_dir}/frame_{frame_id:05d}_{lat:.6f}_{lon:.6f}_{heading:.2f}.{ext}"
            )

            logger.info(
                f"[{shape}] Progress: {frame_id}/{total} "
                f"({(frame_id/total)*100:.1f}%)"
            )
            success = capture_bing_map(lat, lon, heading, ZOOM_LEVEL,
                                       frame_id, output_dir)
            if success:
                successful += 1
                if not preexisting:
                    consecutive_failures = 0  # streak broken by a real capture
            else:
                failures.append((lat, lon, heading, "capture returned False"))
                consecutive_failures += 1
                if consecutive_failures >= RATE_LIMIT_THRESHOLD:
                    trigger_self_restart(
                        f"{consecutive_failures} consecutive failed frames in "
                        f"{shape} pass (last frame_id={frame_id})"
                    )

            time.sleep(0.5)

        except KeyboardInterrupt:
            logger.info(f"\n[{shape}] Interrupted by user")
            raise
        except Exception as e:
            logger.error(f"[{shape}] Unexpected error: {e}")
            failures.append((lat, lon, heading, str(e)))
            consecutive_failures += 1
            if consecutive_failures >= RATE_LIMIT_THRESHOLD:
                trigger_self_restart(
                    f"{consecutive_failures} consecutive errors in {shape} "
                    f"pass (last frame_id={frame_id})"
                )
            continue

    logger.info(f"=== {shape.upper()} SUMMARY ===")
    logger.info(f"  Total:      {total}")
    logger.info(f"  Successful: {successful}")
    logger.info(f"  Failed:     {len(failures)}")
    logger.info(f"  Success %:  {(successful/total)*100:.1f}%")

    if failures:
        fail_file = os.path.join(output_dir, "failed_coordinates.txt")
        with open(fail_file, "w") as f:
            f.write("# Failed coordinates (lat, lon, heading, error)\n")
            for lat, lon, heading, error in failures:
                f.write(f"{lat:.6f},{lon:.6f},{heading:.4f},\"{error}\"\n")
        logger.info(f"  Failures logged to: {fail_file}")

    return successful, failures

# === Main ===
def main_loop():
    global driver, is_first

    # Show the planned straight-line trajectory BEFORE any browser activity,
    # so the route can be sanity-checked before captures begin.
    block_preview = os.environ.get("TRAJECTORY_PREVIEW", "").lower() == "block"
    show_trajectory_preview(
        LINE_COORDINATES,
        OUTPUT_DIR_LINE,
        block=block_preview,
    )

    driver = create_driver()  # browser only starts here, not at import time

    try:
        is_first = True
        run_capture_pass("line", LINE_COORDINATES, OUTPUT_DIR_LINE)
    finally:
        driver.quit()

    logger.info("Straight-line capture pass completed!")


if __name__ == "__main__":
    main_loop()
