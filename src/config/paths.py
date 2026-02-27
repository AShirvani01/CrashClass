from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / 'data'
MODEL_DIR = PROJECT_ROOT / 'models'

HEALTH_SERVICES_PATH = DATA_DIR / 'ontario_health_services.geojson'
NEIGHBOURHOODS_PATH = DATA_DIR / 'toronto_neighbourhoods.geojson'
STREETS_PATH = DATA_DIR / 'canada_streets' / 'canada_streets.shp'
