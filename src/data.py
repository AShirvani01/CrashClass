import pandas as pd
import geopandas as gpd
import requests
from io import StringIO, BytesIO
from zipfile import ZipFile
from shapely.geometry import shape
import json
from pathlib import Path
from arcgis import GIS

from config.paths import DATA_DIR


def get_collision_data() -> pd.DataFrame:
    """Pulls the collision dataset from the Open Data Toronto API.

    Returns:
        pd.DataFrame: Dataframe where each row is a unique collision and each
        column corresponds to a feature.
    """
    base_url = "https://ckan0.cf.opendata.inter.prod-toronto.ca"
    package_url = f"{base_url}/api/3/action/package_show"
    params = {
        "id": ("motor-vehicle-collisions-involving-killed-or-"
               "seriously-injured-persons")
    }
    package = requests.get(package_url, params=params).json()

    resource = package['result']['resources'][0]
    url = f"{base_url}/datastore/dump/{resource['id']}"
    resource_dump_data = requests.get(url).text
    collision_data = pd.read_csv(StringIO(resource_dump_data))

    return collision_data


def convert_to_geodata(df: pd.DataFrame, crs='EPSG:2958') -> gpd.GeoDataFrame:
    """Convert Dataframe to Geodataframe.

    Args:
        df (pd.DataFrame): Dataframe of collision data with 'geometry' column
            in GeoJSON format.
            e.g. {"type": "Point", "coordinates": [-79.3187970, 43.6995952]}
        crs: Coordinate Reference System of the data.

    Returns:
        gpd.GeoDataFrame: Geodataframe with geometry column in WKT format.
    """
    # Convert geometry column from GeoJSON -> WKT -> GeoSeries
    df['geometry'] = (
        df['geometry']
        .apply(lambda x: shape(json.loads(x)).wkt)
        .pipe(gpd.GeoSeries.from_wkt)
    )

    gdf = gpd.GeoDataFrame(df, crs='EPSG:4326', geometry="geometry")
    gdf = gdf.to_crs(crs=crs)

    return gdf


def load_external_data(file_path: Path, crs='EPSG:2958') -> gpd.GeoDataFrame:
    """Load external data.

    Args:
        file_path (Path): File path to external data
        crs: Coordinate Reference System of the data.

    Returns:
        gpd.GeoDataFrame: Geodataframe with geometry column in WKT format.
    """

    gdf = gpd.read_file(file_path).to_crs(crs=crs)

    return gdf


def download_streets_data(output_dir: Path = DATA_DIR / "canada_streets") -> None:
    
    if (output_dir / 'canada_streets.shp').exists():
        print(f"{output_dir}\canada_streets.shp already exists.")
        return

    url = (
        "https://www12.statcan.gc.ca/census-recensement/2011"
        "/geo/RNF-FRR/files-fichiers/lrnf000r24a_e.zip"
    )
    response = requests.get(url)

    with ZipFile(BytesIO(response.content)) as zip_file:
        zip_file.extractall(output_dir)
    
        for file in zip_file.namelist():
            file_path = output_dir / file
            file_path.rename(output_dir / f'canada_streets{file_path.suffix}')


def download_hospital_data(output_dir: Path = DATA_DIR) -> None:
    
    file_name = 'ontario_health_services.geojson'

    if (output_dir / file_name).exists():
        print(f"{output_dir}\{file_name} already exists.")
        return

    url = (
        "https://ws.lioservices.lrc.gov.on.ca/arcgis2/rest/services/"
        "LIO_OPEN_DATA/LIO_Open09/MapServer/26/query"
    )
    params = {
        'outFields': '*',
        'where': '1=1',
        'f': 'geojson',
        'resultOffset': 0
    }

    features = []

    # Bypass max query size (2000)
    while True:
        response = requests.get(url, params=params).json()

        if not response.get('features'):
            break

        features.extend(response['features'])
        params['resultOffset'] += len(response['features'])

    geojson = {
        'type': 'FeatureCollection',
        'features': features,
    }

    with open(output_dir / file_name, 'w') as f:
        json.dump(geojson, f)


def download_neighbourhood_data(output_dir: Path = DATA_DIR) -> None:

    file_name = 'toronto_neighbourhoods.geojson'

    if (output_dir / file_name).exists():
        print(f"{output_dir}\{file_name} already exists.")
        return

    gis = GIS()

    item_id = "5913f337900949d9be150ac6f203eefb"
    item = gis.content.get(item_id)
    feature_layer = item.layers[0]
    url = f"{feature_layer.url}/query"

    params = {
        'outFields': '*',
        'where': '1=1',
        'f': 'geojson'
    }

    response = requests.get(url, params=params).json()
    with open(output_dir / file_name, 'w') as f:
        json.dump(response, f)


