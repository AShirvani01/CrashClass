import streamlit as st
import pandas as pd
import numpy as np
import catboost as cb
from netcal.scaling import LogisticCalibration
import shap
from streamlit_shap import st_shap
import folium as fl
from streamlit_folium import st_folium
import geopandas as gpd
from shapely.geometry import Point
from config.paths import (
    DATA_DIR,
    MODEL_DIR,
    NEIGHBOURHOODS_PATH,
    HEALTH_SERVICES_PATH
)
from preprocessing import (
    dist_to_nearest_hospital,
    filter_toronto_hospitals_with_er
)
from data import load_external_data
import os

os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'


# Formatting
st.set_page_config(initial_sidebar_state='expanded')
st.html("""
    <style>
        .stMainBlockContainer {
            max-width:70rem;
        }
    </style>
    """
)
st.title('Fatality Risk Prediction in Toronto Motor Vehicle Collisions')
st.markdown(
    """
<style>
[data-testid="stMetricValue"] {
    font-size: 25px;
}
</style>
""",
    unsafe_allow_html=True,
)

# Load data
df = pd.read_csv(DATA_DIR / 'processed_data.csv').drop(columns=['ACCLASS'])
cat_cols = df.select_dtypes('object').columns
df[cat_cols] = df[cat_cols].apply(lambda x: x.astype('category'))
hoods = load_external_data(NEIGHBOURHOODS_PATH)
hospitals = (
    load_external_data(HEALTH_SERVICES_PATH)
    .pipe(filter_toronto_hospitals_with_er)
)
model = cb.CatBoostClassifier()
model.load_model(MODEL_DIR / 'final_model.json', format='json')
lc = LogisticCalibration()
lc.load_model(MODEL_DIR / 'final_calibration.pt')
explainer = shap.TreeExplainer(model)


def prepare_input(data: dict, features: list):

    processed_data = []

    for feature in features:
        processed_data.append(data.get(feature))

    input_df = pd.DataFrame(np.array([processed_data]), columns=features).astype(df.dtypes)

    return input_df


def create_sidebar(user_data):
    """Create a markdown summary of user data for the sidebar."""
    st.sidebar.title("✨ Submission Details")
    st.sidebar.markdown("---")
    # Location Section
    st.sidebar.header('Location')
    st.sidebar.markdown(f"""
        - Neighbourhood: {user_data['NEIGHBOURHOOD_158']}
        - District: {user_data['DISTRICT']}
        - Nearest Hospital with ER: {user_data['NEAREST_HOSPITAL']}
    """)

    # if st.sidebar.button("🔄 Reset Parameters"):
    # Add Clear Form Button to Sidebar
    #     for key in st.session_state.user_data:
    #         st.session_state.pop(key)
    #     st.rerun()


def validate_neighbourhood():
    if not map.get('last_clicked'):
        message = 'Select a location on the map.'
        return False, message
    elif collision['NHName'].isna()[0]:
        message = """Collision must have occurred within the City of Toronto.
             Select a location inside the shaded region of the map."""
        return False, message
    return True, ''


def validate_district():
    if district is None:
        message = 'Select a district.'
        return False, message
    return True, ''


def validate_manoeuver():
    if manoeuver is None:
        message = 'Select at least 1 manoeuver.'
        return False, message
    return True, ''


def validate_drivact():
    if drivact is None:
        message = 'Select at least 1 drive action.'
        return False, message
    return True, ''


def validate_persons():
    if numpersons == 0:
        message = 'Must have at least 1 person involved.'
        return False, message
    return True, ''


with st.form('enhanced_validation'):

    st.header('Location')
    m = fl.Map(
        location=[43.6768, -79.3969],
        zoom_start=11,
        tiles='OpenStreetMap',
        max_bounds=True,
        min_lon=-80,
        min_lat=43,
        max_lon=-78,
        max_lat=45
    )
    m.add_child(fl.LatLngPopup())
    style = {
        'color': 'black',
        'weight': 1,
        'fillColor': 'black',
        'fillOpacity': 0.03
    }

    layer = fl.GeoJson(hoods, name='Hoods', style_function=lambda x: style)
    layer.add_to(m)
    map = st_folium(m, height=500, width=1000)

    if map.get('last_clicked'):
        x, y = map['last_clicked']['lng'], map['last_clicked']['lat']
        collision = gpd.GeoDataFrame(geometry=gpd.GeoSeries(Point(x, y)), crs='EPSG:4326')
        collision = collision.to_crs('EPSG:2958')
        collision = collision.sjoin(hoods, how='left', predicate='within')
        print(collision)
        collision = dist_to_nearest_hospital(collision, hospitals)
        print(collision)

    district = st.pills('DISTRICT', list(df.DISTRICT.unique()), selection_mode='single')

    st.header('Date')
    date = st.date_input('DATE')
    time = st.time_input('TIME', value='00:00', step=3600)

    st.header('Road Type')
    road_class = st.selectbox(
        'Road Class',
        options=list(df.ROAD_CLASS.unique()),
        help='Hi'
    )
    traffctl = st.selectbox('TRAFFCTL', list(df.TRAFFCTL.unique()))

    st.header('Environment')
    visibility = st.selectbox('VISIBILITY', list(df.VISIBILITY.unique()))
    light = st.selectbox('LIGHT', list(df.LIGHT.unique()))
    rdsfcond = st.selectbox('RDSFCOND', list(df.RDSFCOND.unique()))

    st.header('Motor Vehicle Characteristics')
    impactype = st.selectbox('IMPACTYPE', list(df.IMPACTYPE.unique()))
    manoeuver = st.multiselect('MANOEUVER',
        ['Changing Lanes',
         'Going Ahead',
         'Making U Turn',
         'Merging',
         'Other',
         'Overtaking',
         'Parked',
         'Pulling Away from Shoulder or Curb',
         'Pulling Onto Shoulder or toward Curb',
         'Reversing',
         'Slowing or Stopping',
         'Stopped',
         'Turning Left',
         'Turning Right',
         'Unknown']
    )

    manoeuver = [x.replace(' ', '') for x in manoeuver]

    drivact = st.multiselect('DRIVACT',
        ['Disobeyed Traffic Control',
         'Driving Properly',
         'Exceeding Speed Limit',
         'Failed to Yield Right of Way',
         'Following too Close',
         'Improper Lane Change',
         'Improper Passing',
         'Improper Turn',
         'Lost control',
         'Other',
         'Speed too Fast For Condition',
         'Speed too Slow',
         'Wrong Way on One Way Road']
    )

    drivact = [x.replace(' ', '') for x in drivact]

    st.header('Type Involved')
    checkboxes = {}
    checkbox_cols = df.select_dtypes('bool').columns.to_list()[:8]
    cols = st.columns(4)
    iter_cols = [0, 1, 2, 3]*2

    for i, col in enumerate(checkbox_cols):
        with cols[iter_cols[i]]:
            checkboxes[col] = st.checkbox(col, key=col)

    st.header('Behaviour')
    checkbox_cols = df.select_dtypes('bool').columns.to_list()[8:13]
    cols = st.columns(3)
    iter_cols = [0, 1, 2]*2

    for i, col in enumerate(checkbox_cols):
        with cols[iter_cols[i]]:
            checkboxes[col] = st.checkbox(col, key=col)

    # Invage
    st.header('Involved Ages')
    invage_cols = [col for col in df.columns if col.startswith('INVAGE')]
    invage = {}
    cols = st.columns(5)
    iter_cols = [0, 1, 2, 3, 4]*5
    for i, col in enumerate(invage_cols):
        with cols[iter_cols[i]]:
            invage[col] = st.number_input(col, min_value=0, step=1)
    numpersons = sum(invage.values())
    st.metric('Number of People Involved', numpersons)

    month = date.month
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    dow = date.weekday()
    dow_sin = np.sin(2 * np.pi * dow / 7)
    dow_cos = np.cos(2 * np.pi * dow / 7)
    hour = int(time.strftime('%H'))
    hour_sin = np.sin(2 * np.pi * dow / 24)
    hour_cos = np.cos(2 * np.pi * dow / 24)

    submitted = st.form_submit_button('Predict')
    if submitted:
        st.session_state.user_data = {
            'ROAD_CLASS': road_class,
            'DISTRICT': district,
            'TRAFFCTL': traffctl,
            'VISIBILITY': visibility,
            'LIGHT': light,
            'RDSFCOND': rdsfcond,
            'IMPACTYPE': impactype,
            'NEIGHBOURHOOD_158': collision['NHName'][0],
            'YEAR': date.year,
            'MONTH_sin': month_sin,
            'MONTH_cos': month_cos,
            'DOW_sin': dow_sin,
            'DOW_cos': dow_cos,
            'HOUR_sin': hour_sin,
            'HOUR_cos': hour_cos,
            'NUMPERSONS': numpersons,
            f'MANOEUVER_{manoeuver}': manoeuver,
            f'DRIVACT_{drivact}': drivact,
            'NEAREST_HOSPITAL': collision['NEAREST_HOSPITAL'][0],
            'DIST_TO_HOS': collision['DIST_TO_HOS'][0]
        }

        st.session_state.user_data.update(invage)
        st.session_state.user_data.update(checkboxes)

        validations = [
            validate_neighbourhood(),
            validate_district(),
            validate_manoeuver(),
            validate_drivact(),
            validate_persons()
        ]

        if all(v[0] for v in validations):
            st.success('Prediction successful')
            input_df = prepare_input(st.session_state.user_data, df.columns)
            pred = model.predict_proba(input_df)
            calibrated = lc.transform(pred)
            st.write(f'Probability of at least 1 fatality: {calibrated[0]*100:.2f}%')
            shap_values = explainer(input_df)
            st_shap(shap.plots.waterfall(shap_values[0]), height=600, width=1000)
            st.markdown("""$f(x)$ is the log-odds *before* calibration.\\
                        $E[f(X)]$ is the average log-odds *before* calibration.\\
                        ***Note***: *The waterfall plot is for general interpretation of the risk estimate.\\
                        Focus should be on relative magnitude and direction of effects,
                        rather than absolute magnitude.*""")

            create_sidebar(st.session_state.user_data)

        else:
            for valid, message in validations:
                if not valid:
                    st.error(message, icon=':material/report:')
