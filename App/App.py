# Imports
# Streamlit
import streamlit as st
import streamlit.components.v1 as components
# Data manipulation
import pandas as pd
import numpy as np
# Dataviz
import plotly.express as px
# Geoplotting
import pydeck as pdk
# UML
import scipy.sparse as ss
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_distances

alt.renderers.set_embed_options(theme='dark')

