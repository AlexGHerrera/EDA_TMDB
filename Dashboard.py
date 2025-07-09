import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS

# --- 1. Presentación y branding ---
st.set_page_config(page_title='EDA TMDB Dashboard', layout='wide')
st.title('🎬 EDA TMDB: Exploración interactiva del éxito en el cine')
st.markdown('''
Este dashboard interactivo permite explorar los factores que influyen en el éxito de taquilla, tendencias de géneros y patrones clave en el dataset de películas de TMDB.
''')
st.markdown('---')

# --- 2. Carga de datos ---
@st.cache_data
def load_data():
    df = pd.read_csv('data/tmdb_5000_movies.csv')
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    import ast
    df['genres'] = df['genres'].apply(lambda x: [d['name'] for d in ast.literal_eval(x)] if pd.notnull(x) else [])
    return df

df = load_data()

# --- 3. Filtros interactivos ---
st.sidebar.image("logo_ag.png", width=90)
st.sidebar.markdown(
    """
    <div style='padding: 16px; text-align: center;'>
        <h2 style='margin-bottom: 20px; text-align: center;'>🎛️ Filtros</h2>
    """,
    unsafe_allow_html=True
)

all_genres = sorted(list(set([g for sublist in df['genres'] for g in sublist])))
all_option = 'All'
genre_options = [all_option] + all_genres
selected_genres = st.sidebar.multiselect(
    'Género',
    genre_options,
    default=[all_option],
    help='Selecciona uno o varios géneros, o "All" para mostrar todos.'
)

# Ajuste para que 'All' muestre todos los géneros
if all_option in selected_genres or not selected_genres:
    genres_to_filter = all_genres
else:
    genres_to_filter = selected_genres

# Espaciado visual antes de sliders
def sidebar_spacer(height=16):
    st.sidebar.markdown(f"<div style='height: {height}px;'></div>", unsafe_allow_html=True)

sidebar_spacer(8)
years = df['release_date'].dt.year.dropna().astype(int)
year_min, year_max = int(years.min()), int(years.max())
year_range = st.sidebar.slider('Año de estreno', year_min, year_max, (year_min, year_max), key='slider_year', help='Selecciona el rango de años de estreno.')
sidebar_spacer(8)
runtime_range = st.sidebar.slider('Duración (min)', int(df['runtime'].min()), int(df['runtime'].max()), (int(df['runtime'].min()), int(df['runtime'].max())), key='slider_runtime', help='Filtra por duración en minutos.')
sidebar_spacer(16)

st.sidebar.markdown("""<hr style='margin:24px 0;'>""", unsafe_allow_html=True)

# --- Perfiles profesionales ---
st.sidebar.markdown(
    """
    <div style='padding: 8px 8px 8px 8px; text-align: center;'>
        <b>👤 Contacto profesional</b><br>
        <a href='https://www.linkedin.com/in/alejandro-guerra-herrera-a86053115/' target='_blank'>
            <img src='https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg' height='20' style='vertical-align:middle;margin-right:4px;'/> LinkedIn
        </a><br>
        <a href='https://github.com/AlexGHerrera' target='_blank'>
            <img src='https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg' height='20' style='vertical-align:middle;margin-right:4px;'/> GitHub
        </a>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("</div>", unsafe_allow_html=True)

# Filtrado de datos con los géneros seleccionados
def filter_data(df):
    mask = df['genres'].apply(lambda genres: any(g in genres for g in genres_to_filter))
    mask &= df['release_date'].dt.year.between(*year_range)
    mask &= df['runtime'].between(*runtime_range)
    return df[mask]

df_filtered = filter_data(df)

# --- 4. KPIs destacados ---
st.subheader('📊 KPIs destacados')
kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric('Películas seleccionadas', len(df_filtered))
kpi2.metric('Ingresos medios', f"${df_filtered['revenue'].mean():,.0f}")
kpi3.metric('Popularidad máxima', f"{df_filtered['popularity'].max():.2f}")

# --- 5. Visualizaciones dinámicas ---
st.markdown('---')
st.subheader('📈 Visualizaciones interactivas')

# Histograma de ingresos
g1, g2 = st.columns(2)
with g1:
    st.markdown('**Distribución de ingresos**')
    fig, ax = plt.subplots()
    sns.histplot(df_filtered['revenue'], bins=30, ax=ax)
    ax.set_xlabel('Ingresos')
    st.pyplot(fig)

with g2:
    st.markdown('**Distribución de popularidad**')
    fig, ax = plt.subplots()
    sns.histplot(df_filtered['popularity'], bins=30, ax=ax, color='orange')
    ax.set_xlabel('Popularidad')
    st.pyplot(fig)

# Boxplot de ingresos por género
st.markdown('**Boxplot de ingresos por género**')
df_exploded = df_filtered.explode('genres')
top_genres = df_exploded['genres'].value_counts().head(8).index.tolist()
fig, ax = plt.subplots(figsize=(10,4))
sns.boxplot(data=df_exploded[df_exploded['genres'].isin(top_genres)], x='genres', y='revenue')
ax.set_yscale('log')
ax.set_xlabel('Género')
ax.set_ylabel('Ingresos (escala log)')
st.pyplot(fig)

# Scatterplot ingresos vs popularidad
st.markdown('**Ingresos vs. Popularidad**')
fig, ax = plt.subplots()
sns.scatterplot(data=df_filtered, x='popularity', y='revenue', alpha=0.6)
ax.set_xlabel('Popularidad')
ax.set_ylabel('Ingresos')
st.pyplot(fig)

# Top 10 películas por ingresos
top10 = df_filtered.sort_values('revenue', ascending=False).head(10)
st.markdown('**Top 10 películas por ingresos**')
st.dataframe(top10[['title', 'genres', 'revenue', 'popularity', 'vote_average', 'runtime', 'release_date']])

# Wordcloud de títulos (opcional, visual)
if st.checkbox('Mostrar Wordcloud de títulos'):
    st.markdown('**Wordcloud de títulos de películas**')
    text = ' '.join(df_filtered['title'].dropna())
    wordcloud = WordCloud(width=800, height=300, background_color='white', stopwords=STOPWORDS).generate(text)
    fig, ax = plt.subplots(figsize=(10,3))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

# --- 6. Insights y conclusiones ---
st.markdown('---')
st.subheader('💡 Insights clave')
if len(df_filtered) == 0:
    st.warning('No hay películas que cumplan los filtros seleccionados.')
else:
    st.write('- **La popularidad** es el factor más determinante para el éxito en taquilla.')
    st.write('- **Acción, aventura, ciencia ficción y animación** dominan el ranking de ingresos.')
    st.write('- Las películas top suelen tener una duración superior a la media.')
    st.write('- Una buena valoración media no garantiza altos ingresos.')
    st.info('Explora los filtros para descubrir cómo cambian los patrones en función del género, año o duración.')
