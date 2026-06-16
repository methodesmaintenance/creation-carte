import streamlit as st
import folium
from streamlit.components.v1 import html
import io
import unicodedata
import os
import uuid

import pandas as pd
from sklearn.cluster import KMeans
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from geopy.exc import GeocoderUnavailable, GeocoderTimedOut

# Configuration du nom de l'application (User-Agent) pour l'API
NOM_USER_AGENT = "generateur_carte_sectorisation_pro_unique_xyz2026"

# Configuration de la page
st.set_page_config(page_title="Générateur de Carte Clustering", layout="wide")

# Fonction pour nettoyer le texte
def clean_text_column(series):
    def clean_value(val):
        if pd.isna(val):
            return val
        text = str(val).strip()
        text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
        text = text.upper()
        text = " ".join(text.split())
        return text
    return series.apply(clean_value)

# Initialisation des états de session
if 'df_original' not in st.session_state:
    st.session_state.df_original = None
if 'df_geocoded' not in st.session_state:
    st.session_state.df_geocoded = None
if 'col_config' not in st.session_state:
    st.session_state.col_config = {}
if 'last_uploaded_file_name' not in st.session_state:
    st.session_state.last_uploaded_file_name = None
if 'show_centroids' not in st.session_state:
    st.session_state.show_centroids = True
if 'manual_points_df' not in st.session_state:
    st.session_state.manual_points_df = None
if 'geocoding_debug_logs' not in st.session_state:
    st.session_state.geocoding_debug_logs = []

if 'agences_df' not in st.session_state:
    st.session_state.agences_df = pd.DataFrame({
        'Name': ['Agence Lyon', 'Agence Clermont-Ferrand', 'Agence Creuzier le Neuf', 
                 'Agence Saint-Etienne', 'Agence Grenoble', 'Agence Aix-les-Bains'],
        'Latitude': [45.777863, 45.780796, 46.163277, 45.437602, 45.137359, 45.697425],
        'Longitude': [5.034605, 3.2125044, 3.411502, 4.331476, 5.706871, 5.9274654]
    })

st.title("📍 Générateur de Carte de Sectorisation")
st.write("Uploadez un fichier CSV ou Excel pour regrouper vos points par zones géographiques.")

uploaded_file = st.sidebar.file_uploader("Choisir un fichier", type=['csv', 'xlsx'])

if uploaded_file is not None:
    if st.session_state.last_uploaded_file_name != uploaded_file.name:
        st.session_state.last_uploaded_file_name = uploaded_file.name
        st.session_state.df_geocoded = None
        st.session_state.col_config = {}
        st.session_state.manual_points_df = None
        st.session_state.geocoding_debug_logs = []

        try:
            if uploaded_file.name.endswith('.csv'):
                df_loaded = pd.read_csv(uploaded_file)
            else:
                df_loaded = pd.read_excel(uploaded_file)
            
            for col in df_loaded.columns:
                if df_loaded[col].dtype == 'object':
                    df_loaded[col] = clean_text_column(df_loaded[col])
                    
            st.session_state.df_original = df_loaded
            st.success(f"Fichier '{uploaded_file.name}' chargé et nettoyé avec succès !")
        except Exception as e:
            st.error(f"Erreur lors du chargement du fichier : {e}")
            st.session_state.df_original = None
            st.stop()

if st.session_state.df_original is not None:
    st.subheader("Extrait des données")
    st.dataframe(st.session_state.df_original.head(5))

    st.sidebar.header("Configuration des colonnes")

    default_name_col = st.session_state.col_config.get('name', st.session_state.df_original.columns[0])
    default_address_col = st.session_state.col_config.get('address', st.session_state.df_original.columns[0])
    default_value_col = st.session_state.col_config.get('value', st.session_state.df_original.columns[0])

    if default_name_col not in st.session_state.df_original.columns:
        default_name_col = st.session_state.df_original.columns[0]
    if default_address_col not in st.session_state.df_original.columns:
        default_address_col = st.session_state.df_original.columns[0]
    if default_value_col not in st.session_state.df_original.columns:
        default_value_col = st.session_state.df_original.columns[0]

    col_name = st.sidebar.selectbox("Colonne pour le Nom", st.session_state.df_original.columns, 
                                    index=list(st.session_state.df_original.columns).index(default_name_col))
    col_address = st.sidebar.selectbox("Colonne pour l'Adresse / CP", st.session_state.df_original.columns,
                                       index=list(st.session_state.df_original.columns).index(default_address_col))
    col_value = st.sidebar.selectbox("Colonne à additionner (ex: Heures)", st.session_state.df_original.columns,
                                     index=list(st.session_state.df_original.columns).index(default_value_col))
    
    st.session_state.col_config = {'name': col_name, 'address': col_address, 'value': col_value}

    st.info(f"💡 **Note sur l'adresse :** Pour géocoder (= calculer la position sur la carte), il est plus fiable d'utiliser un code postal.")

    if st.sidebar.button("⚙️ Lancer le Géocodage"):
        with st.spinner("Géocodage en cours..."):
            st.session_state.geocoding_debug_logs = [] 
            
            geolocator = Nominatim(user_agent=NOM_USER_AGENT, timeout=6)
            geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1.5)

            def prepare_address_for_geocoding(address):
                address_str = str(address).strip()
                if address_str.isdigit() and len(address_str) == 4:
                    address_str = '0' + address_str
                if address_str.isdigit() and len(address_str) != 5:
                    return address_str
                
                lower_address = address_str.lower()
                if not (('france' in lower_address) or ('fr' in lower_address and len(address_str) > 2)):
                    return f"{address_str}, France"
                return address_str

            df_temp = st.session_state.df_original.copy()
            unique_addresses_raw = df_temp[col_address].unique()
            location_map = {}
            
            geocoding_errors = 0
            progress_bar = st.progress(0)
            total_addresses_to_geocode = len(unique_addresses_raw)
            
            # --- CORRECTIF 1 : Lecture sécurisée du Cache ---
            CACHE_FILE = "cache_geocodage.csv"
            cache_dict = {}
            if os.path.exists(CACHE_FILE) and os.path.getsize(CACHE_FILE) > 0: # Vérification de la taille !
                try:
                    df_cache = pd.read_csv(CACHE_FILE)
                    cache_dict = dict(zip(df_cache['Address'], zip(df_cache['Latitude'], df_cache['Longitude'])))
                except Exception as e:
                    st.session_state.geocoding_debug_logs.append({
                        "Adresse/Événement": "Système de Cache", "Statut": "Erreur", "Message": str(e)
                    })

            new_cache_entries = []
            
            for i, addr in enumerate(unique_addresses_raw):
                prepared_addr = prepare_address_for_geocoding(addr)
                
                if prepared_addr in cache_dict:
                    location_map[addr] = cache_dict[prepared_addr]
                    st.session_state.geocoding_debug_logs.append({
                        "Adresse/Événement": addr, "Statut": "🟢 CACHE", "Message": "Récupéré du fichier local sans appel API."
                    })
                else:
                    try:
                        # --- CORRECTIF 2 : Requête structurée pour le Cloud ---
                        # On extrait le code postal s'il s'agit d'un nombre pur à 5 chiffres (ex: 69008 ou 69008, France)
                        clean_zip = prepared_addr.replace(", France", "").strip()
                        if clean_zip.isdigit() and len(clean_zip) == 5:
                            # Forçage d'une recherche structurée par dictionnaire (Zéro dépendance à l'IP du serveur)
                            query_param = {"postalcode": clean_zip, "country": "France"}
                            log_msg = f"Recherche structurée par Code Postal lancée pour : {clean_zip}"
                        else:
                            query_param = prepared_addr
                            log_msg = "Recherche textuelle classique lancée."

                        loc = geocode(query_param)
                        
                        if loc:
                            location_map[addr] = (loc.latitude, loc.longitude)
                            new_cache_entries.append({
                                'Address': prepared_addr, 'Latitude': loc.latitude, 'Longitude': loc.longitude
                            })
                            st.session_state.geocoding_debug_logs.append({
                                "Adresse/Événement": addr, "Statut": "🔵 API SUCCÈS", "Message": f"Coordonnées trouvées : {loc.latitude}, {loc.longitude} ({log_msg})"
                            })
                        else:
                            geocoding_errors += 1
                            st.session_state.geocoding_debug_logs.append({
                                "Adresse/Événement": addr, "Statut": "实用 INTROUVABLE", "Message": "L'API a répondu mais n'a pas trouvé cet emplacement sur les index mondiaux."
                            })
                    except GeocoderTimedOut as e:
                        geocoding_errors += 1
                        st.session_state.geocoding_debug_logs.append({
                            "Adresse/Événement": addr, "Statut": "🔴 TIMEOUT", "Message": f"Le serveur a mis trop de temps à répondre : {e}"
                        })
                    except GeocoderUnavailable as e:
                        geocoding_errors += 1
                        st.session_state.geocoding_debug_logs.append({
                            "Adresse/Événement": addr, "Statut": "❌ REJETÉ", "Message": f"IP Cloud bannie. Détails : {e}"
                        })
                    except Exception as e:
                        geocoding_errors += 1
                        st.session_state.geocoding_debug_logs.append({
                            "Adresse/Événement": addr, "Statut": "⚠️ ERREUR INCONNUE", "Message": str(e)
                        })
                
                progress_bar.progress(min(1.0, (i + 1) / total_addresses_to_geocode))

            # Sauvegarde sécurisée du cache CSV
            if new_cache_entries:
                df_new_cache = pd.DataFrame(new_cache_entries)
                try:
                    if os.path.exists(CACHE_FILE) and os.path.getsize(CACHE_FILE) > 0:
                        df_new_cache.to_csv(CACHE_FILE, mode='a', header=False, index=False)
                    else:
                        df_new_cache.to_csv(CACHE_FILE, mode='w', header=True, index=False)
                except Exception as e:
                    st.session_state.geocoding_debug_logs.append({
                        "Adresse/Événement": "Système de Cache", "Statut": "Erreur Écriture", "Message": str(e)
                    })

            df_temp['coords'] = df_temp[col_address].map(location_map)
            df_temp = df_temp.dropna(subset=['coords'])
            
            if df_temp.empty:
                st.error("Aucune adresse n'a pu être localisée. Regardez la console de diagnostic ci-dessous.")
                st.session_state.df_geocoded = None
                st.stop()
            
            df_temp[['lat', 'lon']] = pd.DataFrame(df_temp['coords'].tolist(), index=df_temp.index)
            st.session_state.df_geocoded = df_temp.drop(columns=['coords'])
            st.session_state.params = {'name': col_name, 'value': col_value, 'address': col_address}
            st.rerun()

# --- ZONE CONSOLE DE DIAGNOSTIC ---
if st.session_state.geocoding_debug_logs:
    st.markdown("---")
    with st.expander("🛠 Honoraires de la Console de Diagnostic", expanded=True):
        df_logs = pd.DataFrame(st.session_state.geocoding_debug_logs)
        st.dataframe(df_logs, use_container_width=True)

# --- SECTION CLUSTERING ET CARTE ---
if st.session_state.df_geocoded is not None:
    st.sidebar.success("✅ Géocodage terminé !")

    df_ready = st.session_state.df_geocoded.copy()
    params = st.session_state.params
    col_name = params['name']
    col_value = params['value']
    col_address = params['address']

    st.download_button(
        label="⬇️ Télécharger les données géocodées (CSV)",
        data=st.session_state.df_geocoded.to_csv(index=False).encode('utf-8'),
        file_name="donnees_geocodes.csv",
        mime="text/csv"
    )

    st.sidebar.markdown("---")
    st.sidebar.header("🎯 Mode de répartition")
    
    clustering_mode = st.sidebar.radio(
        "Choisir la méthode de regroupement :",
        ["Sectorisation intelligente", "Regroupement par colonne", "Rattachement à l'agence la plus proche"]
    )

    if clustering_mode == "Sectorisation intelligente":
        st.sidebar.caption("💡 Calcule automatiquement les zones à vol d'oiseau selon le nombre demandé.")
        n_clusters = st.sidebar.slider("Nombre de secteurs souhaités", 1, 20, 5)
        group_column = None
        use_agency_clustering = False
        
    elif clustering_mode == "Regroupement par colonne":
        st.sidebar.caption("💡 Idéal si vos données contiennent déjà une colonne de répartition.")
        group_column = st.sidebar.selectbox("Choisir la colonne de regroupement :", st.session_state.df_original.columns)
        n_clusters = None
        use_agency_clustering = False
        
    elif clustering_mode == "Rattachement à l'agence la plus proche":
        st.sidebar.caption("💡 Associe chaque point client à l'agence physique la plus proche à vol d'oiseau.")
        n_clusters = None
        group_column = None
        use_agency_clustering = True

    st.session_state.show_centroids = st.sidebar.checkbox("Afficher les centres géographiques / Agences repères", st.session_state.show_centroids)

    # --- SECTION POINTS MANUELS ---
    st.sidebar.markdown("---")
    st.sidebar.header("➕ Ajouter des points")
    manual_points_input = st.sidebar.text_area("Saisissez vos points ici :", key="manual_points_input")

    if st.sidebar.button("➕ Ajouter ces points"):
        if manual_points_input:
            try:
                data = [line.split(',') for line in manual_points_input.strip().split('\n') if line.strip()]
                manual_df = pd.DataFrame(data, columns=['Name', 'Latitude', 'Longitude'])
                manual_df['Latitude'] = pd.to_numeric(manual_df['Latitude'])
                manual_df['Longitude'] = pd.to_numeric(manual_df['Longitude'])
                st.session_state.manual_points_df = manual_df
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Erreur de format : {e}")

    if st.session_state.manual_points_df is not None and not st.session_state.manual_points_df.empty:
        st.sidebar.dataframe(st.session_state.manual_points_df)
        if st.sidebar.button("❌ Supprimer les points manuels"):
            st.session_state.manual_points_df = None
            st.rerun()

    # Bouton de génération de carte
    if st.button("🗺️ Générer la carte des secteurs"):
        with st.spinner("Calcul des secteurs et génération de la carte..."):
            df_ready[col_value] = pd.to_numeric(df_ready[col_value], errors='coerce').fillna(0)
            
            # --- CALCULS ---
            if clustering_mode == "Sectorisation intelligente":
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                df_ready['cluster'] = kmeans.fit_predict(df_ready[['lat', 'lon']])
                
            elif clustering_mode == "Regroupement par colonne":
                unique_values = sorted(df_ready[group_column].dropna().unique())
                cluster_mapping = {val: idx for idx, val in enumerate(unique_values)}
                df_ready['cluster'] = df_ready[group_column].map(cluster_mapping)
                
            elif use_agency_clustering:
                from scipy.spatial.distance import cdist
                points_coords = df_ready[['lat', 'lon']].values
                agency_coords = st.session_state.agences_df[['Latitude', 'Longitude']].values
                distances = cdist(points_coords, agency_coords, metric='euclidean')
                df_ready['cluster'] = distances.argmin(axis=1)

            grouped_points = df_ready.groupby(['lat', 'lon', 'cluster']).agg(
                names=(col_name, lambda x: '<br>'.join(x.astype(str))),
                total_value=(col_value, 'sum'),
                address=(col_address, 'first')
            ).reset_index()

            col_lower = col_value.lower().strip()
            if "heure" in col_lower:
                label_total = f"Total des {col_value}"
            elif col_lower.startswith(('a', 'e', 'é', 'i', 'o', 'u', 'nb', 'nombre')):
                label_total = f"Total des {col_value}" if "nb" not in col_lower and "nombre" not in col_lower else f"{col_value} total"
            else:
                label_total = f"{col_value} total"

            m = folium.Map(location=[df_ready['lat'].mean(), df_ready['lon'].mean()], zoom_start=6)
            colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen',
                      'cadetblue', 'pink', 'lightblue', 'lightgreen', 'darkpurple', 'gray', 'black']

            for idx, row_grouped in grouped_points.iterrows(): 
                cluster_id = int(row_grouped['cluster'])
                
                if clustering_mode == "Regroupement par colonne":
                    reverse_mapping = {v: k for k, v in cluster_mapping.items()}
                    cluster_label = f"Groupe : {reverse_mapping.get(cluster_id, cluster_id)}"
                elif use_agency_clustering:
                    agency_name = st.session_state.agences_df.iloc[cluster_id]['Name']
                    cluster_label = f"Rattaché à : {agency_name}"
                else:
                    cluster_label = f"Secteur : {cluster_id + 1}"

                popup_text = f"""
                <b>Nom(s):</b> {row_grouped['names']}<br>
                <b>Adresse:</b> {row_grouped['address']}<br>
                <b>{cluster_label}</b><br>
                <b>{label_total}:</b> {row_grouped['total_value']:.2f}
                """
                folium.Marker(
                    location=[row_grouped['lat'], row_grouped['lon']],
                    popup=folium.Popup(popup_text, max_width=300),
                    icon=folium.Icon(color=colors[cluster_id % len(colors)], icon='info-sign')
                ).add_to(m)

            if st.session_state.show_centroids:
                if use_agency_clustering:
                    for idx, row in st.session_state.agences_df.iterrows():
                        total_secteur = df_ready[df_ready['cluster'] == idx][col_value].sum()
                        popup_agency = f"<b>🏢 {row['Name']}</b><br><b>{label_total} secteur:</b> {total_secteur:.2f}"
                        folium.Marker(
                            location=[row['Latitude'], row['Longitude']],
                            popup=folium.Popup(popup_agency, max_width=250),
                            icon=folium.Icon(color='black', icon='building', prefix='fa')
                        ).add_to(m)
                else:
                    cluster_summary = df_ready.groupby('cluster').agg(
                        lat_centroid=('lat', 'mean'), lon_centroid=('lon', 'mean'), total_value=(col_value, 'sum')
                    ).reset_index()
                    for idx, row in cluster_summary.iterrows():
                        c_id = int(row['cluster'])
                        if clustering_mode == "Regroupement par colonne":
                            reverse_mapping = {v: k for k, v in cluster_mapping.items()}
                            lbl = f"Centre Zone : {reverse_mapping.get(c_id, f'Groupe {c_id}')}"
                        else:
                            lbl = f"Centre Secteur {c_id + 1}"
                        popup_centroid = f"<b>⭐ {lbl}</b><br><b>{label_total} Zone:</b> {row['total_value']:.2f}"
                        folium.Marker(
                            location=[row['lat_centroid'], row['lon_centroid']],
                            popup=folium.Popup(popup_centroid, max_width=250),
                            icon=folium.Icon(color='black', icon='star', prefix='fa')
                        ).add_to(m)

            if st.session_state.manual_points_df is not None:
                for idx, row in st.session_state.manual_points_df.iterrows():
                    folium.Marker(
                        location=[row['Latitude'], row['Longitude']],
                        popup=folium.Popup(f"📍 {row['Name']}", max_width=200),
                        icon=folium.Icon(color='black', icon='home', prefix='fa')
                    ).add_to(m)

            st.success("Carte générée avec succès !")
            map_html = m._repr_html_()
            html(map_html, height=600)

            st.download_button(
                label="💾 Télécharger la carte (HTML)", data=map_html, file_name="carte_sectorisation.html", mime="text/html"
            )
