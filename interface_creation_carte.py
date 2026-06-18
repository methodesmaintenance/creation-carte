import streamlit as st
import folium
from streamlit.components.v1 import html
import io
import unicodedata
import os
import uuid
import re

import pandas as pd
from sklearn.cluster import KMeans
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from geopy.exc import GeocoderUnavailable, GeocoderTimedOut

# Configuration du nom de l'application (User-Agent) pour l'API
NOM_USER_AGENT = "generateur_carte_sectorisation_pro_unique_xyz2026"

# Configuration de la page
st.set_page_config(page_title="Générateur de Carte Clustering", layout="wide")

# Fonction pour nettoyer le texte et enlever les .0 textuels résiduels
def clean_text_column(series):
    def clean_value(val):
        if pd.isna(val):
            return val
        text = str(val).strip()
        
        # Sécurité : Si le texte est un nombre entier fini par .0 (ex: "75000.0"), on retire le .0
        if re.match(r'^-?\d+\.0$', text):
            text = text.split('.')[0]
            
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
    st.subheader("Extrait des données chargées")
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

    col_name = st.sidebar.selectbox("Colonne pour le Nom / Identifiant", st.session_state.df_original.columns, 
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
            
            # Lecture sécurisée du Cache CSV
            CACHE_FILE = "cache_geocodage.csv"
            cache_dict = {}
            if os.path.exists(CACHE_FILE) and os.path.getsize(CACHE_FILE) > 0:
                try:
                    df_cache = pd.read_csv(CACHE_FILE)
                    if not df_cache.empty and 'Address' in df_cache.columns:
                        cache_dict = dict(zip(df_cache['Address'], zip(df_cache['Latitude'], df_cache['Longitude'])))
                except Exception:
                    try:
                        os.remove(CACHE_FILE)
                    except:
                        pass

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
                        loc = None
                        method_used = ""
                        clean_zip = prepared_addr.replace(", France", "").strip()
                        
                        if clean_zip.isdigit() and len(clean_zip) == 5:
                            method_used = "Structurée (CP)"
                            loc = geocode({"postalcode": clean_zip, "country": "France"}, country_codes="fr")
                            
                            if not loc:
                                method_used = "Repli Textuel (CP)"
                                loc = geocode(f"{clean_zip}, France", country_codes="fr")
                        else:
                            method_used = "Textuelle Classique"
                            loc = geocode(prepared_addr, country_codes="fr")

                        if loc:
                            location_map[addr] = (loc.latitude, loc.longitude)
                            new_cache_entries.append({
                                'Address': prepared_addr, 'Latitude': loc.latitude, 'Longitude': loc.longitude
                            })
                            st.session_state.geocoding_debug_logs.append({
                                "Adresse/Événement": addr, "Statut": "🔵 API SUCCÈS", "Message": f"Trouvé via {method_used} -> {loc.latitude}, {loc.longitude}"
                            })
                        else:
                            geocoding_errors += 1
                            location_map[addr] = ("Erreur", "Erreur")  # Écrit "Erreur" si l'algo ne trouve rien
                            st.session_state.geocoding_debug_logs.append({
                                "Adresse/Événement": addr, "Statut": "🔴 INTROUVABLE", "Message": f"Aucun résultat trouvé en France pour '{prepared_addr}'."
                            })
                    except Exception as e:
                        geocoding_errors += 1
                        location_map[addr] = ("Erreur", "Erreur")  # Écrit "Erreur" en cas de plantage / Timeout
                        st.session_state.geocoding_debug_logs.append({
                            "Adresse/Événement": addr, "Statut": "⚠️ ERREUR API", "Message": str(e)
                        })
                
                progress_bar.progress(min(1.0, (i + 1) / total_addresses_to_geocode))

            # Sauvegarde du cache
            if new_cache_entries:
                df_new_cache = pd.DataFrame(new_cache_entries)
                try:
                    if os.path.exists(CACHE_FILE) and os.path.getsize(CACHE_FILE) > 0:
                        df_new_cache.to_csv(CACHE_FILE, mode='a', header=False, index=False)
                    else:
                        df_new_cache.to_csv(CACHE_FILE, mode='w', header=True, index=False)
                except Exception as e:
                    pass

            # Mapping global (Sans suppression des lignes introuvables)
            df_temp['coords'] = df_temp[col_address].map(location_map)
            df_temp['lat'] = df_temp['coords'].apply(lambda x: x[0] if isinstance(x, tuple) else "Erreur")
            df_temp['lon'] = df_temp['coords'].apply(lambda x: x[1] if isinstance(x, tuple) else "Erreur")
            
            st.session_state.df_geocoded = df_temp.drop(columns=['coords'])
            st.session_state.params = {'name': col_name, 'value': col_value, 'address': col_address}
            st.rerun()


# --- SECTION CLUSTERING ET CARTE ---
if st.session_state.df_geocoded is not None:
    st.sidebar.success("✅ Géocodage terminé !")

    df_ready = st.session_state.df_geocoded.copy()
    params = st.session_state.params
    col_name = params['name']
    col_value = params['value']
    col_address = params['address']

    st.sidebar.markdown("---")
    st.sidebar.header("🎯 Mode de répartition")
    
    clustering_mode = st.sidebar.radio(
        "Choisir la méthode de regroupement :",
        ["Sectorisation intelligente", "Regroupement par colonne", "Rattachement à l'agence la plus proche"]
    )

    if clustering_mode == "Sectorisation intelligente":
        # Compte uniquement les points géocodés avec succès pour calibrer le slider
        df_count_valid = df_ready[(df_ready['lat'] != "Erreur") & (df_ready['lon'] != "Erreur")]
        n_points_disponibles = len(df_count_valid)
        n_clusters_ajuste = max(1, min(20, n_points_disponibles))
        
        st.sidebar.caption("💡 Calcule automatiquement les zones à vol d'oiseau selon le nombre demandé.")
        n_clusters = st.sidebar.slider("Nombre de secteurs souhaités", 1, n_clusters_ajuste, min(5, n_clusters_ajuste))
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
    st.sidebar.header("➕ Ajouter des points agences")
    st.sidebar.write("Nom, Latitude, Longitude (un point par ligne) :")
    st.sidebar.code("Agence Lyon,45.777863,5.034605\nAgence Clermont-Ferrand,45.780796,3.2125044")
    
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

    # --- BOUTON DE GÉNÉRATION DE LA CARTE ---
    if st.button("🗺️ Générer la carte des secteurs"):
        with st.spinner("Calcul des secteurs et génération de la carte..."):
            df_ready[col_value] = pd.to_numeric(df_ready[col_value], errors='coerce').fillna(0)
            
            # SÉPARATION STRICTE : Points Valides vs Points Erreurs
            df_valid = df_ready[(df_ready['lat'] != "Erreur") & (df_ready['lon'] != "Erreur")].copy()
            df_errors = df_ready[(df_ready['lat'] == "Erreur") | (df_ready['lon'] == "Erreur")].copy()
            
            if df_valid.empty:
                st.error("Aucune coordonnée valide n'a pu être trouvée pour générer la carte.")
                st.stop()
            
            # Conversion forcée en numérique uniquement pour les calculs géographiques
            df_valid['lat'] = pd.to_numeric(df_valid['lat'])
            df_valid['lon'] = pd.to_numeric(df_valid['lon'])
            
            # --- CALCUL DES CLUSTERS SUR LES POINTS VALIDES ---
            if clustering_mode == "Sectorisation intelligente":
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                df_valid['cluster'] = kmeans.fit_predict(df_valid[['lat', 'lon']])
                df_valid['Zone'] = df_valid['cluster'] + 1
                
            elif clustering_mode == "Regroupement par colonne":
                unique_values = sorted(df_valid[group_column].dropna().unique())
                cluster_mapping = {val: idx for idx, val in enumerate(unique_values)}
                df_valid['cluster'] = df_valid[group_column].map(cluster_mapping)
                reverse_mapping = {v: k for k, v in cluster_mapping.items()}
                df_valid['Zone'] = df_valid['cluster'].map(reverse_mapping)
                
            elif use_agency_clustering:
                from scipy.spatial.distance import cdist
                points_coords = df_valid[['lat', 'lon']].values
                agency_coords = st.session_state.agences_df[['Latitude', 'Longitude']].values
                distances = cdist(points_coords, agency_coords, metric='euclidean')
                df_valid['cluster'] = distances.argmin(axis=1)
                df_valid['Zone'] = df_valid['cluster'].apply(lambda idx: st.session_state.agences_df.iloc[idx]['Name'])

            # Attribution d'une zone spécifique par défaut pour les erreurs de géocodage
            if not df_errors.empty:
                df_errors['cluster'] = -1
                df_errors['Zone'] = "Erreur Géocodage"

            # --- CONSTRUIRE LA CARTE (Points valides uniquement) ---
            grouped_points = df_valid.groupby(['lat', 'lon', 'cluster']).agg(
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

            m = folium.Map(location=[df_valid['lat'].mean(), df_valid['lon'].mean()], zoom_start=6)
            colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen',
                      'cadetblue', 'pink', 'lightblue', 'lightgreen', 'darkpurple', 'gray', 'black']

            for idx, row_grouped in grouped_points.iterrows(): 
                cluster_id = int(row_grouped['cluster'])
                
                if clustering_mode == "Regroupement par colonne":
                    cluster_label = f"Groupe : {df_valid[df_valid['cluster'] == cluster_id]['Zone'].iloc[0]}"
                elif use_agency_clustering:
                    cluster_label = f"Rattaché à : {df_valid[df_valid['cluster'] == cluster_id]['Zone'].iloc[0]}"
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

            # Affichage des Repères / Centrides
            if st.session_state.show_centroids:
                if use_agency_clustering:
                    for idx, row in st.session_state.agences_df.iterrows():
                        total_secteur = df_valid[df_valid['cluster'] == idx][col_value].sum()
                        popup_agency = f"<b>🏢 {row['Name']}</b><br><b>{label_total} secteur:</b> {total_secteur:.2f}"
                        folium.Marker(
                            location=[row['Latitude'], row['Longitude']],
                            popup=folium.Popup(popup_agency, max_width=250),
                            icon=folium.Icon(color='black', icon='building', prefix='fa')
                        ).add_to(m)
                else:
                    cluster_summary = df_valid.groupby('cluster').agg(
                        lat_centroid=('lat', 'mean'), lon_centroid=('lon', 'mean'), total_value=(col_value, 'sum')
                    ).reset_index()
                    for idx, row in cluster_summary.iterrows():
                        c_id = int(row['cluster'])
                        lbl = f"Centre Zone : {df_valid[df_valid['cluster'] == c_id]['Zone'].iloc[0]}" if clustering_mode == "Regroupement par colonne" else f"Centre Secteur {c_id + 1}"
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

            # Bouton de téléchargement de la carte HTML
            st.download_button(
                label="💾 Télécharger la carte visuelle (HTML)", data=map_html, file_name="carte_sectorisation.html", mime="text/html"
            )

            # --- RECOMBINAISON ET PRÉPARATION DE L'EXPORT STRICT (4 COLONNES) ---
            df_export_ready = pd.concat([df_valid, df_errors], ignore_index=True)
            df_export = df_export_ready[[col_name, 'lat', 'lon', 'Zone']].copy()
            df_export.columns = ['Identifiant', 'Latitude', 'Longitude', 'Zone']

            # NETTOYAGE STRICT DES .0 (Ne modifie jamais un vrai float comme 45.1234)
            for col in ['Identifiant', 'Zone']:
                # On ne traite que si la colonne contient des types Float natifs
                if pd.api.types.is_float_dtype(df_export[col]):
                    # On vérifie si TOUTES les valeurs non-nulles sont des entiers (ex: 12.0, 75000.0)
                    if (df_export[col].dropna() % 1 == 0).all():
                        df_export[col] = df_export[col].astype('Int64')

            # Nouveau bouton de téléchargement de données restreint (séparateur point-virgule)
            st.download_button(
                label="⬇️ Télécharger le tableau complet des secteurs (CSV ;)",
                data=df_export.to_csv(index=False, sep=';').encode('utf-8'),
                file_name="donnees_sectorisees.csv",
                mime="text/csv"
            )

# Log d'erreurs en bas de page pour le diagnostic
if st.session_state.geocoding_debug_logs:
    with st.expander("🔍 Afficher la console de diagnostic du Géocodage"):
        st.dataframe(pd.DataFrame(st.session_state.geocoding_debug_logs))
