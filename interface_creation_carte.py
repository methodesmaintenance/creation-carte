import streamlit as st
import folium
from streamlit.components.v1 import html
import io

import pandas as pd
from sklearn.cluster import KMeans
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from geopy.exc import GeocoderUnavailable, GeocoderTimedOut # Importation des exceptions spécifiques

# Configuration de la page
st.set_page_config(page_title="Générateur de Carte Clustering", layout="wide")

# Initialisation des états de session si ce n'est pas déjà fait
if 'df_original' not in st.session_state:
    st.session_state.df_original = None
if 'df_geocoded' not in st.session_state:
    st.session_state.df_geocoded = None
if 'col_config' not in st.session_state:
    st.session_state.col_config = {}
if 'last_uploaded_file_name' not in st.session_state:
    st.session_state.last_uploaded_file_name = None

st.title("📍 Map Clustering Tool")
st.write("Uploadez un fichier CSV ou Excel pour regrouper vos points par zones.")

# 1. Upload du fichier
uploaded_file = st.sidebar.file_uploader("Choisir un fichier", type=['csv', 'xlsx'])

# Logique pour charger le fichier et gérer les changements
if uploaded_file is not None:
    # Si un nouveau fichier est uploadé, ou si c'est la première fois
    if st.session_state.last_uploaded_file_name != uploaded_file.name:
        st.session_state.last_uploaded_file_name = uploaded_file.name
        # Réinitialiser l'état quand un nouveau fichier est chargé
        st.session_state.df_geocoded = None
        st.session_state.col_config = {} # Réinitialise aussi la config des colonnes
        
        # Lecture des données du nouveau fichier
        try:
            if uploaded_file.name.endswith('.csv'):
                st.session_state.df_original = pd.read_csv(uploaded_file)
            else: # assumed .xlsx
                st.session_state.df_original = pd.read_excel(uploaded_file)
            st.success(f"Fichier '{uploaded_file.name}' chargé avec succès !")
        except Exception as e:
            st.error(f"Erreur lors du chargement du fichier : {e}")
            st.session_state.df_original = None # S'assurer que df_original est None en cas d'erreur
            st.stop() # Arrêter l'exécution pour que l'utilisateur corrige
    
    # Afficher l'extrait du DataFrame original si disponible
    if st.session_state.df_original is not None:
        st.subheader("Extrait des données")
        st.dataframe(st.session_state.df_original.head(5))

        # 2. Configuration des colonnes (basé sur le DataFrame original)
        st.sidebar.header("Configuration des colonnes")
        
        # Pour éviter les erreurs si df_original n'a pas encore été lu
        if st.session_state.df_original is not None:
            # Assurer que les colonnes par défaut existent
            default_name_col = st.session_state.col_config.get('name', st.session_state.df_original.columns[0])
            default_address_col = st.session_state.col_config.get('address', st.session_state.df_original.columns[0])
            default_value_col = st.session_state.col_config.get('value', st.session_state.df_original.columns[0])

            # Assurer que les colonnes par défaut sont toujours dans les options
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
            
            # Stocker la configuration pour la réutilisation
            st.session_state.col_config = {'name': col_name, 'address': col_address, 'value': col_value}

            st.info(f"💡 **Note sur l'adresse :** Pour géocoder ( = calculer longitude et latitude), il est plus fiable d'utiliser seulement le code postal")

            # Bouton de géocodage
            if st.sidebar.button("⚙️ Lancer le Géocodage"):
                with st.spinner("Géocodage en cours... Cela peut prendre un certain temps en fonction du nombre d'adresses"):
                    # Augmentation du délai pour Nominatim
                    geolocator = Nominatim(user_agent="my_clustering_app_v2")
                    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1.5) # Délai augmenté ici

                    # Fonction utilitaire pour préparer l'adresse avant le géocodage (version robuste)
                    def prepare_address_for_geocoding(address):
                        address_str = str(address).strip()
                        if address_str.isdigit() and len(address_str) == 4:
                            address_str = '0' + address_str
                            st.info(f"Correction : '{address}' a été transformé en '{address_str}' pour le géocodage.")
                        # Vérifie si l'adresse est juste un nombre et ne ressemble pas à un CP français (5 chiffres)
                        if address_str.isdigit() and len(address_str) != 5:
                            st.warning(f"Adresse '{address_str}' ne semble pas être valide. Le géocodage pourrait échouer.")
                            return address_str # Ne pas ajouter "France" à un nombre ambigu
                        
                        lower_address = address_str.lower()
                        # Si "France" ou "FR" n'est pas déjà présent
                        if not (('france' in lower_address) or ('fr' in lower_address and len(address_str) > 2)):
                            return f"{address_str}, France"
                        return address_str

                    # Créer une copie du DataFrame original pour le géocodage
                    df_temp = st.session_state.df_original.copy()

                    # On géocode uniquement les adresses uniques pour aller vite
                    unique_addresses_raw = df_temp[col_address].unique()
                    location_map = {}
                    
                    geocoding_errors = 0
                    
                    progress_bar = st.progress(0)
                    total_addresses_to_geocode = len(unique_addresses_raw)
                    
                    for i, addr in enumerate(unique_addresses_raw):
                        prepared_addr = prepare_address_for_geocoding(addr)
                        try:
                            loc = geocode(prepared_addr)
                            if loc:
                                location_map[addr] = (loc.latitude, loc.longitude)
                            else:
                                st.warning(f"Impossible de géocoder l'adresse '{addr}'. Skipped.")
                                geocoding_errors += 1
                        except (GeocoderUnavailable, GeocoderTimedOut) as e:
                            st.warning(f"Timeout ou erreur du service de géocodage pour '{addr}'. {e}. Cette adresse sera ignorée pour le moment.")
                            geocoding_errors += 1
                        except Exception as e: # Pour toute autre erreur inattendue
                            st.warning(f"Erreur inattendue lors du géocodage de '{addr}': {e}. Cette adresse sera ignorée.")
                            geocoding_errors += 1
                        
                        # Mettre à jour la barre de progression, s'assurer que la valeur est entre 0 et 1
                        progress_bar.progress(min(1.0, (i + 1) / total_addresses_to_geocode))

                    df_temp['coords'] = df_temp[col_address].map(location_map)
                    df_temp = df_temp.dropna(subset=['coords'])
                    
                    if df_temp.empty:
                        st.error("Aucune adresse n'a pu être géocodée. Veuillez vérifier la colonne d'adresses et réessayer.")
                        st.session_state.df_geocoded = None
                        st.stop()
                    
                    # Séparer les coordonnées en lat et lon
                    df_temp[['lat', 'lon']] = pd.DataFrame(df_temp['coords'].tolist(), index=df_temp.index)
                    
                    st.session_state.df_geocoded = df_temp.drop(columns=['coords']) # Supprimer la colonne temporaire
                    st.session_state.params = {'name': col_name, 'value': col_value, 'address': col_address}
                    
                    if geocoding_errors > 0:
                        st.warning(f"{geocoding_errors} adresses n'ont pas pu être géocodées. Elles ont été exclues du jeu de données.")
                    st.success("Géocodage terminé ! " + (f"{len(df_temp)} adresses géocodées avec succès." if not df_temp.empty else ""))

                    st.rerun() # Re-exécuter l'application pour afficher le résultat du géocodage

else:
    st.info("Veuillez uploader un fichier dans la barre latérale pour commencer.")


# --- SECTION CLUSTERING ET CARTE ---
if st.session_state.df_geocoded is not None:
    st.sidebar.success("✅ Géocodage terminé ! Vous pouvez maintenant générer la carte.")
    
    df_ready = st.session_state.df_geocoded
    params = st.session_state.params
    col_name = params['name']
    col_value = params['value']
    col_address = params['address'] # Récupérer l'original col_address pour le popup si besoin

    n_clusters = st.sidebar.slider("Nombre de clusters (K)", 2, 20, 5)

    if st.button("🗺️ Générer la carte des clusters"):
        if df_ready.empty:
            st.warning("Aucune donnée géocodée disponible pour générer la carte. Veuillez d'abord géocoder les adresses.")
        else:
            with st.spinner("Calcul des clusters et création de la carte en cours..."):
                
                # --- ÉTAPE : Clustering (K-Means) ---
                # Utilise df_ready qui est le DataFrame géocodé
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10) # Ajout de n_init
                df_ready['cluster'] = kmeans.fit_predict(df_ready[['lat', 'lon']])

                # --- ÉTAPE : Calcul des totaux et centroïdes par cluster ---
                cluster_summary = df_ready.groupby('cluster').agg(
                    lat_centroid=('lat', 'mean'),
                    lon_centroid=('lon', 'mean'),
                    total_value=(col_value, 'sum')
                ).reset_index()

                # --- ÉTAPE : Création de la Carte ---
                # Utiliser les moyennes des coordonnées du DataFrame géocodé
                m = folium.Map(location=[df_ready['lat'].mean(), df_ready['lon'].mean()], zoom_start=6)

                # Couleurs pour les clusters (plus de couleurs pour gérer jusqu'à 20 clusters)
                colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen',
                          'cadetblue', 'pink', 'lightblue', 'lightgreen', 'darkpurple', 'gray', 'black', 'white', 'lightgray', 'darkgray']

                # Ajouter un gestionnaire de progression pour la création de la carte
                map_progress_bar = st.progress(0)
                
                # --- Ajout des marqueurs pour chaque point ---
                total_items_on_map = len(df_ready) + len(cluster_summary) # Points individuels + centroïdes
                current_item_count = 0

                for idx, row in df_ready.iterrows(): 
                    cluster_id = row['cluster']

                    # Gérer le cas où col_address pourrait être un nombre ou un objet non-string
                    address_display = str(row[col_address])

                    popup_text = f"""
                    <b>Nom:</b> {row[col_name]}<br>
                    <b>Adresse:</b> {address_display}<br>
                    <b>Cluster:</b> {cluster_id}<br>
                    <b>Valeur ({col_value}):</b> {row[col_value]}
                    """

                    folium.Marker(
                        location=[row['lat'], row['lon']],
                        popup=folium.Popup(popup_text, max_width=300),
                        icon=folium.Icon(color=colors[cluster_id % len(colors)], icon='info-sign')
                    ).add_to(m)
                    
                    current_item_count += 1
                    map_progress_bar.progress(min(1.0, current_item_count / total_items_on_map))

                # --- Ajout des marqueurs pour les centroïdes ---
                for idx, row in cluster_summary.iterrows():
                    cluster_id = row['cluster']
                    
                    # Popup pour le centroïde
                    centroid_popup_text = f"""
                    <b>Cluster {cluster_id} - CENTROÏDE</b><br>
                    <b>Total Zone ({col_value}):</b> {row['total_value']}
                    """
                    
                    folium.Marker(
                        location=[row['lat_centroid'], row['lon_centroid']],
                        popup=folium.Popup(centroid_popup_text, max_width=300),
                        icon=folium.Icon(color='black', icon='star') # Marqueur noir en forme d'étoile pour les centroïdes
                    ).add_to(m)

                    current_item_count += 1
                    map_progress_bar.progress(min(1.0, current_item_count / total_items_on_map))

                
                st.success("Carte générée !")
                map_html = m._repr_html_()

                # 2. Affichage de la carte dans l'interface
                html(map_html, height=600)

                # 3. Création du bouton de téléchargement
                st.download_button(
                label="💾 Télécharger la carte (HTML)",
                data=map_html,
                file_name="ma_carte_clusters.html",
                mime="text/html"
                )

