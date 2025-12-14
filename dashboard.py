import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# Config
st.set_page_config(page_title="Horse AI Dashboard", layout="wide")
LOG_PATH = "data/paper_trading_log.csv"

def load_data():
    if not os.path.exists(LOG_PATH):
        return pd.DataFrame()
    df = pd.read_csv(LOG_PATH)
    # Conversion des dates
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    return df

def main():
    st.title("🏇 Horse AI - Monitoring Robot")
    
    df = load_data()
    
    if df.empty:
        st.warning("Aucune donnée de pari trouvée.")
        return
        
    # Sidebar filtres
    st.sidebar.header("Filtres")
    if 'date' in df.columns:
        min_date = df['date'].min().date()
        max_date = df['date'].max().date()
        date_range = st.sidebar.date_input("Période", [min_date, max_date])
    
    # KPI Globaux
    st.header("📊 Performance Globale")
    
    # Calculs
    # On ne compte que les paris terminés (Gagné/Perdu)
    df_finished = df[df['statut'].isin(['Gagné', 'Perdu'])].copy()
    
    total_bets = len(df)
    pending_bets = len(df[df['statut'] == 'En cours'])
    finished_bets = len(df_finished)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Paris", total_bets)
    col2.metric("En Cours", pending_bets)
    col3.metric("Terminés", finished_bets)
    
    if finished_bets > 0:
        # Calcul P&L
        # Si Gagné: Gain = mise * cote - mise
        # Si Perdu: Perte = -mise
        # Note: Dans le log, on a 'gain_net' normalement si update_results a tourné
        if 'gain_net' not in df_finished.columns:
            # Fallback calcul si pas encore mis à jour
            df_finished['gain_net'] = df_finished.apply(
                lambda x: (x['mise'] * x['cote'] - x['mise']) if x['statut'] == 'Gagné' else -x['mise'], axis=1
            )
            
        total_pnl = df_finished['gain_net'].sum()
        roi = (total_pnl / df_finished['mise'].sum()) * 100
        win_rate = (len(df_finished[df_finished['statut'] == 'Gagné']) / finished_bets) * 100
        
        col4.metric("P&L Net", f"{total_pnl:.2f} €", delta_color="normal")
        
        st.subheader(f"ROI: {roi:.2f}% | Win Rate: {win_rate:.2f}%")
        
        # Graphique P&L Cumulé
        df_finished = df_finished.sort_values('date')
        df_finished['pnl_cumul'] = df_finished['gain_net'].cumsum()
        
        fig = px.line(df_finished, x='date', y='pnl_cumul', title="Évolution du P&L Cumulé", markers=True)
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.info("En attente de résultats de courses...")

    # Détail des paris récents
    st.header("📝 Derniers Paris")
    st.dataframe(df.sort_values('date', ascending=False).head(50))
    
    # Analyse par probabilité
    if finished_bets > 0:
        st.header("🎯 Analyse Calibration")
        df_finished['proba_bin'] = pd.cut(df_finished['proba_modele'], bins=5)
        calibration = df_finished.groupby('proba_bin').agg(
            win_rate=('statut', lambda x: (x == 'Gagné').mean()),
            count=('statut', 'count')
        ).reset_index()
        
        fig_calib = px.bar(calibration, x='proba_bin', y='win_rate', title="Taux de victoire réel par tranche de probabilité prédite")
        st.plotly_chart(fig_calib)

if __name__ == "__main__":
    main()
