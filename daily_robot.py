import subprocess
import logging
import sys
import os
from datetime import datetime, timedelta

# Config
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(command, description, critical=True):
    """Exécute une commande shell et loggue le résultat."""
    logger.info(f"🚀 Démarrage: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"✅ Succès: {description}")
        if result.stdout:
            logger.info(f"Sortie:\n{result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Échec: {description}")
        logger.error(f"Erreur:\n{e.stderr.strip()}")
        if not critical:
            logger.warning("⚠️ Continuons malgré l'erreur (non-critique)")
            return True
        return False


def check_calibration_health():
    """
    Vérifie la santé de la calibration.
    Retourne True si OK, False si recalibration nécessaire.
    """
    import json
    from pathlib import Path
    
    health_path = Path("calibration/health.json")
    if not health_path.exists():
        logger.warning("⚠️ Pas de fichier health.json - calibration requise")
        return False
    
    try:
        with open(health_path, 'r') as f:
            health = json.load(f)
        
        last_calib = datetime.fromisoformat(health.get('last_calibration', '1970-01-01'))
        age_days = (datetime.now() - last_calib).days
        
        # Recalibrer si plus de 7 jours
        if age_days > 7:
            logger.warning(f"⚠️ Calibration vieille de {age_days} jours - recalibration requise")
            return False
        
        # Vérifier les métriques
        metrics = health.get('metrics', {})
        brier = metrics.get('brier_score', 1.0)
        ece = metrics.get('ece', 1.0)
        
        if brier > 0.25 or ece > 0.10:
            logger.warning(f"⚠️ Métriques dégradées (Brier={brier:.3f}, ECE={ece:.3f}) - recalibration requise")
            return False
        
        logger.info(f"✅ Calibration OK (âge: {age_days}j, Brier={brier:.3f}, ECE={ece:.3f})")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur lecture health: {e}")
        return False


def main():
    today = datetime.now().strftime("%Y-%m-%d")
    weekday = datetime.now().weekday()  # 0=Lundi, 6=Dimanche
    
    logger.info("=" * 60)
    logger.info(f"🤖 DAILY ROBOT - {today}")
    logger.info("=" * 60)
    
    # ========================================
    # 1. Mise à jour des résultats (J-1)
    # ========================================
    if not run_command("python update_results.py", "Mise à jour P&L", critical=False):
        logger.warning("⚠️ Problème lors de la mise à jour des résultats (peut-être normal si pas de paris en cours)")

    # ========================================
    # 2. Rapport quotidien
    # ========================================
    run_command("python cli.py report --days 7", "Rapport 7 jours", critical=False)

    # ========================================
    # 3. Health check calibration
    # ========================================
    run_command("python cli.py health", "Health check calibration", critical=False)
    
    # Vérifier si recalibration nécessaire (hebdomadaire ou si dégradé)
    needs_recalibration = not check_calibration_health()
    
    # Recalibration hebdomadaire le dimanche ou si nécessaire
    if weekday == 6 or needs_recalibration:
        logger.info("🎯 Lancement de la recalibration...")
        if not run_command("python cli.py calibrate --days 30", "Recalibration modèle", critical=False):
            logger.warning("⚠️ Échec recalibration - on continue avec les anciens paramètres")

    # ========================================
    # 4. Scraping du jour (J)
    # ========================================
    if not run_command("python scraper_today.py", "Scraping courses du jour"):
        logger.error("⛔ Arrêt du robot : Échec scraping.")
        return

    # ========================================
    # 5. Feature Engineering (J)
    # ========================================
    if not run_command(f"python prepare_daily_features.py --date {today}", "Calcul features du jour"):
        logger.error("⛔ Arrêt du robot : Échec features.")
        return
    
    # ========================================
    # 6. Génération des pronostics via CLI
    # ========================================
    if not run_command(f"python cli.py pick --date {today}", "Génération pronostics", critical=False):
        logger.warning("⚠️ Échec génération pronostics CLI")
        
    # ========================================
    # 7. Paper Trading (J)
    # ========================================
    if not run_command("python paper_trading_v2.py --input data/daily_features.csv", "Génération paris paper trading"):
        logger.error("⛔ Arrêt du robot : Échec paper trading.")
        return
    
    # ========================================
    # 8. Génération tickets exotiques (Quinté du jour si disponible)
    # ========================================
    # Note: Le Quinté+ est généralement couru vers 13h45
    # On génère les tickets pour la course principale si trouvée
    try:
        from db_connection import get_connection
        conn = get_connection()
        cur = conn.cursor()
        
        # Chercher une course avec beaucoup de partants (probable Quinté)
        cur.execute("""
            SELECT race_key, COUNT(*) as n_partants
            FROM cheval_courses_seen
            WHERE date_course = %s
            GROUP BY race_key
            HAVING COUNT(*) >= 10
            ORDER BY n_partants DESC
            LIMIT 1
        """, (today,))
        
        result = cur.fetchone()
        conn.close()
        
        if result:
            race_key = result[0]
            logger.info(f"🎰 Course exotique détectée: {race_key} ({result[1]} partants)")
            run_command(f"python cli.py exotic --race {race_key}", "Génération tickets exotiques", critical=False)
        else:
            logger.info("ℹ️ Pas de course Quinté détectée aujourd'hui")
            
    except Exception as e:
        logger.warning(f"⚠️ Erreur lors de la recherche de course exotique: {e}")
    
    # ========================================
    # Résumé final
    # ========================================
    logger.info("=" * 60)
    logger.info("🎉 CYCLE QUOTIDIEN TERMINÉ AVEC SUCCÈS")
    logger.info("=" * 60)
    logger.info("📁 Fichiers générés:")
    logger.info("   • data/paper_trading_log.csv - Paris du jour")
    logger.info(f"   • data/picks/picks_{today}.json - Pronostics JSON")
    logger.info(f"   • data/picks/portfolio_{today}.yaml - Portfolio YAML")
    logger.info("   • data/exotic/ - Tickets exotiques (si générés)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
