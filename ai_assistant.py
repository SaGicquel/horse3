import os
import logging
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd
from db_connection import get_connection

# Configuration logging
logger = logging.getLogger(__name__)

class HorseRacingAssistant:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = None
        if self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
            except ImportError:
                logger.warning("OpenAI library not installed. Please install it with `pip install openai`.")
        else:
            logger.warning("OPENAI_API_KEY not found in environment variables.")

    def get_system_prompt(self) -> str:
        return """Tu es un expert en courses hippiques et en analyse de données. 
Ton rôle est d'aider les utilisateurs à analyser les performances des chevaux, jockeys et entraîneurs.
Tu as accès à une base de données de courses hippiques.
Tu dois fournir des réponses précises, basées sur les données, et des conseils stratégiques pertinents.
Si tu ne connais pas la réponse, dis-le honnêtement.
Sois concis, professionnel et encourageant.
"""

    def query_db(self, sql: str, params: tuple = None) -> List[Dict[str, Any]]:
        """Exécute une requête SQL sécurisée et retourne les résultats."""
        conn = None
        try:
            conn = get_connection()
            with conn.cursor() as cur:
                cur.execute(sql, params)
                columns = [desc[0] for desc in cur.description]
                results = [dict(zip(columns, row)) for row in cur.fetchall()]
                return results
        except Exception as e:
            logger.error(f"Database query error: {e}")
            return []
        finally:
            if conn:
                conn.close()

    def get_top_horses(self, limit: int = 5) -> str:
        """Retourne les chevaux les plus performants (basé sur le taux de victoire)."""
        sql = """
            SELECT c.nom, COUNT(p.id) as courses, 
                   SUM(CASE WHEN p.place = 1 THEN 1 ELSE 0 END) as victoires,
                   ROUND(CAST(SUM(CASE WHEN p.place = 1 THEN 1 ELSE 0 END) AS NUMERIC) / COUNT(p.id) * 100, 2) as taux_victoire
            FROM participants p
            JOIN chevaux c ON p.cheval_id = c.id
            GROUP BY c.id, c.nom
            HAVING COUNT(p.id) >= 10
            ORDER BY taux_victoire DESC
            LIMIT %s
        """
        results = self.query_db(sql, (limit,))
        if not results:
            return "Je n'ai pas trouvé de données suffisantes pour déterminer les meilleurs chevaux."
        
        response = "Voici les chevaux les plus performants (min. 10 courses) :\n"
        for r in results:
            response += f"- **{r['nom']}** : {r['taux_victoire']}% de victoires ({r['victoires']}/{r['courses']} courses)\n"
        return response

    def get_top_jockeys(self, limit: int = 5) -> str:
        """Retourne les meilleurs jockeys."""
        sql = """
            SELECT j.nom, COUNT(p.id) as courses,
                   SUM(CASE WHEN p.place = 1 THEN 1 ELSE 0 END) as victoires,
                   ROUND(CAST(SUM(CASE WHEN p.place = 1 THEN 1 ELSE 0 END) AS NUMERIC) / COUNT(p.id) * 100, 2) as taux_victoire
            FROM participants p
            JOIN jockeys j ON p.jockey_id = j.id
            GROUP BY j.id, j.nom
            HAVING COUNT(p.id) >= 20
            ORDER BY taux_victoire DESC
            LIMIT %s
        """
        results = self.query_db(sql, (limit,))
        if not results:
            return "Je n'ai pas trouvé de données suffisantes pour les jockeys."
            
        response = "Voici les jockeys les plus performants (min. 20 courses) :\n"
        for r in results:
            response += f"- **{r['nom']}** : {r['taux_victoire']}% de victoires ({r['victoires']}/{r['courses']} courses)\n"
        return response

    def get_todays_races(self) -> str:
        """Retourne les courses du jour."""
        sql = """
            SELECT r.reunion_nom, r.nom, r.heure, r.hippodrome
            FROM courses r
            WHERE DATE(r.date) = CURRENT_DATE
            ORDER BY r.heure
        """
        results = self.query_db(sql)
        if not results:
            return "Il n'y a pas de courses enregistrées pour aujourd'hui dans ma base de données."
            
        response = "Voici les courses d'aujourd'hui :\n"
        for r in results:
            response += f"- {r['heure']} : **{r['nom']}** ({r['hippodrome']}) - {r['reunion_nom']}\n"
        return response

    def process_message(self, message: str, history: List[Dict[str, str]]) -> str:
        """Traite le message de l'utilisateur et retourne une réponse."""
        
        # Détection d'intention simple (fallback si pas d'LLM ou pour rapidité)
        message_lower = message.lower()
        
        if "chevaux" in message_lower and ("performant" in message_lower or "meilleur" in message_lower):
            return self.get_top_horses()
        
        if "jockey" in message_lower and ("performant" in message_lower or "meilleur" in message_lower):
            return self.get_top_jockeys()

        if "course" in message_lower and ("aujourd'hui" in message_lower or "jour" in message_lower):
            return self.get_todays_races()
            
        # Si OpenAI est configuré, utiliser l'LLM
        if self.client:
            try:
                # Préparation des messages
                messages = [{"role": "system", "content": self.get_system_prompt()}]
                # Ajouter l'historique (limité aux 5 derniers échanges pour économiser les tokens)
                for msg in history[-10:]: 
                    messages.append({"role": msg["role"], "content": msg["content"]})
                messages.append({"role": "user", "content": message})

                # Appel à l'API
                completion = self.client.chat.completions.create(
                    model="gpt-4-turbo-preview", # Ou gpt-3.5-turbo
                    messages=messages,
                    temperature=0.7,
                )
                return completion.choices[0].message.content
            except Exception as e:
                logger.error(f"OpenAI API error: {e}")
                return "Désolé, je rencontre des difficultés pour contacter mon cerveau numérique. Veuillez vérifier ma configuration."

        # Fallback générique
        return """Je suis un assistant IA spécialisé en courses hippiques. 
Pour l'instant, je peux vous donner des statistiques sur :
- Les meilleurs chevaux
- Les meilleurs jockeys

Posez-moi une question précise sur ces sujets !
(Note: Pour une intelligence complète, configurez une clé API OpenAI)."""
