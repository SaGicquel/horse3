# 🏇 Docker Stack - Horse Race Predictor

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Port 80                               │
│                     ┌──────────┐                            │
│                     │ Frontend │                            │
│                     │  Nginx   │                            │
│                     └────┬─────┘                            │
│                          │                                  │
│            ┌─────────────┴─────────────┐                    │
│            │ /api/*                    │ /*                 │
│            ▼                           ▼                    │
│     ┌──────────────┐           ┌─────────────┐             │
│     │   Backend    │           │    React    │             │
│     │   FastAPI    │           │   Static    │             │
│     │  Port 8000   │           │   Files     │             │
│     └──────┬───────┘           └─────────────┘             │
│            │                                                │
│            ▼                                                │
│     ┌──────────────┐                                       │
│     │  PostgreSQL  │                                       │
│     │  Port 5432   │                                       │
│     └──────────────┘                                       │
└─────────────────────────────────────────────────────────────┘
```

## Démarrage rapide

### 1. Configuration
```bash
cd /Users/gicquelsacha/horse3/web

# Copier le fichier d'environnement
cp .env.example .env

# Éditer les variables (optionnel)
nano .env
```

### 2. Lancement
```bash
# Build et démarrage de tous les services
docker-compose up --build -d

# Voir les logs
docker-compose logs -f
```

### 3. Accès
- **Frontend**: http://localhost (port 80)
- **Backend API**: http://localhost:8000
- **PostgreSQL**: localhost:5432

## Commandes utiles

```bash
# Voir les logs d'un service
docker-compose logs -f backend
docker-compose logs -f frontend
docker-compose logs -f db

# Redémarrer un service
docker-compose restart backend

# Arrêter tout
docker-compose down

# Arrêter et supprimer les volumes (reset DB)
docker-compose down -v

# Rebuild un service spécifique
docker-compose build --no-cache backend
docker-compose up -d backend
```

## Développement

Pour le développement local sans Docker :

```bash
# Terminal 1 - Backend
cd backend
source venv/bin/activate
uvicorn main:app --reload --port 8000

# Terminal 2 - Frontend
cd frontend
npm run dev
```

## Variables d'environnement

| Variable | Description | Default |
|----------|-------------|---------|
| `POSTGRES_USER` | Utilisateur PostgreSQL | horse |
| `POSTGRES_PASSWORD` | Mot de passe PostgreSQL | horse_password |
| `POSTGRES_DB` | Nom de la base | horserace |
| `OPENAI_API_KEY` | Clé API OpenAI (optionnel) | - |

## Résolution de problèmes

### Le backend ne démarre pas
```bash
# Vérifier les logs
docker-compose logs backend

# Vérifier la connexion à la DB
docker-compose exec backend python -c "from db_connection import get_connection; print('OK')"
```

### Le frontend affiche une erreur CORS
Vérifiez que le backend autorise bien le CORS depuis le frontend.

### La base de données est vide
Importez vos données existantes :
```bash
# Depuis un dump SQL
docker-compose exec -T db psql -U horse horserace < backup.sql
```
