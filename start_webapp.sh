#!/bin/bash
echo "🚀 Démarrage de l'application Web Horse AI..."

# Démarrage Backend
echo "Backend (FastAPI) sur http://localhost:8000..."
cd web/backend || exit
# Utiliser le venv global s'il existe, sinon celui du backend
if [ -d "../../.venv" ]; then
    source ../../.venv/bin/activate
    uvicorn main:app --reload --port 8000 > ../../backend.log 2>&1 &
else
    source venv/bin/activate
    uvicorn main:app --reload --port 8000 > ../../backend.log 2>&1 &
fi
BACKEND_PID=$!
cd ../.. || exit

# Démarrage Frontend
echo "Frontend (React) sur http://localhost:5173..."
cd web/frontend || exit
npm run dev > ../../frontend.log 2>&1 &
FRONTEND_PID=$!
cd ../.. || exit

echo "✅ Application lancée !"
echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo "Accédez à : http://localhost:5173"
echo "Logs dans backend.log et frontend.log"

# Trap pour tuer les processus à la sortie
trap 'kill $BACKEND_PID $FRONTEND_PID' EXIT

wait
