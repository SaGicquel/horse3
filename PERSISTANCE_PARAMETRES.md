# 🎯 Système de Persistance des Paramètres Utilisateur

## ✅ **Problème résolu**

**Avant :** Les paramètres utilisateur (bankroll, profil) se réinitialisaient à chaque rechargement de page.

**Après :** Persistance automatique avec synchronisation entre toutes les pages de l'application.

---

## 🔧 **Implémentation**

### **1. Hook personnalisé `useUserSettings`**
- **Sauvegarde automatique** dans `localStorage`
- **Synchronisation** entre pages avec événements personnalisés
- **Réactivité** aux changements depuis d'autres onglets

### **2. Integration dans les pages**
- **Page Paris** : Utilise les paramètres pour personnaliser les recommandations
- **Page Settings** : Section dédiée à la configuration utilisateur
- **Notifications** : Confirmation visuelle lors des modifications

### **3. API Backend adaptée**
- Paramètres transmis automatiquement : `?bankroll=X&profil=Y`
- Filtrage intelligent selon le profil
- Calcul des mises personnalisées

---

## 🚀 **Fonctionnalités**

### **Persistance**
```javascript
// Sauvegarde automatique
setBankroll(1000); // → localStorage + événement
setProfil('AGRESSIF'); // → localStorage + événement

// Chargement automatique
const { bankroll, profil } = useUserSettings();
```

### **Synchronisation inter-pages**
- Modifier dans **Settings** → Visible dans **Paris** immédiatement
- Événements `userSettingsChanged` pour coordination
- Support multi-onglets avec `localStorage`

### **Notifications visuelles**
- Confirmation des modifications
- Animation élégante (top-right)
- Masquage automatique après 3s

---

## 🎨 **Expérience utilisateur**

1. **Configuration dans Settings** :
   - Slider bankroll (100-10000€)
   - 3 profils : Prudent 🛡️ / Standard ⚖️ / Agressif 🚀
   - Résumé configuration actuelle

2. **Application automatique** :
   - Paris personnalisés selon profil
   - Budget journalier respecté
   - Filtres de value/cotes adaptés

3. **Feedback immédiat** :
   - Notification de sauvegarde
   - Mise à jour temps réel des recommandations
   - Persistance entre sessions

---

## 📊 **Exemple concret**

**Utilisateur configure :**
- Bankroll : 1500€
- Profil : Agressif 🚀

**Résultat automatique :**
- Budget journalier : 300€/jour
- Max par pari : 60€
- Kelly multiplier : x1.0
- Cotes jusqu'à 50.0
- Values dès 1%

**Navigation :**
- Settings → Paris → Refresh → **Mêmes paramètres !** ✅

---

## 🔗 **Test**

```bash
# 1. Ouvrir http://localhost/settings
# 2. Modifier bankroll + profil
# 3. Aller sur http://localhost/conseils  
# 4. → Paramètres conservés ! 🎉
```

Le système garantit une **expérience fluide** où l'utilisateur configure une fois et retrouve ses préférences partout dans l'application !