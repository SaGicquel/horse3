import React from 'react';

/**
 * Composant de démonstration de la palette de couleurs
 * Ce composant affiche toutes les couleurs disponibles dans le système de design
 */
const ColorPalette = () => {
  return (
    <div className="min-h-screen bg-neutral-900 p-8">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-4xl font-bold text-neutral-50 mb-8">
          Palette de Couleurs
        </h1>

        {/* Couleur Primaire */}
        <section className="mb-12">
          <h2 className="text-2xl font-semibold text-neutral-50 mb-4">
            Couleur Primaire (p)
          </h2>
          <div className="grid grid-cols-3 gap-4">
            <div className="card-dark p-6">
              <div className="w-full h-24 bg-p-300 rounded-lg mb-3"></div>
              <p className="text-neutral-50 font-mono">p-300</p>
              <p className="text-neutral-400 text-sm">#C86D8A</p>
            </div>
            <div className="card-dark p-6">
              <div className="w-full h-24 bg-p-500 rounded-lg mb-3"></div>
              <p className="text-neutral-50 font-mono">p-500</p>
              <p className="text-neutral-400 text-sm">#9D3656</p>
            </div>
            <div className="card-dark p-6">
              <div className="w-full h-24 bg-p-600 rounded-lg mb-3"></div>
              <p className="text-neutral-50 font-mono">p-600</p>
              <p className="text-neutral-400 text-sm">#812C47</p>
            </div>
          </div>
        </section>

        {/* Couleur Secondaire */}
        <section className="mb-12">
          <h2 className="text-2xl font-semibold text-neutral-50 mb-4">
            Couleur Secondaire (s)
          </h2>
          <div className="grid grid-cols-3 gap-4">
            <div className="card-dark p-6">
              <div className="w-full h-24 bg-s-500 rounded-lg mb-3"></div>
              <p className="text-neutral-50 font-mono">s-500</p>
              <p className="text-neutral-400 text-sm">#F5C3CE</p>
            </div>
          </div>
        </section>

        {/* Statuts */}
        <section className="mb-12">
          <h2 className="text-2xl font-semibold text-neutral-50 mb-4">
            Couleurs de Statut
          </h2>
          <div className="grid grid-cols-3 gap-4">
            <div className="card-dark p-6">
              <div className="w-full h-24 bg-success rounded-lg mb-3"></div>
              <p className="text-neutral-50 font-mono">success</p>
              <p className="text-neutral-400 text-sm">#2ED573</p>
            </div>
            <div className="card-dark p-6">
              <div className="w-full h-24 bg-error rounded-lg mb-3"></div>
              <p className="text-neutral-50 font-mono">error</p>
              <p className="text-neutral-400 text-sm">#DC2626</p>
            </div>
          </div>
        </section>

        {/* Neutral */}
        <section className="mb-12">
          <h2 className="text-2xl font-semibold text-neutral-50 mb-4">
            Échelle Neutral
          </h2>
          <div className="grid grid-cols-5 gap-4">
            {[50, 100, 200, 300, 400, 500, 600, 700, 800, 900].map((shade) => (
              <div key={shade} className="card-dark p-4">
                <div
                  className={`w-full h-16 bg-neutral-${shade} rounded-lg mb-2`}
                  style={{
                    backgroundColor:
                      shade === 50
                        ? '#F9FAFB'
                        : shade === 100
                        ? '#F3F4F6'
                        : shade === 200
                        ? '#E5E7EB'
                        : shade === 300
                        ? '#D1D5DB'
                        : shade === 400
                        ? '#9CA3AF'
                        : shade === 500
                        ? '#6B7280'
                        : shade === 600
                        ? '#374151'
                        : shade === 700
                        ? '#1F2937'
                        : shade === 800
                        ? '#111827'
                        : '#0B0F1A',
                  }}
                ></div>
                <p className="text-neutral-50 font-mono text-sm">
                  neutral-{shade}
                </p>
              </div>
            ))}
          </div>
        </section>

        {/* Exemples de Boutons */}
        <section className="mb-12">
          <h2 className="text-2xl font-semibold text-neutral-50 mb-4">
            Exemples de Boutons
          </h2>
          <div className="flex flex-wrap gap-4">
            <button className="bg-p-500 hover:bg-p-600 text-white px-6 py-3 rounded-lg shadow-p-glow hover:shadow-p-glow-strong transition-all">
              Bouton Primaire
            </button>
            <button className="bg-p-300 hover:bg-p-500 text-white px-6 py-3 rounded-lg transition-all">
              Bouton Secondaire
            </button>
            <button className="border-2 border-p-500 text-p-500 hover:bg-p-500 hover:text-white px-6 py-3 rounded-lg transition-all">
              Bouton Outline
            </button>
            <button className="bg-success hover:bg-success/80 text-white px-6 py-3 rounded-lg transition-all">
              Succès
            </button>
            <button className="bg-error hover:bg-error/80 text-white px-6 py-3 rounded-lg transition-all">
              Erreur
            </button>
          </div>
        </section>

        {/* Exemples de Cartes */}
        <section className="mb-12">
          <h2 className="text-2xl font-semibold text-neutral-50 mb-4">
            Exemples de Cartes
          </h2>
          <div className="grid grid-cols-2 gap-6">
            <div className="card-dark p-6">
              <h3 className="text-neutral-50 text-xl font-semibold mb-2">
                Card Dark
              </h3>
              <p className="text-neutral-300 mb-4">
                Carte avec fond sombre et bordure primaire subtile.
              </p>
              <div className="divider-soft my-4"></div>
              <button className="bg-p-500 text-white px-4 py-2 rounded-lg">
                Action
              </button>
            </div>
            <div className="card-dark-solid p-6">
              <h3 className="text-neutral-50 text-xl font-semibold mb-2">
                Card Dark Solid
              </h3>
              <p className="text-neutral-300 mb-4">
                Carte avec fond plus solide et bordure primaire accentuée.
              </p>
              <div className="divider-soft my-4"></div>
              <button className="bg-p-500 text-white px-4 py-2 rounded-lg">
                Action
              </button>
            </div>
          </div>
        </section>

        {/* Badges */}
        <section className="mb-12">
          <h2 className="text-2xl font-semibold text-neutral-50 mb-4">
            Badges et Tags
          </h2>
          <div className="flex flex-wrap gap-3">
            <span className="bg-p-500 text-white px-4 py-2 rounded-full text-sm font-medium">
              Badge Primaire
            </span>
            <span className="bg-s-500 text-p-600 px-4 py-2 rounded-full text-sm font-medium">
              Badge Secondaire
            </span>
            <span className="bg-success/10 text-success px-4 py-2 rounded-full text-sm font-medium">
              ✓ Succès
            </span>
            <span className="bg-error/10 text-error px-4 py-2 rounded-full text-sm font-medium">
              ✗ Erreur
            </span>
            <span className="bg-neutral-700 text-neutral-300 px-4 py-2 rounded-full text-sm font-medium">
              Badge Neutral
            </span>
          </div>
        </section>

        {/* Test de Contraste */}
        <section className="mb-12">
          <h2 className="text-2xl font-semibold text-neutral-50 mb-4">
            Test de Contraste (AA ✅)
          </h2>
          <div className="grid grid-cols-3 gap-4">
            <div className="bg-p-500 p-6 rounded-lg">
              <p className="text-white font-semibold mb-2">
                Texte Blanc sur p-500
              </p>
              <p className="text-white text-sm">
                Ratio: 5.3:1 - Conformité AA ✅
              </p>
            </div>
            <div className="bg-p-600 p-6 rounded-lg">
              <p className="text-white font-semibold mb-2">
                Texte Blanc sur p-600
              </p>
              <p className="text-white text-sm">
                Excellent contraste ✅
              </p>
            </div>
            <div className="bg-s-500 p-6 rounded-lg">
              <p className="text-p-600 font-semibold mb-2">
                Texte p-600 sur s-500
              </p>
              <p className="text-p-600 text-sm">
                Très bon contraste ✅
              </p>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
};

export default ColorPalette;
