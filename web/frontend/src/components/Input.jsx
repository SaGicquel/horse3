/**
 * Composants Input réutilisables avec effet Glassmorphism parfait
 * 
 * Caractéristiques :
 * - Effet glassmorphism avec backdrop-blur
 * - Focus : border primaire avec ring glass
 * - Label : couleur texte adaptative
 * - Radius : rounded-xl (16px)
 * - Contraste AA validé
 */

export const Input = ({ 
  label,
  type = 'text',
  value,
  onChange,
  placeholder = '',
  required = false,
  disabled = false,
  error = '',
  className = '' 
}) => {
  return (
    <div className="w-full">
      {label && (
        <label 
          className="block font-medium text-sm mb-2"
          style={{ color: 'var(--color-text)' }}
        >
          {label}
          {required && <span style={{ color: 'var(--color-error)' }} className="ml-1">*</span>}
        </label>
      )}
      <input
        type={type}
        value={value}
        onChange={onChange}
        placeholder={placeholder}
        required={required}
        disabled={disabled}
        className={`
          w-full rounded-xl py-3 px-4 
          text-sm transition-all duration-300
          focus:outline-none
          disabled:opacity-50 disabled:cursor-not-allowed
          glass-input
          ${className}
        `}
        style={{
          background: 'rgba(var(--color-card-rgb), 0.4)',
          backdropFilter: 'blur(12px) saturate(150%)',
          WebkitBackdropFilter: 'blur(12px) saturate(150%)',
          color: 'var(--color-text)',
          border: error ? '1px solid var(--color-error)' : '1px solid rgba(255, 255, 255, 0.15)',
          boxShadow: 'inset 0 2px 4px rgba(0, 0, 0, 0.06), inset 0 1px 0 rgba(255, 255, 255, 0.2)'
        }}
      />
      {error && (
        <p className="mt-1 text-sm" style={{ color: 'var(--color-error)' }}>{error}</p>
      )}
    </div>
  );
};

export const SearchInput = ({ 
  value,
  onChange,
  placeholder = 'Rechercher...',
  icon: Icon,
  className = '' 
}) => {
  return (
    <div className="relative">
      {Icon && (
        <Icon 
          className="pointer-events-none absolute left-4 top-1/2 -translate-y-1/2" 
          size={18} 
          style={{ color: 'var(--color-muted)' }}
        />
      )}
      <input
        type="text"
        value={value}
        onChange={onChange}
        placeholder={placeholder}
        className={`
          w-full rounded-xl 
          py-3 ${Icon ? 'pl-12' : 'pl-4'} pr-4 
          text-sm
          transition-all duration-300
          focus:outline-none
          ${className}
        `}
        style={{
          background: 'rgba(var(--color-card-rgb), 0.5)',
          backdropFilter: 'blur(16px) saturate(150%)',
          WebkitBackdropFilter: 'blur(16px) saturate(150%)',
          color: 'var(--color-text)',
          border: '1px solid rgba(255, 255, 255, 0.15)',
          boxShadow: 'inset 0 2px 4px rgba(0, 0, 0, 0.06), inset 0 1px 0 rgba(255, 255, 255, 0.2)'
        }}
      />
    </div>
  );
};

export const TextArea = ({ 
  label,
  value,
  onChange,
  placeholder = '',
  required = false,
  disabled = false,
  rows = 4,
  error = '',
  className = '' 
}) => {
  return (
    <div className="w-full">
      {label && (
        <label 
          className="block font-medium text-sm mb-2"
          style={{ color: 'var(--color-text)' }}
        >
          {label}
          {required && <span style={{ color: 'var(--color-error)' }} className="ml-1">*</span>}
        </label>
      )}
      <textarea
        value={value}
        onChange={onChange}
        placeholder={placeholder}
        required={required}
        disabled={disabled}
        rows={rows}
        className={`
          w-full rounded-xl 
          py-3 px-4 
          text-sm
          transition-all duration-300
          focus:outline-none
          disabled:opacity-50 disabled:cursor-not-allowed
          resize-vertical
          glass-input
          ${className}
        `}
        style={{
          background: 'rgba(var(--color-card-rgb), 0.4)',
          backdropFilter: 'blur(12px) saturate(150%)',
          WebkitBackdropFilter: 'blur(12px) saturate(150%)',
          color: 'var(--color-text)',
          border: error ? '1px solid var(--color-error)' : '1px solid rgba(255, 255, 255, 0.15)',
          boxShadow: 'inset 0 2px 4px rgba(0, 0, 0, 0.06), inset 0 1px 0 rgba(255, 255, 255, 0.2)'
        }}
      />
      {error && (
        <p className="mt-1 text-sm" style={{ color: 'var(--color-error)' }}>{error}</p>
      )}
    </div>
  );
};

/**
 * Exemple d'utilisation :
 * 
 * import { Input, SearchInput, TextArea } from './components/Input';
 * import { Search } from 'lucide-react';
 * 
 * <Input
 *   label="Nom"
 *   value={name}
 *   onChange={(e) => setName(e.target.value)}
 *   required
 * />
 * 
 * <SearchInput
 *   icon={Search}
 *   value={search}
 *   onChange={(e) => setSearch(e.target.value)}
 *   placeholder="Rechercher un cheval..."
 * />
 * 
 * <TextArea
 *   label="Description"
 *   value={description}
 *   onChange={(e) => setDescription(e.target.value)}
 *   rows={6}
 * />
 */
