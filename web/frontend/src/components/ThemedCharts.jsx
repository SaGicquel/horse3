/**
 * Exemple de composant de graphique compatible avec le mode sombre
 *
 * Utilise le hook useThemeColors pour adapter automatiquement
 * les couleurs des graphiques Recharts au thème actif.
 *
 * Tous les éléments (axes, grille, tooltips, légendes) sont adaptés
 * pour maintenir un contraste AA en mode clair et sombre.
 */

import { LineChart, Line, AreaChart, Area, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { useThemeColors } from '../hooks/useThemeColors';

/**
 * Tooltip personnalisé avec support du mode sombre
 */
const CustomTooltip = ({ active, payload, label, colors }) => {
  if (active && payload && payload.length) {
    return (
      <div
        className="p-3 rounded-lg shadow-lg"
        style={{
          backgroundColor: colors.card,
          border: `1px solid ${colors.border}`,
        }}
      >
        <p className="font-semibold mb-2" style={{ color: colors.text }}>
          {label}
        </p>
        {payload.map((entry, index) => (
          <p key={index} className="text-sm" style={{ color: entry.color }}>
            {entry.name}: {entry.value}
          </p>
        ))}
      </div>
    );
  }
  return null;
};

/**
 * Graphique en ligne adaptatif
 */
export const ThemedLineChart = ({ data, dataKeys = ['value'], height = 300 }) => {
  const colors = useThemeColors();

  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
        <CartesianGrid
          strokeDasharray="3 3"
          stroke={colors.border}
          opacity={0.3}
        />
        <XAxis
          dataKey="name"
          stroke={colors.muted}
          tick={{ fill: colors.text, fontSize: 12 }}
          tickLine={{ stroke: colors.muted }}
        />
        <YAxis
          stroke={colors.muted}
          tick={{ fill: colors.text, fontSize: 12 }}
          tickLine={{ stroke: colors.muted }}
        />
        <Tooltip
          content={<CustomTooltip colors={colors} />}
          cursor={{ stroke: colors.primary, strokeWidth: 1 }}
        />
        <Legend
          wrapperStyle={{ color: colors.text }}
          iconType="line"
        />
        {dataKeys.map((key, index) => (
          <Line
            key={key}
            type="monotone"
            dataKey={key}
            stroke={index === 0 ? colors.primary : colors.primaryLight}
            strokeWidth={2}
            dot={{ fill: index === 0 ? colors.primary : colors.primaryLight, r: 4 }}
            activeDot={{ r: 6, stroke: colors.card, strokeWidth: 2 }}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
};

/**
 * Graphique en aires adaptatif
 */
export const ThemedAreaChart = ({ data, dataKeys = ['value'], height = 300 }) => {
  const colors = useThemeColors();

  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={data} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
        <defs>
          <linearGradient id="colorPrimary" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor={colors.primary} stopOpacity={0.3} />
            <stop offset="95%" stopColor={colors.primary} stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid
          strokeDasharray="3 3"
          stroke={colors.border}
          opacity={0.3}
        />
        <XAxis
          dataKey="name"
          stroke={colors.muted}
          tick={{ fill: colors.text, fontSize: 12 }}
          tickLine={{ stroke: colors.muted }}
        />
        <YAxis
          stroke={colors.muted}
          tick={{ fill: colors.text, fontSize: 12 }}
          tickLine={{ stroke: colors.muted }}
        />
        <Tooltip
          content={<CustomTooltip colors={colors} />}
          cursor={{ stroke: colors.primary, strokeWidth: 1 }}
        />
        <Legend
          wrapperStyle={{ color: colors.text }}
          iconType="rect"
        />
        {dataKeys.map((key, index) => (
          <Area
            key={key}
            type="monotone"
            dataKey={key}
            stroke={index === 0 ? colors.primary : colors.primaryLight}
            strokeWidth={2}
            fill={index === 0 ? "url(#colorPrimary)" : colors.primaryLight}
            fillOpacity={0.2}
          />
        ))}
      </AreaChart>
    </ResponsiveContainer>
  );
};

/**
 * Graphique en barres adaptatif
 */
export const ThemedBarChart = ({ data, dataKeys = ['value'], height = 300 }) => {
  const colors = useThemeColors();

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
        <CartesianGrid
          strokeDasharray="3 3"
          stroke={colors.border}
          opacity={0.3}
        />
        <XAxis
          dataKey="name"
          stroke={colors.muted}
          tick={{ fill: colors.text, fontSize: 12 }}
          tickLine={{ stroke: colors.muted }}
        />
        <YAxis
          stroke={colors.muted}
          tick={{ fill: colors.text, fontSize: 12 }}
          tickLine={{ stroke: colors.muted }}
        />
        <Tooltip
          content={<CustomTooltip colors={colors} />}
          cursor={{ fill: colors.border, opacity: 0.3 }}
        />
        <Legend
          wrapperStyle={{ color: colors.text }}
          iconType="rect"
        />
        {dataKeys.map((key, index) => (
          <Bar
            key={key}
            dataKey={key}
            fill={index === 0 ? colors.primary : colors.primaryLight}
            radius={[4, 4, 0, 0]}
          />
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
};

/**
 * Exemple d'utilisation :
 *
 * import { ThemedLineChart, ThemedAreaChart, ThemedBarChart } from './components/ThemedCharts';
 *
 * const data = [
 *   { name: 'Jan', value: 400, value2: 240 },
 *   { name: 'Fev', value: 300, value2: 139 },
 *   { name: 'Mar', value: 600, value2: 980 },
 * ];
 *
 * <ThemedLineChart data={data} dataKeys={['value', 'value2']} height={400} />
 * <ThemedAreaChart data={data} dataKeys={['value']} />
 * <ThemedBarChart data={data} dataKeys={['value', 'value2']} />
 */
