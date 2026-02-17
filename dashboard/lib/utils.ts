/** Formatting helpers for the AMRCAIS dashboard. */

/** Format a number as a percentage string: 0.85 → "85.0%" */
export function pct(value: number | null | undefined, decimals = 1): string {
  if (value == null || Number.isNaN(value)) return "—";
  return `${(value * 100).toFixed(decimals)}%`;
}

/** Format a percentage that is already in % form: 14.2 → "14.2%" */
export function pctRaw(value: number | null | undefined, decimals = 1): string {
  if (value == null || Number.isNaN(value)) return "—";
  return `${value.toFixed(decimals)}%`;
}

/** Format a float with fixed decimals: 0.8532 → "0.85" */
export function num(value: number | null | undefined, decimals = 2): string {
  if (value == null || Number.isNaN(value)) return "—";
  return value.toFixed(decimals);
}

/** Format large numbers with commas: 100000 → "100,000" */
export function comma(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return "—";
  return value.toLocaleString("en-US", { maximumFractionDigits: 0 });
}

/** Format currency: 123456.78 → "$123,457" */
export function currency(
  value: number | null | undefined,
  decimals = 0,
): string {
  if (value == null || Number.isNaN(value)) return "—";
  return value.toLocaleString("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: decimals,
  });
}

/** Clamp a value between min and max. */
export function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

/** Conditional CSS class joiner — filters falsy values. */
export function cn(...classes: (string | false | undefined | null)[]): string {
  return classes.filter(Boolean).join(" ");
}
