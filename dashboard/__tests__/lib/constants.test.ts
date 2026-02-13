import { describe, it, expect } from "vitest";
import {
  REGIME_NAMES,
  REGIME_COLORS,
  REGIME_BG_COLORS,
  SIGNAL_COLORS,
  TRACKED_ASSETS,
  MODULE_NAMES,
} from "@/lib/constants";

describe("REGIME_NAMES", () => {
  it("has all 4 regimes defined", () => {
    expect(Object.keys(REGIME_NAMES)).toHaveLength(4);
    expect(REGIME_NAMES[1]).toBe("Risk-On Growth");
    expect(REGIME_NAMES[2]).toBe("Risk-Off Crisis");
    expect(REGIME_NAMES[3]).toBe("Stagflation");
    expect(REGIME_NAMES[4]).toBe("Disinflationary Boom");
  });
});

describe("REGIME_COLORS", () => {
  it("has a color for each regime", () => {
    for (const id of [1, 2, 3, 4]) {
      expect(REGIME_COLORS[id]).toBeDefined();
      expect(REGIME_COLORS[id]).toMatch(/^#[0-9a-f]{6}$/i);
    }
  });
});

describe("REGIME_BG_COLORS", () => {
  it("has a background color for each regime", () => {
    for (const id of [1, 2, 3, 4]) {
      expect(REGIME_BG_COLORS[id]).toBeDefined();
      expect(REGIME_BG_COLORS[id]).toMatch(/^rgba/);
    }
  });
});

describe("SIGNAL_COLORS", () => {
  it("defines colors for all signal types", () => {
    expect(SIGNAL_COLORS.bullish).toBeDefined();
    expect(SIGNAL_COLORS.bearish).toBeDefined();
    expect(SIGNAL_COLORS.neutral).toBeDefined();
    expect(SIGNAL_COLORS.cautious).toBeDefined();
  });
});

describe("TRACKED_ASSETS", () => {
  it("includes all 6 tracked assets", () => {
    expect(TRACKED_ASSETS).toHaveLength(6);
    expect(TRACKED_ASSETS).toContain("SPX");
    expect(TRACKED_ASSETS).toContain("TLT");
    expect(TRACKED_ASSETS).toContain("GLD");
    expect(TRACKED_ASSETS).toContain("DXY");
    expect(TRACKED_ASSETS).toContain("WTI");
    expect(TRACKED_ASSETS).toContain("VIX");
  });
});

describe("MODULE_NAMES", () => {
  it("maps all 5 module keys", () => {
    expect(Object.keys(MODULE_NAMES)).toHaveLength(5);
    expect(MODULE_NAMES.macro).toBe("Macro Events");
    expect(MODULE_NAMES.yield_curve).toBe("Yield Curve");
    expect(MODULE_NAMES.options).toBe("Options Surface");
    expect(MODULE_NAMES.factors).toBe("Factor Exposure");
    expect(MODULE_NAMES.correlations).toBe("Correlations");
  });
});
