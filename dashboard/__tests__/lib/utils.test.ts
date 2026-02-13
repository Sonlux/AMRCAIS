import { describe, it, expect } from "vitest";
import { pct, pctRaw, num, comma, currency, clamp, cn } from "@/lib/utils";

describe("pct", () => {
  it("formats 0.85 as 85.0%", () => {
    expect(pct(0.85)).toBe("85.0%");
  });

  it("formats 0 as 0.0%", () => {
    expect(pct(0)).toBe("0.0%");
  });

  it("formats 1 as 100.0%", () => {
    expect(pct(1)).toBe("100.0%");
  });

  it("respects custom decimals", () => {
    expect(pct(0.8532, 2)).toBe("85.32%");
  });

  it("handles negative values", () => {
    expect(pct(-0.15)).toBe("-15.0%");
  });
});

describe("pctRaw", () => {
  it("formats 14.2 as 14.2%", () => {
    expect(pctRaw(14.2)).toBe("14.2%");
  });

  it("respects decimals", () => {
    expect(pctRaw(14.256, 2)).toBe("14.26%");
  });
});

describe("num", () => {
  it("formats with 2 decimal places by default", () => {
    expect(num(0.8532)).toBe("0.85");
  });

  it("formats whole numbers", () => {
    expect(num(3)).toBe("3.00");
  });
});

describe("comma", () => {
  it("adds commas to large numbers", () => {
    expect(comma(100000)).toBe("100,000");
  });

  it("handles small numbers", () => {
    expect(comma(42)).toBe("42");
  });
});

describe("currency", () => {
  it("formats as USD", () => {
    expect(currency(123456.78)).toBe("$123,457");
  });

  it("respects decimals", () => {
    expect(currency(123.456, 2)).toBe("$123.46");
  });
});

describe("clamp", () => {
  it("clamps below min", () => {
    expect(clamp(-5, 0, 100)).toBe(0);
  });

  it("clamps above max", () => {
    expect(clamp(150, 0, 100)).toBe(100);
  });

  it("keeps value in range", () => {
    expect(clamp(50, 0, 100)).toBe(50);
  });

  it("handles edge cases at boundaries", () => {
    expect(clamp(0, 0, 100)).toBe(0);
    expect(clamp(100, 0, 100)).toBe(100);
  });
});

describe("cn", () => {
  it("joins class strings", () => {
    expect(cn("a", "b", "c")).toBe("a b c");
  });

  it("filters falsy values", () => {
    expect(cn("a", false, undefined, null, "b")).toBe("a b");
  });

  it("returns empty string for all falsy", () => {
    expect(cn(false, undefined, null)).toBe("");
  });
});
