import { describe, it, expect, vi, beforeEach } from "vitest";

// We need to override the mock for specific searchParams in some tests
const mockReplace = vi.fn();

vi.mock("next/navigation", () => ({
  useRouter: () => ({
    replace: mockReplace,
    push: vi.fn(),
    back: vi.fn(),
    forward: vi.fn(),
    refresh: vi.fn(),
    prefetch: vi.fn(),
  }),
  useSearchParams: () => new URLSearchParams(),
  usePathname: () => "/test",
}));

import { renderHook, act } from "@testing-library/react";
import { useQueryState, useQueryNumber } from "@/lib/hooks";

describe("useQueryState", () => {
  beforeEach(() => {
    mockReplace.mockClear();
  });

  it("returns the default value when no query param is set", () => {
    const { result } = renderHook(() => useQueryState("tab", "macro"));
    expect(result.current[0]).toBe("macro");
  });

  it("provides a setter function", () => {
    const { result } = renderHook(() => useQueryState("tab", "macro"));
    expect(typeof result.current[1]).toBe("function");
  });

  it("calls router.replace with new param value", () => {
    const { result } = renderHook(() => useQueryState("tab", "macro"));

    act(() => {
      result.current[1]("yield_curve");
    });

    expect(mockReplace).toHaveBeenCalledWith("/test?tab=yield_curve", {
      scroll: false,
    });
  });

  it("deletes param when set to default value", () => {
    const { result } = renderHook(() => useQueryState("tab", "macro"));

    act(() => {
      result.current[1]("macro");
    });

    // Should replace without query string (param deleted)
    expect(mockReplace).toHaveBeenCalledWith("/test", { scroll: false });
  });
});

describe("useQueryNumber", () => {
  beforeEach(() => {
    mockReplace.mockClear();
  });

  it("returns the default numeric value", () => {
    const { result } = renderHook(() => useQueryNumber("window", 60));
    expect(result.current[0]).toBe(60);
  });

  it("calls router.replace with numeric param", () => {
    const { result } = renderHook(() => useQueryNumber("window", 60));

    act(() => {
      result.current[1](90);
    });

    expect(mockReplace).toHaveBeenCalledWith("/test?window=90", {
      scroll: false,
    });
  });

  it("deletes param when set to default", () => {
    const { result } = renderHook(() => useQueryNumber("window", 60));

    act(() => {
      result.current[1](60);
    });

    expect(mockReplace).toHaveBeenCalledWith("/test", { scroll: false });
  });
});
