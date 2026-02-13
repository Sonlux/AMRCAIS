import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach, vi } from "vitest";

// Automatic cleanup after each test
afterEach(() => {
  cleanup();
});

// Mock next/navigation (used by useQueryState / useQueryNumber)
vi.mock("next/navigation", () => ({
  useRouter: () => ({
    replace: vi.fn(),
    push: vi.fn(),
    back: vi.fn(),
    forward: vi.fn(),
    refresh: vi.fn(),
    prefetch: vi.fn(),
  }),
  useSearchParams: () => new URLSearchParams(),
  usePathname: () => "/",
}));

// Mock next/dynamic — render children directly (skip SSR guard)
vi.mock("next/dynamic", () => ({
  __esModule: true,
  default: (loader: () => Promise<{ default: React.ComponentType }>) => {
    // Return a component that lazily loads the real one
    // For tests we just render a placeholder
    const Component = (props: Record<string, unknown>) => {
      return null; // Plotly / lightweight-charts don't render in jsdom
    };
    Component.displayName = "DynamicMock";
    return Component;
  },
}));

// Stub ResizeObserver (not available in jsdom)
class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}
globalThis.ResizeObserver =
  ResizeObserverStub as unknown as typeof ResizeObserver;
