import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import SignalCard from "@/components/ui/SignalCard";

// We need to read the component to understand its props
// Based on usage: module, signal, strength, explanation

describe("SignalCard", () => {
  it("renders module name and signal", () => {
    render(
      <SignalCard
        module="Macro Events"
        signal="bullish"
        strength={0.8}
        explanation="Strong growth indicators"
      />,
    );

    expect(screen.getByText("Macro Events")).toBeInTheDocument();
    expect(screen.getByText("bullish")).toBeInTheDocument();
  });

  it("renders explanation text", () => {
    render(
      <SignalCard
        module="Yield Curve"
        signal="bearish"
        strength={0.65}
        explanation="Curve inversion detected"
      />,
    );

    expect(screen.getByText("Curve inversion detected")).toBeInTheDocument();
  });

  it("renders neutral signal correctly", () => {
    render(
      <SignalCard
        module="Options"
        signal="neutral"
        strength={0.3}
        explanation="Insufficient data"
      />,
    );

    expect(screen.getByText("neutral")).toBeInTheDocument();
    expect(screen.getByText("Options")).toBeInTheDocument();
  });
});
