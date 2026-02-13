import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import ErrorState from "@/components/ui/ErrorState";
import RegimeBadge from "@/components/ui/RegimeBadge";
import MetricsCard from "@/components/ui/MetricsCard";

describe("ErrorState", () => {
  it("renders default error message", () => {
    render(<ErrorState />);
    expect(screen.getByText("Failed to load data")).toBeInTheDocument();
  });

  it("renders custom error message", () => {
    render(<ErrorState message="Network timeout" />);
    expect(screen.getByText("Network timeout")).toBeInTheDocument();
  });

  it("renders retry button when onRetry is provided", () => {
    render(<ErrorState onRetry={() => {}} />);
    expect(screen.getByText("Retry")).toBeInTheDocument();
  });

  it("does not render retry button when onRetry is absent", () => {
    render(<ErrorState />);
    expect(screen.queryByText("Retry")).not.toBeInTheDocument();
  });

  it("calls onRetry when retry button is clicked", async () => {
    const user = userEvent.setup();
    const onRetry = vi.fn();
    render(<ErrorState onRetry={onRetry} />);

    await user.click(screen.getByText("Retry"));
    expect(onRetry).toHaveBeenCalledOnce();
  });
});

describe("RegimeBadge", () => {
  it("renders regime name for regime 1", () => {
    render(<RegimeBadge regime={1} />);
    expect(screen.getByText("Risk-On Growth")).toBeInTheDocument();
  });

  it("renders regime name for regime 2", () => {
    render(<RegimeBadge regime={2} />);
    expect(screen.getByText("Risk-Off Crisis")).toBeInTheDocument();
  });

  it("renders regime name for regime 3", () => {
    render(<RegimeBadge regime={3} />);
    expect(screen.getByText("Stagflation")).toBeInTheDocument();
  });

  it("renders regime name for regime 4", () => {
    render(<RegimeBadge regime={4} />);
    expect(screen.getByText("Disinflationary Boom")).toBeInTheDocument();
  });

  it("shows confidence when provided", () => {
    render(<RegimeBadge regime={1} confidence={0.85} />);
    expect(screen.getByText("85%")).toBeInTheDocument();
  });

  it("does not show confidence when not provided", () => {
    render(<RegimeBadge regime={1} />);
    expect(screen.queryByText(/%$/)).not.toBeInTheDocument();
  });
});

describe("MetricsCard", () => {
  it("renders label and value", () => {
    render(<MetricsCard label="Sharpe Ratio" value="1.45" />);
    expect(screen.getByText("Sharpe Ratio")).toBeInTheDocument();
    expect(screen.getByText("1.45")).toBeInTheDocument();
  });

  it("renders subtitle when provided", () => {
    render(
      <MetricsCard label="Total Return" value="24.5%" sub="Since inception" />,
    );
    expect(screen.getByText("Since inception")).toBeInTheDocument();
  });

  it("does not render subtitle when absent", () => {
    render(<MetricsCard label="VIX" value="18.2" />);
    // Only label + value, no sub
    expect(screen.queryByText("Since inception")).not.toBeInTheDocument();
  });

  it("handles numeric values", () => {
    render(<MetricsCard label="Count" value={42} />);
    expect(screen.getByText("42")).toBeInTheDocument();
  });
});
