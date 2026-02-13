import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { type ColumnDef } from "@tanstack/react-table";
import DataTable from "@/components/ui/DataTable";

interface TestRow {
  id: number;
  name: string;
  value: number;
}

const columns: ColumnDef<TestRow, unknown>[] = [
  { accessorKey: "id", header: "ID" },
  { accessorKey: "name", header: "Name" },
  { accessorKey: "value", header: "Value" },
];

const data: TestRow[] = [
  { id: 1, name: "Alpha", value: 10 },
  { id: 2, name: "Beta", value: 20 },
  { id: 3, name: "Gamma", value: 30 },
];

describe("DataTable", () => {
  it("renders all column headers", () => {
    render(<DataTable columns={columns} data={data} />);
    expect(screen.getByText("ID")).toBeInTheDocument();
    expect(screen.getByText("Name")).toBeInTheDocument();
    expect(screen.getByText("Value")).toBeInTheDocument();
  });

  it("renders all rows", () => {
    render(<DataTable columns={columns} data={data} />);
    expect(screen.getByText("Alpha")).toBeInTheDocument();
    expect(screen.getByText("Beta")).toBeInTheDocument();
    expect(screen.getByText("Gamma")).toBeInTheDocument();
  });

  it("renders correct row count", () => {
    render(<DataTable columns={columns} data={data} />);
    expect(screen.getByText(/3 rows/)).toBeInTheDocument();
  });

  it("renders search input when searchable is true", () => {
    render(
      <DataTable
        columns={columns}
        data={data}
        searchable
        searchPlaceholder="Filter rows…"
      />,
    );
    expect(screen.getByPlaceholderText("Filter rows…")).toBeInTheDocument();
  });

  it("does not render search input by default", () => {
    render(<DataTable columns={columns} data={data} />);
    expect(screen.queryByRole("textbox")).not.toBeInTheDocument();
  });

  it("filters rows via global search", async () => {
    const user = userEvent.setup();
    render(
      <DataTable columns={columns} data={data} searchable />,
    );

    const input = screen.getByPlaceholderText("Search…");
    await user.type(input, "Beta");

    // Beta should remain, Alpha/Gamma should be filtered out
    expect(screen.getByText("Beta")).toBeInTheDocument();
    expect(screen.queryByText("Alpha")).not.toBeInTheDocument();
    expect(screen.queryByText("Gamma")).not.toBeInTheDocument();
  });

  it("sorts rows when clicking a column header", async () => {
    const user = userEvent.setup();
    render(<DataTable columns={columns} data={data} />);

    // Click "Name" header to sort ascending
    const nameHeader = screen.getByText("Name");
    await user.click(nameHeader);

    // Verify ascending sort indicator appears
    const headerCell = nameHeader.closest("th");
    expect(headerCell?.textContent).toContain("▲");
  });

  it("handles empty data gracefully", () => {
    render(<DataTable columns={columns} data={[]} />);
    expect(screen.getByText(/0 rows/)).toBeInTheDocument();
  });
});
