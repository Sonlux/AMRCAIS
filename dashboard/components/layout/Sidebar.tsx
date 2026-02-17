"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutGrid,
  Layers,
  PuzzleIcon,
  ScatterChart,
  FlaskConical,
  Brain,
  TrendingUp,
  Network,
  BarChart3,
  Shield,
  Bell,
  Wallet,
  BookOpen,
  FileText,
} from "lucide-react";
import { cn } from "@/lib/utils";

const NAV_ITEMS = [
  { href: "/", icon: LayoutGrid, label: "Overview" },
  { href: "/regime", icon: Layers, label: "Regime" },
  { href: "/modules", icon: PuzzleIcon, label: "Modules" },
  { href: "/correlations", icon: ScatterChart, label: "Correlations" },
  { href: "/backtest", icon: FlaskConical, label: "Backtest" },
  { href: "/meta", icon: Brain, label: "Meta" },
  // Phase 2
  { href: "/intelligence", icon: TrendingUp, label: "Intelligence" },
  { href: "/contagion", icon: Network, label: "Contagion" },
  // Phase 3
  { href: "/predictions", icon: BarChart3, label: "Predictions" },
  { href: "/risk", icon: Shield, label: "Risk" },
  // Phase 4
  { href: "/alerts", icon: Bell, label: "Alerts" },
  { href: "/trading", icon: Wallet, label: "Trading" },
  // Phase 5
  { href: "/knowledge", icon: BookOpen, label: "Knowledge" },
  { href: "/research", icon: FileText, label: "Research" },
] as const;

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="group fixed left-0 top-0 z-40 flex h-screen w-14 flex-col border-r border-border bg-surface transition-all duration-200 hover:w-48">
      {/* Logo area */}
      <div className="flex h-14 items-center justify-center border-b border-border">
        <span className="font-mono text-sm font-bold text-accent">A</span>
        <span className="hidden overflow-hidden whitespace-nowrap font-mono text-sm font-bold text-accent group-hover:inline">
          MRCAIS
        </span>
      </div>

      {/* Nav items */}
      <nav className="mt-4 flex flex-1 flex-col gap-1 px-2">
        {NAV_ITEMS.map(({ href, icon: Icon, label }) => {
          const isActive =
            href === "/" ? pathname === "/" : pathname.startsWith(href);
          return (
            <Link
              key={href}
              href={href}
              className={cn(
                "relative flex items-center gap-3 rounded-md px-2.5 py-2 text-sm transition-colors",
                isActive
                  ? "bg-surface-elevated text-foreground"
                  : "text-text-secondary hover:bg-surface-elevated hover:text-foreground",
              )}
            >
              {isActive && (
                <span className="absolute left-0 top-1/2 h-5 w-0.5 -translate-y-1/2 rounded-r bg-accent" />
              )}
              <Icon size={18} className="shrink-0" />
              <span className="hidden overflow-hidden whitespace-nowrap group-hover:inline">
                {label}
              </span>
            </Link>
          );
        })}
      </nav>

      {/* Bottom */}
      <div className="border-t border-border px-3 py-3">
        <span className="hidden truncate text-xs text-text-muted group-hover:block">
          v1.0.0
        </span>
      </div>
    </aside>
  );
}
