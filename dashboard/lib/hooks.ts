"use client";

import { useSearchParams, useRouter, usePathname } from "next/navigation";
import { useCallback } from "react";

/**
 * Hook that syncs a single state value with a URL search parameter.
 *
 * Usage:
 *   const [tab, setTab] = useQueryState("tab", "macro");
 *   // URL becomes ?tab=macro, clicking setTab("yield_curve") → ?tab=yield_curve
 *
 * Falls back to `defaultValue` when the param is absent.
 * Uses shallow navigation (no full page reload).
 */
export function useQueryState(
  key: string,
  defaultValue: string,
): [string, (value: string) => void] {
  const searchParams = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();

  const value = searchParams.get(key) ?? defaultValue;

  const setValue = useCallback(
    (newValue: string) => {
      const params = new URLSearchParams(searchParams.toString());
      if (newValue === defaultValue) {
        params.delete(key);
      } else {
        params.set(key, newValue);
      }
      const qs = params.toString();
      router.replace(`${pathname}${qs ? `?${qs}` : ""}`, { scroll: false });
    },
    [key, defaultValue, searchParams, router, pathname],
  );

  return [value, setValue];
}

/**
 * Hook that syncs a numeric state value with a URL search parameter.
 *
 * Usage:
 *   const [window, setWindow] = useQueryNumber("window", 60);
 */
export function useQueryNumber(
  key: string,
  defaultValue: number,
): [number, (value: number) => void] {
  const searchParams = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();

  const raw = searchParams.get(key);
  const value = raw !== null ? Number(raw) : defaultValue;

  const setValue = useCallback(
    (newValue: number) => {
      const params = new URLSearchParams(searchParams.toString());
      if (newValue === defaultValue) {
        params.delete(key);
      } else {
        params.set(key, String(newValue));
      }
      const qs = params.toString();
      router.replace(`${pathname}${qs ? `?${qs}` : ""}`, { scroll: false });
    },
    [key, defaultValue, searchParams, router, pathname],
  );

  return [value, setValue];
}
