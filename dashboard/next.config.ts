import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // ─── Security Headers ────────────────────────────────────────
  // These are applied to every response from the Next.js server.
  async headers() {
    return [
      {
        source: "/(.*)",
        headers: [
          // Prevent MIME-type sniffing
          { key: "X-Content-Type-Options", value: "nosniff" },
          // Block framing (clickjacking protection)
          { key: "X-Frame-Options", value: "DENY" },
          // Legacy XSS filter
          { key: "X-XSS-Protection", value: "1; mode=block" },
          // Referrer policy
          {
            key: "Referrer-Policy",
            value: "strict-origin-when-cross-origin",
          },
          // HSTS — enforce HTTPS for 1 year
          {
            key: "Strict-Transport-Security",
            value: "max-age=31536000; includeSubDomains; preload",
          },
          // Permissions policy — disable unnecessary browser APIs
          {
            key: "Permissions-Policy",
            value:
              "camera=(), microphone=(), geolocation=(), payment=(), usb=()",
          },
          // Content Security Policy
          {
            key: "Content-Security-Policy",
            value: [
              "default-src 'self'",
              // Next.js requires inline scripts/styles for hydration
              "script-src 'self' 'unsafe-inline' 'unsafe-eval'",
              "style-src 'self' 'unsafe-inline'",
              // Allow connections to the API backend and fonts
              `connect-src 'self' ${process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"}`,
              "font-src 'self' https://fonts.gstatic.com",
              "img-src 'self' data: blob:",
              "frame-ancestors 'none'",
              "base-uri 'self'",
              "form-action 'self'",
            ].join("; "),
          },
          // Cross-origin isolation
          { key: "X-Permitted-Cross-Domain-Policies", value: "none" },
          { key: "Cross-Origin-Opener-Policy", value: "same-origin" },
        ],
      },
    ];
  },

  // ─── General ─────────────────────────────────────────────────
  // Disable x-powered-by to reduce fingerprinting surface
  poweredByHeader: false,

  // Strict mode for React (catches bugs early)
  reactStrictMode: true,
};

export default nextConfig;
