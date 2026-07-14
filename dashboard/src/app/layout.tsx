import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "E-commerce Return Gatekeeper",
  description: "Multi-tiered LLM dispute arbitration — benchmark dashboard",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-gray-950 text-gray-100 min-h-screen">
        <nav className="border-b border-gray-800 px-6 py-3 flex gap-6 text-sm">
          <a href="/" className="font-semibold text-white hover:text-blue-400">KPI Summary</a>
          <a href="/rows" className="text-gray-400 hover:text-white">Row Inspector</a>
          <a href="/cost" className="text-gray-400 hover:text-white">Cost Calculator</a>
          <a href="/router" className="text-gray-400 hover:text-white">Router Demo</a>
        </nav>
        <main className="px-6 py-8 max-w-7xl mx-auto">{children}</main>
      </body>
    </html>
  );
}
