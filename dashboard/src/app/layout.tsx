import type { Metadata } from "next";
import Link from "next/link";
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
          <Link href="/" className="font-semibold text-white hover:text-blue-400">KPI Summary</Link>
          <Link href="/rows" className="text-gray-400 hover:text-white">Row Inspector</Link>
          <Link href="/cost" className="text-gray-400 hover:text-white">Cost Calculator</Link>
          <Link href="/router" className="text-gray-400 hover:text-white">Router Demo</Link>
        </nav>
        <main className="px-6 py-8 max-w-7xl mx-auto">{children}</main>
      </body>
    </html>
  );
}
