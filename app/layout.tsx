import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "NextWM",
  description: "A browser-native latent world model built on Next.js",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
