import "./globals.css";
import type { ReactNode } from "react";

export const metadata = {
  title: "SYNAPSE - Research Intelligence",
  description: "Multi-agent scientific research intelligence and evidence verification",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
