import type { Metadata } from "next";
import "./styles.css";

export const metadata: Metadata = {
  title: "OrthoLens Research Prototype",
  description: "Knee radiograph research workflow. Not for clinical diagnosis.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

