import "./globals.css";
import { AppProviders } from "@/components/providers/AppProviders";

export const metadata = {
  title: "Axiom Portfolio Intelligence",
  description: "Portfolio risk analysis, holdings diagnostics, benchmark comparison, and analytics workflows.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="app-body">
        <AppProviders>
          {children}
        </AppProviders>
      </body>
    </html>
  );
}
