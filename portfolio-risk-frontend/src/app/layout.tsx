import "./globals.css";
import { AppFrame } from "@/components/shell/AppFrame";
import { AppProviders } from "@/components/providers/AppProviders";

export const metadata = {
  title: "Portfolio Risk Analyst Workspace",
  description: "Premium portfolio risk analytics frontend for holdings, metrics, and attribution.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <AppProviders>
          <AppFrame>{children}</AppFrame>
        </AppProviders>
      </body>
    </html>
  );
}
