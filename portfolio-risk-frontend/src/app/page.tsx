import { SiteHeader } from "@/components/marketing/SiteHeader";
import { SiteFooter } from "@/components/marketing/SiteFooter";
import { LandingPage } from "@/components/marketing/LandingPage";

export default function HomePage() {
  return (
    <div className="marketing-shell">
      <SiteHeader />
      <LandingPage />
      <SiteFooter />
    </div>
  );
}
