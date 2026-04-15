import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";

export default function ProfilePage() {
  return (
    <div className="grid gap-5 xl:grid-cols-[0.9fr_1.1fr]">
      <section className="panel hero-panel rounded-[30px] p-6">
        <div className="eyebrow text-[var(--accent)]">Account Profile</div>
        <h2 className="mt-3 text-4xl font-semibold tracking-[-0.06em]">Manage the analyst identity behind the workspace.</h2>
        <p className="mt-4 max-w-2xl text-sm leading-7 text-[var(--text-soft)]">
          Keep your account information current and preserve a professional product feel even in account-level settings.
        </p>
      </section>

      <section className="panel rounded-[30px] p-6">
        <div className="grid gap-5">
          <label className="space-y-2">
            <span className="text-sm font-medium text-[var(--text)]">Display name</span>
            <Input defaultValue="Portfolio Analyst" />
          </label>
          <label className="space-y-2">
            <span className="text-sm font-medium text-[var(--text)]">Email</span>
            <Input defaultValue="analyst@firm.com" />
          </label>
          <label className="space-y-2">
            <span className="text-sm font-medium text-[var(--text)]">Organization</span>
            <Input defaultValue="Independent Research" />
          </label>
          <div>
            <Button>Save Profile</Button>
          </div>
        </div>
      </section>
    </div>
  );
}
