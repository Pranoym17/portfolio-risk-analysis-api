import { Button } from "@/components/ui/Button";
import { Input, Select } from "@/components/ui/Input";

export default function SettingsPage() {
  return (
    <div className="grid gap-5 xl:grid-cols-[0.9fr_1.1fr]">
      <section className="panel hero-panel rounded-[30px] p-6">
        <div className="eyebrow text-[var(--accent)]">Workspace Settings</div>
        <h2 className="mt-3 text-4xl font-semibold tracking-[-0.06em]">Tune the product defaults around how you analyze portfolios.</h2>
        <p className="mt-4 max-w-2xl text-sm leading-7 text-[var(--text-soft)]">
          Choose default benchmark assumptions, notification preferences, and basic display behavior for the workspace.
        </p>
      </section>

      <section className="panel rounded-[30px] p-6">
        <div className="grid gap-5">
          <label className="space-y-2">
            <span className="text-sm font-medium text-[var(--text)]">Default benchmark</span>
            <Select defaultValue="SPY">
              <option value="SPY">SPY</option>
              <option value="QQQ">QQQ</option>
              <option value="VTI">VTI</option>
              <option value="ACWI">ACWI</option>
            </Select>
          </label>
          <label className="space-y-2">
            <span className="text-sm font-medium text-[var(--text)]">Default risk-free rate</span>
            <Input defaultValue="0.02" />
          </label>
          <label className="space-y-2">
            <span className="text-sm font-medium text-[var(--text)]">Notification email</span>
            <Input defaultValue="analyst@firm.com" />
          </label>
          <div>
            <Button>Save Preferences</Button>
          </div>
        </div>
      </section>
    </div>
  );
}
