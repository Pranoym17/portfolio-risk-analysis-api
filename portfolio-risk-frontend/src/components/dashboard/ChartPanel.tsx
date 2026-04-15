import { cn } from "@/lib/utils";
import { Panel, PanelHeader } from "@/components/ui/Panel";

export function ChartPanel({
  kicker,
  title,
  description,
  children,
  className,
  action,
}: {
  kicker: string;
  title: string;
  description?: string;
  children: React.ReactNode;
  className?: string;
  action?: React.ReactNode;
}) {
  return (
    <Panel className={cn("rounded-[26px]", className)}>
      <PanelHeader eyebrow={kicker} title={title} description={description} action={action} />
      <div className="px-5 py-4">{children}</div>
    </Panel>
  );
}
