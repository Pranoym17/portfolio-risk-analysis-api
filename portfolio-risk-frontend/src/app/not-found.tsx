import Link from "next/link";
import { Button } from "@/components/ui/Button";

export default function NotFound() {
  return (
    <div className="page-shell flex min-h-screen items-center justify-center px-4">
      <div className="panel rounded-[30px] px-8 py-12 text-center">
        <div className="eyebrow text-[var(--text-faint)]">404</div>
        <h1 className="mt-3 text-5xl font-semibold tracking-[-0.06em]">Page not found</h1>
        <p className="mt-3 max-w-lg text-sm leading-7 text-[var(--text-soft)]">
          The route you requested does not exist in the portfolio analytics workspace.
        </p>
        <div className="mt-6">
          <Link href="/">
            <Button>Back to Home</Button>
          </Link>
        </div>
      </div>
    </div>
  );
}
