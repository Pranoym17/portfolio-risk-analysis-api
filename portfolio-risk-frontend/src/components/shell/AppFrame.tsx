"use client";

import { usePathname } from "next/navigation";
import { Sidebar } from "./Sidebar";
import { Topbar } from "./Topbar";

export function AppFrame({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const isLogin = pathname === "/login";

  if (isLogin) {
    return <main className="min-h-screen">{children}</main>;
  }

  return (
    <div className="app-shell min-h-screen">
      <div className="mx-auto flex min-h-screen max-w-[1600px] gap-4 px-3 py-3 lg:px-5 lg:py-5">
        <Sidebar />
        <div className="min-w-0 flex-1">
          <Topbar />
          <div className="pt-4 lg:pt-5">{children}</div>
        </div>
      </div>
    </div>
  );
}
