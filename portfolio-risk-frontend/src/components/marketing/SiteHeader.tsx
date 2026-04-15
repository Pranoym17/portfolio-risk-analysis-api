"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { ArrowRight, BriefcaseBusiness } from "lucide-react";
import { Button } from "@/components/ui/Button";

const nav = [
  { href: "/features", label: "Platform" },
  { href: "/pricing", label: "Pricing" },
  { href: "/about", label: "About" },
  { href: "/contact", label: "Contact" },
];

export function SiteHeader() {
  return (
    <header className="site-container pt-6">
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.45 }}
        className="glass-strip flex items-center justify-between rounded-[22px] px-4 py-3 sm:px-5"
      >
        <Link href="/" className="flex items-center gap-3">
          <div className="grid h-11 w-11 place-items-center rounded-[16px] border border-[var(--line)] bg-[rgba(123,162,255,0.12)] text-[var(--text)]">
            <BriefcaseBusiness size={20} />
          </div>
          <div>
            <div className="eyebrow text-[var(--text-faint)]">Portfolio Intelligence</div>
            <div className="text-lg font-semibold tracking-[-0.04em]">Axiom Risk</div>
          </div>
        </Link>

        <nav className="hidden items-center gap-6 lg:flex">
          {nav.map((item) => (
            <Link key={item.href} href={item.href} className="text-sm text-[var(--text-soft)] transition hover:text-[var(--text)]">
              {item.label}
            </Link>
          ))}
        </nav>

        <div className="flex items-center gap-2">
          <Link href="/login">
            <Button variant="quiet" className="hidden sm:inline-flex">
              Sign In
            </Button>
          </Link>
          <Link href="/signup">
            <Button>
              Open Workspace
              <ArrowRight size={16} />
            </Button>
          </Link>
        </div>
      </motion.div>
    </header>
  );
}
