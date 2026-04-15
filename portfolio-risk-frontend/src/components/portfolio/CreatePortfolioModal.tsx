"use client";

import { AnimatePresence, motion } from "framer-motion";
import { useState } from "react";
import { toast } from "sonner";
import { createPortfolio } from "@/lib/api";
import { getErrorMessage } from "@/lib/utils";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";

export function CreatePortfolioModal({
  open,
  onClose,
  onCreated,
}: {
  open: boolean;
  onClose: () => void;
  onCreated: () => void;
}) {
  const [name, setName] = useState("");
  const [loading, setLoading] = useState(false);

  async function submit() {
    const trimmed = name.trim();
    if (!trimmed) {
      toast.error("Portfolio name is required");
      return;
    }

    setLoading(true);
    try {
      await createPortfolio(trimmed);
      toast.success("Portfolio created");
      setName("");
      onCreated();
      onClose();
    } catch (error: unknown) {
      toast.error(getErrorMessage(error, "Unable to create portfolio"));
    } finally {
      setLoading(false);
    }
  }

  return (
    <AnimatePresence>
      {open ? (
        <motion.div
          className="fixed inset-0 z-50 flex items-center justify-center bg-[rgba(5,11,19,0.62)] px-4 backdrop-blur-sm"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
        >
          <motion.div
            className="surface-dark w-full max-w-lg rounded-[26px] p-6"
            initial={{ opacity: 0, y: 22, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 18, scale: 0.98 }}
            transition={{ duration: 0.24, ease: "easeOut" }}
          >
            <div className="section-kicker text-slate-400">New Portfolio</div>
            <h3 className="mt-3 text-3xl font-semibold tracking-[-0.05em] text-white">Create a portfolio shell</h3>
            <p className="mt-3 text-sm leading-7 text-slate-300">
              Start with a clear mandate-style name, then move straight into holdings and analytics.
            </p>

            <div className="mt-6 space-y-3">
              <label className="block space-y-2">
                <span className="text-sm font-semibold text-white">Portfolio name</span>
                <Input
                  value={name}
                  onChange={(event) => setName(event.target.value)}
                  placeholder="US Quality Compounders"
                  className="border-white/10 bg-white/6 text-white placeholder:text-slate-500"
                />
              </label>

              <div className="rounded-[18px] border border-white/10 bg-white/6 p-4 text-sm leading-7 text-slate-300">
                Examples: Global Core Equity, Balanced Income Sleeve, Technology Concentrated Growth.
              </div>
            </div>

            <div className="mt-6 flex items-center justify-end gap-3">
              <Button variant="ghost" onClick={onClose}>
                Cancel
              </Button>
              <Button loading={loading} onClick={submit}>
                Create Portfolio
              </Button>
            </div>
          </motion.div>
        </motion.div>
      ) : null}
    </AnimatePresence>
  );
}
