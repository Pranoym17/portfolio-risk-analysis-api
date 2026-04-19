"use client"

import { Logo } from "@/components/brand/logo"
import { motion } from "framer-motion"
import { BarChart3, Shield, TrendingUp, Activity } from "lucide-react"

const features = [
  {
    icon: BarChart3,
    title: "Real-time Analytics",
    description: "Track 25+ risk metrics updated in real-time",
  },
  {
    icon: Shield,
    title: "Portfolio Insights",
    description: "Understand your true risk exposure",
  },
  {
    icon: TrendingUp,
    title: "Performance Tracking",
    description: "Compare against benchmarks and peers",
  },
  {
    icon: Activity,
    title: "Smart Alerts",
    description: "Get notified when risk thresholds are breached",
  },
]

export default function AuthLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <div className="min-h-screen grid lg:grid-cols-2">
      {/* Left panel - Branding */}
      <div className="hidden lg:flex flex-col bg-surface-2 relative overflow-hidden">
        {/* Background pattern */}
        <div className="absolute inset-0 pattern-grid opacity-30" />
        <div className="absolute inset-0 bg-gradient-to-br from-primary/5 via-transparent to-transparent" />
        
        {/* Content */}
        <div className="relative flex flex-col h-full p-10">
          <Logo href="/" size="lg" />
          
          <div className="flex-1 flex items-center">
            <div className="max-w-md">
              <motion.h1
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
                className="text-3xl font-bold text-foreground leading-tight"
              >
                Professional portfolio analytics for serious investors
              </motion.h1>
              <motion.p
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.1 }}
                className="mt-4 text-muted-foreground"
              >
                Use the portfolio risk workspace to review holdings, benchmark sensitivity, and rolling metrics.
              </motion.p>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.2 }}
                className="mt-10 space-y-4"
              >
                {features.map((feature, i) => (
                  <motion.div
                    key={feature.title}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.4, delay: 0.3 + i * 0.1 }}
                    className="flex items-start gap-3"
                  >
                    <div className="h-9 w-9 rounded-lg bg-primary/10 flex items-center justify-center shrink-0">
                      <feature.icon className="h-4 w-4 text-primary" />
                    </div>
                    <div>
                      <p className="font-medium text-foreground text-sm">{feature.title}</p>
                      <p className="text-sm text-muted-foreground">{feature.description}</p>
                    </div>
                  </motion.div>
                ))}
              </motion.div>
            </div>
          </div>

          {/* Testimonial */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.6 }}
            className="mt-auto pt-8 border-t border-border/50"
          >
            <blockquote className="text-sm text-muted-foreground italic">
              &quot;This frontend gives portfolio review a much stronger structure for holdings, risk context, and benchmark comparison.&quot;
            </blockquote>
            <div className="mt-3 flex items-center gap-3">
              <div className="h-9 w-9 rounded-full bg-surface-3 flex items-center justify-center text-sm font-semibold text-muted-foreground">
                MR
              </div>
              <div>
                <p className="text-sm font-medium text-foreground">Michael Roberts</p>
                <p className="text-xs text-muted-foreground">Portfolio Manager</p>
              </div>
            </div>
          </motion.div>
        </div>

        {/* Decorative chart visualization */}
        <div className="absolute right-0 top-1/2 -translate-y-1/2 translate-x-1/3 opacity-10">
          <svg width="400" height="400" viewBox="0 0 400 400" className="text-foreground">
            <path
              d="M50 350 L100 280 L150 320 L200 200 L250 240 L300 120 L350 160"
              stroke="currentColor"
              strokeWidth="3"
              fill="none"
            />
            <circle cx="350" cy="160" r="8" fill="currentColor" />
          </svg>
        </div>
      </div>

      {/* Right panel - Auth form */}
      <div className="flex items-center justify-center p-6 sm:p-10 bg-background">
        <div className="w-full max-w-md">
          {/* Mobile logo */}
          <div className="lg:hidden mb-10">
            <Logo href="/" />
          </div>
          {children}
        </div>
      </div>
    </div>
  )
}
