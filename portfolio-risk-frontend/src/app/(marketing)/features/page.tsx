"use client"

import { motion } from "framer-motion"
import { Button } from "@/components/ui/button"
import Link from "next/link"
import {
  BarChart3,
  Activity,
  PieChart,
  TrendingUp,
  Zap,
  Lock,
  Database,
  LineChart,
  ArrowRight,
  Calculator,
  GitBranch,
  Clock,
  FileSpreadsheet,
} from "lucide-react"

const fadeUp = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0 },
}

const stagger = {
  visible: { transition: { staggerChildren: 0.1 } },
}

const primaryFeatures = [
  {
    icon: BarChart3,
    title: "Comprehensive Risk Metrics",
    description: "Calculate Sharpe ratio, Sortino ratio, Beta, Alpha, VaR, CVaR, maximum drawdown, and 20+ other institutional-grade risk metrics. All computed in real-time as markets move.",
    details: ["Real-time calculation", "Customizable time periods", "Benchmark comparison"],
  },
  {
    icon: Activity,
    title: "Rolling Window Analysis",
    description: "Track how your portfolio metrics evolve over time with customizable rolling windows. Identify trends, regime changes, and risk build-up before they impact your returns.",
    details: ["Configurable windows", "Multiple timeframes", "Trend detection"],
  },
  {
    icon: PieChart,
    title: "Portfolio Optimization",
    description: "Leverage Modern Portfolio Theory to find optimal allocations. Visualize the efficient frontier, understand the risk-return tradeoff, and get rebalancing recommendations.",
    details: ["Efficient frontier", "Constraint support", "Rebalancing suggestions"],
  },
  {
    icon: TrendingUp,
    title: "Benchmark Comparison",
    description: "Compare your portfolio against major indices, custom benchmarks, or peer portfolios. Understand your true alpha generation and tracking error.",
    details: ["Multiple benchmarks", "Attribution analysis", "Peer comparison"],
  },
]

const secondaryFeatures = [
  {
    icon: Zap,
    title: "Real-Time Data",
    description: "Sub-second market data from global exchanges. Your analytics always reflect current market conditions.",
  },
  {
    icon: Database,
    title: "Historical Analysis",
    description: "20+ years of historical data for backtesting and scenario analysis. Test your strategies against past market events.",
  },
  {
    icon: Calculator,
    title: "Monte Carlo Simulation",
    description: "Run thousands of scenarios to understand potential outcomes. Stress test your portfolio against extreme events.",
  },
  {
    icon: LineChart,
    title: "Correlation Matrix",
    description: "Understand how your holdings move together. Identify concentration risks and diversification opportunities.",
  },
  {
    icon: GitBranch,
    title: "Version History",
    description: "Track changes to your portfolio over time. Compare different allocation strategies side by side.",
  },
  {
    icon: Clock,
    title: "Scheduled Reports",
    description: "Automated daily, weekly, or monthly reports delivered to your inbox. Stay informed without logging in.",
  },
  {
    icon: FileSpreadsheet,
    title: "CSV/Excel Import",
    description: "Import your existing portfolios from spreadsheets and common portfolio export formats.",
  },
  {
    icon: Lock,
    title: "Enterprise Security",
    description: "SOC 2 Type II certified. Bank-grade encryption for data at rest and in transit. Your data stays yours.",
  },
]

export default function FeaturesPage() {
  return (
    <div className="pt-20">
      {/* Hero */}
      <section className="py-20 lg:py-28">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial="hidden"
            animate="visible"
            variants={stagger}
            className="max-w-3xl"
          >
            <motion.p variants={fadeUp} className="text-sm font-medium text-primary uppercase tracking-wider mb-3">
              Features
            </motion.p>
            <motion.h1 variants={fadeUp} className="text-4xl sm:text-5xl font-bold tracking-tight text-foreground">
              Institutional-grade analytics, accessible to everyone
            </motion.h1>
            <motion.p variants={fadeUp} className="mt-6 text-xl text-muted-foreground">
              Every tool you need to analyze, optimize, and monitor your portfolio risk. 
              Built for serious investors who demand precision.
            </motion.p>
          </motion.div>
        </div>
      </section>

      {/* Primary Features */}
      <section className="py-16 lg:py-24 bg-surface-1">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
            variants={stagger}
            className="space-y-16 lg:space-y-24"
          >
            {primaryFeatures.map((feature, i) => (
              <motion.div
                key={feature.title}
                variants={fadeUp}
                className={`grid lg:grid-cols-2 gap-12 items-center ${i % 2 === 1 ? "lg:flex-row-reverse" : ""}`}
              >
                <div className={i % 2 === 1 ? "lg:order-2" : ""}>
                  <div className="inline-flex items-center justify-center h-12 w-12 rounded-lg bg-primary/10 text-primary mb-6">
                    <feature.icon className="h-6 w-6" />
                  </div>
                  <h2 className="text-2xl sm:text-3xl font-bold text-foreground">{feature.title}</h2>
                  <p className="mt-4 text-lg text-muted-foreground leading-relaxed">{feature.description}</p>
                  <ul className="mt-6 space-y-2">
                    {feature.details.map((detail) => (
                      <li key={detail} className="flex items-center gap-2 text-sm text-foreground">
                        <div className="h-1.5 w-1.5 rounded-full bg-primary" />
                        {detail}
                      </li>
                    ))}
                  </ul>
                </div>
                
                <div className={`relative ${i % 2 === 1 ? "lg:order-1" : ""}`}>
                  <div className="bg-card border border-border/60 rounded-lg p-6 shadow-lg">
                    <div className="h-64 flex items-center justify-center">
                      <div className="text-center">
                        <feature.icon className="h-16 w-16 text-primary/20 mx-auto mb-4" />
                        <p className="text-sm text-muted-foreground">Feature visualization</p>
                      </div>
                    </div>
                  </div>
                  <div className="absolute -z-10 inset-0 bg-gradient-to-r from-primary/5 to-transparent rounded-lg transform translate-x-4 translate-y-4" />
                </div>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* Secondary Features Grid */}
      <section className="py-20 lg:py-28">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
            variants={stagger}
            className="text-center max-w-3xl mx-auto mb-16"
          >
            <motion.h2 variants={fadeUp} className="text-3xl sm:text-4xl font-bold tracking-tight text-foreground">
              And much more
            </motion.h2>
            <motion.p variants={fadeUp} className="mt-4 text-lg text-muted-foreground">
              Every feature is designed to give you a complete picture of your portfolio risk.
            </motion.p>
          </motion.div>

          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-50px" }}
            variants={stagger}
            className="grid sm:grid-cols-2 lg:grid-cols-4 gap-6"
          >
            {secondaryFeatures.map((feature) => (
              <motion.div
                key={feature.title}
                variants={fadeUp}
                className="p-5 bg-card border border-border/60 rounded-lg hover:border-border hover:shadow-sm transition-all"
              >
                <feature.icon className="h-5 w-5 text-primary mb-3" />
                <h3 className="font-semibold text-foreground mb-2">{feature.title}</h3>
                <p className="text-sm text-muted-foreground">{feature.description}</p>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-20 lg:py-28 bg-surface-1">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            variants={stagger}
          >
            <motion.h2 variants={fadeUp} className="text-3xl sm:text-4xl font-bold tracking-tight text-foreground">
              Ready to see it in action?
            </motion.h2>
            <motion.p variants={fadeUp} className="mt-4 text-lg text-muted-foreground">
              Start your free 14-day trial. No credit card required.
            </motion.p>
            <motion.div variants={fadeUp} className="mt-8 flex items-center justify-center gap-4">
              <Button size="lg" className="h-12 px-8 gap-2" asChild>
                <Link href="/signup">
                  Get started free
                  <ArrowRight className="h-4 w-4" />
                </Link>
              </Button>
              <Button size="lg" variant="outline" className="h-12 px-8" asChild>
                <Link href="/dashboard">View dashboard</Link>
              </Button>
            </motion.div>
          </motion.div>
        </div>
      </section>
    </div>
  )
}
