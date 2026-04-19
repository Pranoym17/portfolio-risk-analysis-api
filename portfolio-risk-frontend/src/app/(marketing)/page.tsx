"use client"

import Link from "next/link"
import { motion } from "framer-motion"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import {
  ArrowRight,
  BarChart3,
  Shield,
  Zap,
  TrendingUp,
  PieChart,
  Activity,
  Lock,
  Globe,
  Check,
} from "lucide-react"

// Animation variants
const fadeUp = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0 },
}

const stagger = {
  visible: { transition: { staggerChildren: 0.08 } },
}

// Hero Section
function HeroSection() {
  return (
    <section className="relative min-h-[90vh] flex items-center pt-20 overflow-hidden">
      {/* Background pattern */}
      <div className="absolute inset-0 pattern-dots opacity-30 dark:opacity-20" />
      <div className="absolute inset-0 bg-gradient-to-b from-transparent via-background/50 to-background" />
      
      {/* Accent glow */}
      <div className="absolute top-1/4 left-1/2 -translate-x-1/2 w-[600px] h-[400px] bg-primary/5 rounded-full blur-3xl" />

      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20 lg:py-32">
        <motion.div
          initial="hidden"
          animate="visible"
          variants={stagger}
          className="max-w-4xl mx-auto text-center"
        >
          <motion.div variants={fadeUp}>
            <Badge variant="secondary" className="mb-6 px-3 py-1 text-xs font-medium">
              Built for portfolio risk workflows
            </Badge>
          </motion.div>

          <motion.h1
            variants={fadeUp}
            className="text-4xl sm:text-5xl lg:text-6xl xl:text-7xl font-bold tracking-tight text-foreground leading-[1.1]"
          >
            <span className="text-balance">
              Portfolio analytics{" "}
              <span className="text-primary">built for precision</span>
            </span>
          </motion.h1>

          <motion.p
            variants={fadeUp}
            className="mt-6 text-lg sm:text-xl text-muted-foreground max-w-2xl mx-auto leading-relaxed text-pretty"
          >
            Professional-grade risk analysis, performance tracking, and portfolio optimization. 
            Review portfolios, holdings, rolling metrics, and benchmark context in one cleaner workspace.
          </motion.p>

          <motion.div
            variants={fadeUp}
            className="mt-10 flex flex-col sm:flex-row items-center justify-center gap-4"
          >
            <Button size="lg" className="h-12 px-8 text-base gap-2" asChild>
              <Link href="/signup">
                Start analyzing
                <ArrowRight className="h-4 w-4" />
              </Link>
            </Button>
            <Button size="lg" variant="outline" className="h-12 px-8 text-base" asChild>
              <Link href="/dashboard">View dashboard</Link>
            </Button>
          </motion.div>

          <motion.p
            variants={fadeUp}
            className="mt-6 text-sm text-muted-foreground"
          >
            Free 14-day trial. No credit card required.
          </motion.p>
        </motion.div>

        {/* Dashboard preview */}
        <motion.div
          initial={{ opacity: 0, y: 60 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4 }}
          className="mt-16 lg:mt-24 relative"
        >
          <div className="relative mx-auto max-w-5xl">
            {/* Browser frame */}
            <div className="rounded-lg border border-border/60 bg-card shadow-2xl shadow-black/10 dark:shadow-black/30 overflow-hidden">
              {/* Browser bar */}
              <div className="flex items-center gap-2 px-4 py-3 bg-surface-2 border-b border-border/50">
                <div className="flex gap-1.5">
                  <div className="h-3 w-3 rounded-full bg-red-500/80" />
                  <div className="h-3 w-3 rounded-full bg-yellow-500/80" />
                  <div className="h-3 w-3 rounded-full bg-green-500/80" />
                </div>
                <div className="flex-1 flex justify-center">
                  <div className="px-3 py-1 bg-background rounded text-xs text-muted-foreground">
                    localhost:3000/dashboard
                  </div>
                </div>
              </div>
              
              {/* Dashboard preview content */}
              <div className="p-6 bg-background">
                <div className="grid grid-cols-4 gap-4 mb-4">
                  {[
                    { label: "Total Value", value: "$1,284,520", change: "+12.4%" },
                    { label: "Sharpe Ratio", value: "1.87", change: "+0.23" },
                    { label: "Volatility", value: "14.2%", change: "-2.1%" },
                    { label: "Max Drawdown", value: "-8.4%", change: "+1.2%" },
                  ].map((metric) => (
                    <div key={metric.label} className="p-4 bg-surface-1 rounded-lg border border-border/40">
                      <p className="text-xs text-muted-foreground mb-1">{metric.label}</p>
                      <p className="text-lg font-semibold tabular-nums">{metric.value}</p>
                      <p className="text-xs text-positive mt-1">{metric.change}</p>
                    </div>
                  ))}
                </div>
                <div className="grid grid-cols-3 gap-4">
                  <div className="col-span-2 h-48 bg-surface-1 rounded-lg border border-border/40 p-4">
                    <div className="text-xs text-muted-foreground mb-3">Portfolio Performance</div>
                    <div className="h-32 flex items-end gap-1">
                      {[40, 35, 45, 50, 48, 55, 52, 60, 58, 65, 70, 68, 75, 80, 78, 85, 90, 88, 95, 100].map((h, i) => (
                        <div key={i} className="flex-1 bg-primary/20 rounded-t" style={{ height: `${h}%` }}>
                          <div className="h-full bg-primary rounded-t" style={{ height: `${h > 50 ? 100 : 60}%` }} />
                        </div>
                      ))}
                    </div>
                  </div>
                  <div className="h-48 bg-surface-1 rounded-lg border border-border/40 p-4">
                    <div className="text-xs text-muted-foreground mb-3">Allocation</div>
                    <div className="h-32 flex items-center justify-center">
                      <div className="relative h-28 w-28">
                        <svg viewBox="0 0 36 36" className="h-full w-full -rotate-90">
                          <circle cx="18" cy="18" r="14" fill="none" stroke="currentColor" strokeWidth="4" className="text-chart-1" strokeDasharray="40 100" />
                          <circle cx="18" cy="18" r="14" fill="none" stroke="currentColor" strokeWidth="4" className="text-chart-2" strokeDasharray="25 100" strokeDashoffset="-40" />
                          <circle cx="18" cy="18" r="14" fill="none" stroke="currentColor" strokeWidth="4" className="text-chart-3" strokeDasharray="20 100" strokeDashoffset="-65" />
                          <circle cx="18" cy="18" r="14" fill="none" stroke="currentColor" strokeWidth="4" className="text-chart-4" strokeDasharray="15 100" strokeDashoffset="-85" />
                        </svg>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Floating elements */}
            <div className="absolute -left-4 top-1/3 p-3 bg-card border border-border/60 rounded-lg shadow-lg hidden lg:block">
              <div className="flex items-center gap-2 text-xs">
                <div className="h-2 w-2 rounded-full bg-positive animate-pulse" />
                <span className="text-muted-foreground">Real-time sync</span>
              </div>
            </div>
            <div className="absolute -right-4 bottom-1/4 p-3 bg-card border border-border/60 rounded-lg shadow-lg hidden lg:block">
              <div className="flex items-center gap-2 text-xs">
                <Shield className="h-4 w-4 text-primary" />
                <span className="text-muted-foreground">Bank-grade security</span>
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  )
}

// Features Section
function FeaturesSection() {
  const features = [
    {
      icon: BarChart3,
      title: "Risk Metrics",
      description: "Sharpe ratio, Sortino, VaR, drawdown, and benchmark-aware metrics presented in a cleaner frontend workspace.",
    },
    {
      icon: Activity,
      title: "Rolling Analysis",
      description: "Track how your metrics evolve over customizable time windows. Identify trends before they impact returns.",
    },
    {
      icon: PieChart,
      title: "Portfolio Optimization",
      description: "Efficient frontier analysis, correlation matrices, and rebalancing recommendations backed by MPT.",
    },
    {
      icon: TrendingUp,
      title: "Benchmark Comparison",
      description: "Compare against S&P 500, custom benchmarks, or peer portfolios. Understand your alpha generation.",
    },
    {
      icon: Zap,
      title: "Real-Time Data",
      description: "Sub-second market data integration. Your analytics reflect the market as it moves.",
    },
    {
      icon: Lock,
      title: "Enterprise Security",
      description: "SOC 2 Type II certified. Your portfolio data is encrypted at rest and in transit.",
    },
  ]

  return (
    <section className="py-24 lg:py-32 bg-surface-1">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-100px" }}
          variants={stagger}
          className="text-center max-w-3xl mx-auto mb-16"
        >
          <motion.p variants={fadeUp} className="text-sm font-medium text-primary uppercase tracking-wider mb-3">
            Capabilities
          </motion.p>
          <motion.h2 variants={fadeUp} className="text-3xl sm:text-4xl font-bold tracking-tight text-foreground">
            Everything you need to analyze risk
          </motion.h2>
          <motion.p variants={fadeUp} className="mt-4 text-lg text-muted-foreground">
            Portfolio analytics surfaces shaped for practical review instead of generic finance dashboards.
          </motion.p>
        </motion.div>

        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-50px" }}
          variants={stagger}
          className="grid md:grid-cols-2 lg:grid-cols-3 gap-6"
        >
          {features.map((feature) => (
            <motion.div
              key={feature.title}
              variants={fadeUp}
              className="group relative p-6 bg-card border border-border/60 rounded-lg hover:border-border hover:shadow-sm transition-all"
            >
              <div className="inline-flex items-center justify-center h-10 w-10 rounded-lg bg-primary/10 text-primary mb-4">
                <feature.icon className="h-5 w-5" />
              </div>
              <h3 className="text-lg font-semibold text-foreground mb-2">{feature.title}</h3>
              <p className="text-sm text-muted-foreground leading-relaxed">{feature.description}</p>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  )
}

// Workflow Section
function WorkflowSection() {
  const steps = [
    {
      step: "01",
      title: "Create your portfolio",
      description: "Add tickers, set weights, and move into holdings and risk review without losing context.",
    },
    {
      step: "02",
      title: "Analyze risk metrics",
      description: "Instantly see Sharpe, volatility, VaR, and correlation analysis. Compare against benchmarks.",
    },
    {
      step: "03",
      title: "Optimize allocation",
      description: "Use our optimization engine to find efficient allocations that match your risk tolerance.",
    },
  ]

  return (
    <section className="py-24 lg:py-32">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid lg:grid-cols-2 gap-12 lg:gap-20 items-center">
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
            variants={stagger}
          >
            <motion.p variants={fadeUp} className="text-sm font-medium text-primary uppercase tracking-wider mb-3">
              How it works
            </motion.p>
            <motion.h2 variants={fadeUp} className="text-3xl sm:text-4xl font-bold tracking-tight text-foreground">
              From portfolio to insights in minutes
            </motion.h2>
            <motion.p variants={fadeUp} className="mt-4 text-lg text-muted-foreground">
              No complex setup. No learning curve. Just powerful analytics that work the way you expect.
            </motion.p>

            <motion.div variants={stagger} className="mt-10 space-y-8">
              {steps.map((step, i) => (
                <motion.div key={i} variants={fadeUp} className="flex gap-4">
                  <div className="flex-shrink-0 h-10 w-10 rounded-lg bg-primary text-primary-foreground flex items-center justify-center text-sm font-semibold">
                    {step.step}
                  </div>
                  <div>
                    <h3 className="text-base font-semibold text-foreground">{step.title}</h3>
                    <p className="mt-1 text-sm text-muted-foreground">{step.description}</p>
                  </div>
                </motion.div>
              ))}
            </motion.div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, x: 40 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            className="relative"
          >
            {/* Analytics visualization */}
            <div className="bg-card border border-border/60 rounded-lg p-6 shadow-lg">
              <div className="flex items-center justify-between mb-6">
                <div>
                  <h3 className="font-semibold text-foreground">Risk Analysis</h3>
                  <p className="text-xs text-muted-foreground">Growth Strategy Portfolio</p>
                </div>
                <Badge variant="secondary">Live</Badge>
              </div>
              
              <div className="space-y-4">
                {[
                  { label: "Sharpe Ratio", value: 1.87, max: 3, color: "bg-chart-1" },
                  { label: "Sortino Ratio", value: 2.14, max: 3, color: "bg-chart-2" },
                  { label: "Beta", value: 0.92, max: 2, color: "bg-chart-3" },
                  { label: "Alpha", value: 0.08, max: 0.2, color: "bg-chart-4" },
                ].map((metric) => (
                  <div key={metric.label}>
                    <div className="flex items-center justify-between text-sm mb-1.5">
                      <span className="text-muted-foreground">{metric.label}</span>
                      <span className="font-medium tabular-nums">{metric.value.toFixed(2)}</span>
                    </div>
                    <div className="h-2 bg-surface-2 rounded-full overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        whileInView={{ width: `${(metric.value / metric.max) * 100}%` }}
                        viewport={{ once: true }}
                        transition={{ duration: 0.8, delay: 0.2 }}
                        className={`h-full rounded-full ${metric.color}`}
                      />
                    </div>
                  </div>
                ))}
              </div>

              <div className="mt-6 pt-6 border-t border-border/50">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-xs text-muted-foreground">VaR (95%)</p>
                    <p className="text-lg font-semibold text-negative">-2.34%</p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">Expected Return</p>
                    <p className="text-lg font-semibold text-positive">+12.8%</p>
                  </div>
                </div>
              </div>
            </div>

            {/* Decorative elements */}
            <div className="absolute -z-10 inset-0 bg-gradient-to-r from-primary/5 to-transparent rounded-lg transform translate-x-4 translate-y-4" />
          </motion.div>
        </div>
      </div>
    </section>
  )
}

// Metrics Showcase
function MetricsShowcase() {
  const metrics = [
    { value: "50K+", label: "Portfolios Analyzed" },
    { value: "99.9%", label: "Uptime SLA" },
    { value: "<100ms", label: "Query Latency" },
    { value: "24/7", label: "Market Coverage" },
  ]

  return (
    <section className="py-16 border-y border-border bg-surface-2">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          variants={stagger}
          className="grid grid-cols-2 lg:grid-cols-4 gap-8 lg:gap-12"
        >
          {metrics.map((metric) => (
            <motion.div key={metric.label} variants={fadeUp} className="text-center">
              <p className="text-3xl sm:text-4xl font-bold text-foreground tabular-nums">{metric.value}</p>
              <p className="mt-2 text-sm text-muted-foreground">{metric.label}</p>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  )
}

// Pricing Teaser
function PricingTeaser() {
  const plans = [
    {
      name: "Starter",
      price: "Free",
      description: "For individual investors getting started",
      features: ["3 portfolios", "Basic risk metrics", "Daily data refresh", "Email support"],
      cta: "Get started",
      highlighted: false,
    },
    {
      name: "Pro",
      price: "$29",
      period: "/month",
      description: "For serious investors and analysts",
      features: ["Unlimited portfolios", "All risk metrics", "Real-time data", "API access", "Priority support"],
      cta: "Start free trial",
      highlighted: true,
    },
    {
      name: "Enterprise",
      price: "Custom",
      description: "For teams and institutions",
      features: ["Everything in Pro", "Custom integrations", "Dedicated support", "SLA guarantees", "SSO/SAML"],
      cta: "Contact support",
      highlighted: false,
    },
  ]

  return (
    <section className="py-24 lg:py-32">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-100px" }}
          variants={stagger}
          className="text-center max-w-3xl mx-auto mb-16"
        >
          <motion.p variants={fadeUp} className="text-sm font-medium text-primary uppercase tracking-wider mb-3">
            Pricing
          </motion.p>
          <motion.h2 variants={fadeUp} className="text-3xl sm:text-4xl font-bold tracking-tight text-foreground">
            Simple, transparent pricing
          </motion.h2>
          <motion.p variants={fadeUp} className="mt-4 text-lg text-muted-foreground">
            Start free and scale as you grow. No hidden fees.
          </motion.p>
        </motion.div>

        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-50px" }}
          variants={stagger}
          className="grid md:grid-cols-3 gap-6 lg:gap-8"
        >
          {plans.map((plan) => (
            <motion.div
              key={plan.name}
              variants={fadeUp}
              className={`relative p-6 lg:p-8 rounded-lg border transition-all ${
                plan.highlighted
                  ? "bg-card border-primary shadow-lg scale-[1.02]"
                  : "bg-card border-border/60 hover:border-border"
              }`}
            >
              {plan.highlighted && (
                <div className="absolute -top-3 left-1/2 -translate-x-1/2">
                  <Badge className="bg-primary text-primary-foreground">Most popular</Badge>
                </div>
              )}
              
              <div className="mb-6">
                <h3 className="text-lg font-semibold text-foreground">{plan.name}</h3>
                <div className="mt-2 flex items-baseline">
                  <span className="text-4xl font-bold text-foreground">{plan.price}</span>
                  {plan.period && <span className="text-muted-foreground ml-1">{plan.period}</span>}
                </div>
                <p className="mt-2 text-sm text-muted-foreground">{plan.description}</p>
              </div>

              <ul className="space-y-3 mb-8">
                {plan.features.map((feature) => (
                  <li key={feature} className="flex items-center gap-2 text-sm">
                    <Check className="h-4 w-4 text-primary shrink-0" />
                    <span className="text-foreground">{feature}</span>
                  </li>
                ))}
              </ul>

              <Button 
                className="w-full" 
                variant={plan.highlighted ? "default" : "outline"}
                asChild
              >
                <Link href={plan.highlighted ? "/signup" : "/contact"}>{plan.cta}</Link>
              </Button>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  )
}

// CTA Section
function CTASection() {
  return (
    <section className="py-24 lg:py-32 bg-surface-1 relative overflow-hidden">
      {/* Background pattern */}
      <div className="absolute inset-0 pattern-grid opacity-30" />
      <div className="absolute inset-0 bg-gradient-to-t from-surface-1 via-transparent to-surface-1" />
      
      <div className="relative max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-100px" }}
          variants={stagger}
        >
          <motion.div variants={fadeUp} className="inline-flex items-center gap-2 mb-6">
            <Globe className="h-5 w-5 text-primary" />
            <span className="text-sm font-medium text-primary">Join 10,000+ investors worldwide</span>
          </motion.div>
          
          <motion.h2 variants={fadeUp} className="text-3xl sm:text-4xl lg:text-5xl font-bold tracking-tight text-foreground">
            Ready to take control of your portfolio risk?
          </motion.h2>
          
          <motion.p variants={fadeUp} className="mt-6 text-lg text-muted-foreground max-w-2xl mx-auto">
            Use this frontend as the presentation layer for your portfolio risk analysis backend.
            Start your free trial today.
          </motion.p>
          
          <motion.div variants={fadeUp} className="mt-10 flex flex-col sm:flex-row items-center justify-center gap-4">
            <Button size="lg" className="h-12 px-8 text-base gap-2" asChild>
              <Link href="/signup">
                Start your free trial
                <ArrowRight className="h-4 w-4" />
              </Link>
            </Button>
            <Button size="lg" variant="ghost" className="h-12 px-8 text-base" asChild>
              <Link href="/contact">Contact support</Link>
            </Button>
          </motion.div>
        </motion.div>
      </div>
    </section>
  )
}

// Main Page
export default function LandingPage() {
  return (
    <>
      <HeroSection />
      <FeaturesSection />
      <WorkflowSection />
      <MetricsShowcase />
      <PricingTeaser />
      <CTASection />
    </>
  )
}
