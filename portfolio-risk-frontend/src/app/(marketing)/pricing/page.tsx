"use client"

import { motion } from "framer-motion"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import Link from "next/link"
import { Check, ArrowRight, HelpCircle } from "lucide-react"
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip"

const fadeUp = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0 },
}

const stagger = {
  visible: { transition: { staggerChildren: 0.1 } },
}

const plans = [
  {
    name: "Starter",
    price: "Free",
    period: null,
    description: "Perfect for individual investors exploring portfolio analytics",
    features: [
      { text: "3 portfolios", tooltip: "Create and manage up to 3 different portfolios" },
      { text: "Basic risk metrics", tooltip: "Sharpe ratio, volatility, beta, max drawdown" },
      { text: "Daily data refresh", tooltip: "Market data updated once per day at market close" },
      { text: "7-day historical analysis", tooltip: "Access to 7 days of rolling analysis" },
      { text: "Email support", tooltip: "Response within 48 hours" },
    ],
    cta: "Get started",
    href: "/signup",
    highlighted: false,
  },
  {
    name: "Pro",
    price: "$29",
    period: "/month",
    description: "For serious investors who demand professional-grade tools",
    features: [
      { text: "Unlimited portfolios", tooltip: "No limits on portfolio creation" },
      { text: "All risk metrics", tooltip: "Full suite of 25+ risk and performance metrics" },
      { text: "Real-time data", tooltip: "Sub-second market data from global exchanges" },
      { text: "5-year historical analysis", tooltip: "Deep historical backtesting and analysis" },
      { text: "API access", tooltip: "Full REST API for integration" },
      { text: "Monte Carlo simulation", tooltip: "Run 10,000+ scenario simulations" },
      { text: "Custom benchmarks", tooltip: "Create and track custom benchmark indices" },
      { text: "Priority support", tooltip: "Response within 4 hours" },
    ],
    cta: "Start free trial",
    href: "/signup?plan=pro",
    highlighted: true,
  },
  {
    name: "Enterprise",
    price: "Custom",
    period: null,
    description: "For teams, student projects, and advanced portfolio review",
    features: [
      { text: "Everything in Pro", tooltip: null },
      { text: "Unlimited team members", tooltip: "Add your entire investment team" },
      { text: "20+ year historical data", tooltip: "Full historical dataset for backtesting" },
      { text: "Custom integrations", tooltip: "Connect your existing systems" },
      { text: "SSO/SAML authentication", tooltip: "Enterprise identity management" },
      { text: "Dedicated account manager", tooltip: "Personal support contact" },
      { text: "SLA guarantees", tooltip: "99.99% uptime guarantee" },
      { text: "Custom compliance reports", tooltip: "Reports tailored to your regulatory needs" },
    ],
    cta: "Contact support",
    href: "/contact",
    highlighted: false,
  },
]

const faqs = [
  {
    question: "Can I change plans at any time?",
    answer: "Yes, you can upgrade or downgrade your plan at any time. Changes take effect immediately, and we prorate billing accordingly.",
  },
  {
    question: "What happens when my trial ends?",
    answer: "After your 14-day trial, you can continue with a free Starter plan or upgrade to Pro. Your data is never deleted.",
  },
  {
    question: "Do you offer annual billing?",
    answer: "Yes, annual billing is available at a 20% discount. Contact us for Enterprise annual pricing.",
  },
  {
    question: "What payment methods do you accept?",
    answer: "We accept all major credit cards, ACH transfers, and wire transfers for Enterprise plans.",
  },
]

export default function PricingPage() {
  return (
    <TooltipProvider>
      <div className="pt-20">
        {/* Hero */}
        <section className="py-20 lg:py-28">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <motion.div
              initial="hidden"
              animate="visible"
              variants={stagger}
              className="text-center max-w-3xl mx-auto"
            >
              <motion.p variants={fadeUp} className="text-sm font-medium text-primary uppercase tracking-wider mb-3">
                Pricing
              </motion.p>
              <motion.h1 variants={fadeUp} className="text-4xl sm:text-5xl font-bold tracking-tight text-foreground">
                Simple, transparent pricing
              </motion.h1>
              <motion.p variants={fadeUp} className="mt-6 text-xl text-muted-foreground">
                Start free and upgrade as you grow. No hidden fees, no surprises.
              </motion.p>
            </motion.div>
          </div>
        </section>

        {/* Pricing Cards */}
        <section className="pb-20 lg:pb-28">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <motion.div
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true }}
              variants={stagger}
              className="grid md:grid-cols-3 gap-6 lg:gap-8"
            >
              {plans.map((plan) => (
                <motion.div
                  key={plan.name}
                  variants={fadeUp}
                  className={`relative flex flex-col p-6 lg:p-8 rounded-lg border ${
                    plan.highlighted
                      ? "bg-card border-primary shadow-xl ring-1 ring-primary/20"
                      : "bg-card border-border/60 hover:border-border"
                  }`}
                >
                  {plan.highlighted && (
                    <div className="absolute -top-3 left-1/2 -translate-x-1/2">
                      <Badge className="bg-primary text-primary-foreground shadow-sm">Most popular</Badge>
                    </div>
                  )}

                  <div className="mb-6">
                    <h3 className="text-lg font-semibold text-foreground">{plan.name}</h3>
                    <div className="mt-3 flex items-baseline">
                      <span className="text-4xl font-bold text-foreground">{plan.price}</span>
                      {plan.period && (
                        <span className="ml-1 text-muted-foreground">{plan.period}</span>
                      )}
                    </div>
                    <p className="mt-3 text-sm text-muted-foreground">{plan.description}</p>
                  </div>

                  <ul className="flex-1 space-y-3 mb-8">
                    {plan.features.map((feature) => (
                      <li key={feature.text} className="flex items-start gap-2">
                        <Check className="h-4 w-4 text-primary shrink-0 mt-0.5" />
                        <span className="text-sm text-foreground flex items-center gap-1">
                          {feature.text}
                          {feature.tooltip && (
                            <Tooltip>
                              <TooltipTrigger asChild>
                                <HelpCircle className="h-3.5 w-3.5 text-muted-foreground cursor-help" />
                              </TooltipTrigger>
                              <TooltipContent>
                                <p className="max-w-xs">{feature.tooltip}</p>
                              </TooltipContent>
                            </Tooltip>
                          )}
                        </span>
                      </li>
                    ))}
                  </ul>

                  <Button
                    className="w-full"
                    variant={plan.highlighted ? "default" : "outline"}
                    size="lg"
                    asChild
                  >
                    <Link href={plan.href}>
                      {plan.cta}
                      {plan.highlighted && <ArrowRight className="h-4 w-4 ml-2" />}
                    </Link>
                  </Button>
                </motion.div>
              ))}
            </motion.div>
          </div>
        </section>

        {/* FAQ */}
        <section className="py-20 lg:py-28 bg-surface-1">
          <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8">
            <motion.div
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true }}
              variants={stagger}
            >
              <motion.h2 variants={fadeUp} className="text-2xl sm:text-3xl font-bold text-foreground text-center mb-12">
                Frequently asked questions
              </motion.h2>
              
              <motion.div variants={stagger} className="space-y-6">
                {faqs.map((faq) => (
                  <motion.div
                    key={faq.question}
                    variants={fadeUp}
                    className="p-6 bg-card border border-border/60 rounded-lg"
                  >
                    <h3 className="font-semibold text-foreground mb-2">{faq.question}</h3>
                    <p className="text-sm text-muted-foreground">{faq.answer}</p>
                  </motion.div>
                ))}
              </motion.div>

              <motion.div variants={fadeUp} className="mt-12 text-center">
                <p className="text-muted-foreground mb-4">Have more questions?</p>
                <Button variant="outline" asChild>
                  <Link href="/contact">Contact our team</Link>
                </Button>
              </motion.div>
            </motion.div>
          </div>
        </section>
      </div>
    </TooltipProvider>
  )
}
