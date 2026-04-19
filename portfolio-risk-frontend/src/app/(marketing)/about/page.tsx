"use client"

import { motion } from "framer-motion"
import { Button } from "@/components/ui/button"
import Link from "next/link"
import { ArrowRight, Target, Users, Lightbulb, Award } from "lucide-react"

const fadeUp = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0 },
}

const stagger = {
  visible: { transition: { staggerChildren: 0.1 } },
}

const values = [
  {
    icon: Target,
    title: "Precision First",
    description: "Every metric, every calculation, every insight is designed to be accurate and actionable. We don't compromise on data quality.",
  },
  {
    icon: Users,
    title: "Built for Users",
    description: "Complex analytics shouldn't mean complex interfaces. We obsess over making powerful tools accessible.",
  },
  {
    icon: Lightbulb,
    title: "Continuous Innovation",
    description: "The markets evolve, and so do we. We're constantly adding new metrics, data sources, and analytical capabilities.",
  },
  {
    icon: Award,
    title: "Trust & Security",
    description: "Your portfolio data is sensitive. We maintain the highest security standards and never sell your data.",
  },
]

const team = [
  { name: "Sarah Chen", role: "CEO & Co-founder", image: null },
  { name: "Michael Torres", role: "CTO & Co-founder", image: null },
  { name: "Emily Watson", role: "Head of Product", image: null },
  { name: "David Kim", role: "Head of Engineering", image: null },
]

export default function AboutPage() {
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
              About Portfolio Risk Analysis
            </motion.p>
            <motion.h1 variants={fadeUp} className="text-4xl sm:text-5xl font-bold tracking-tight text-foreground">
              Building a sharper interface for portfolio risk review
            </motion.h1>
            <motion.p variants={fadeUp} className="mt-6 text-xl text-muted-foreground leading-relaxed">
              This project turns a portfolio risk backend into a more polished product experience for holdings, metrics, and benchmark-aware analysis.
            </motion.p>
          </motion.div>
        </div>
      </section>

      {/* Story */}
      <section className="py-16 lg:py-24 bg-surface-1">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid lg:grid-cols-2 gap-12 lg:gap-20 items-center">
            <motion.div
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true }}
              variants={stagger}
            >
              <motion.h2 variants={fadeUp} className="text-3xl font-bold text-foreground mb-6">
                Our story
              </motion.h2>
              <motion.div variants={stagger} className="space-y-4 text-muted-foreground">
                <motion.p variants={fadeUp}>
                  The goal here is straightforward: take serious portfolio analytics and present them in a cleaner, more usable web interface.
                </motion.p>
                <motion.p variants={fadeUp}>
                  Users should be able to understand portfolio construction, rolling metrics, and benchmark context without digging through rough or generic UI.
                </motion.p>
                <motion.p variants={fadeUp}>
                  The frontend is designed to sit on top of your backend and make those analytics easier to navigate, inspect, and present.
                </motion.p>
              </motion.div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 40 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6 }}
              className="relative"
            >
              <div className="bg-card border border-border/60 rounded-lg p-8 shadow-lg">
                <div className="grid grid-cols-2 gap-8">
                  <div>
                    <p className="text-4xl font-bold text-foreground">50K+</p>
                    <p className="text-sm text-muted-foreground mt-1">Active users</p>
                  </div>
                  <div>
                    <p className="text-4xl font-bold text-foreground">$2B+</p>
                    <p className="text-sm text-muted-foreground mt-1">Assets analyzed</p>
                  </div>
                  <div>
                    <p className="text-4xl font-bold text-foreground">150+</p>
                    <p className="text-sm text-muted-foreground mt-1">Countries</p>
                  </div>
                  <div>
                    <p className="text-4xl font-bold text-foreground">99.9%</p>
                    <p className="text-sm text-muted-foreground mt-1">Uptime</p>
                  </div>
                </div>
              </div>
              <div className="absolute -z-10 inset-0 bg-gradient-to-r from-primary/5 to-transparent rounded-lg transform translate-x-4 translate-y-4" />
            </motion.div>
          </div>
        </div>
      </section>

      {/* Values */}
      <section className="py-20 lg:py-28">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            variants={stagger}
            className="text-center max-w-3xl mx-auto mb-16"
          >
            <motion.h2 variants={fadeUp} className="text-3xl sm:text-4xl font-bold text-foreground">
              What we stand for
            </motion.h2>
            <motion.p variants={fadeUp} className="mt-4 text-lg text-muted-foreground">
              Our values guide every decision we make.
            </motion.p>
          </motion.div>

          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            variants={stagger}
            className="grid md:grid-cols-2 gap-6"
          >
            {values.map((value) => (
              <motion.div
                key={value.title}
                variants={fadeUp}
                className="p-6 bg-card border border-border/60 rounded-lg hover:border-border transition-colors"
              >
                <div className="inline-flex items-center justify-center h-10 w-10 rounded-lg bg-primary/10 text-primary mb-4">
                  <value.icon className="h-5 w-5" />
                </div>
                <h3 className="text-lg font-semibold text-foreground mb-2">{value.title}</h3>
                <p className="text-muted-foreground">{value.description}</p>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* Team */}
      <section className="py-20 lg:py-28 bg-surface-1">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            variants={stagger}
            className="text-center max-w-3xl mx-auto mb-16"
          >
            <motion.h2 variants={fadeUp} className="text-3xl sm:text-4xl font-bold text-foreground">
              Our leadership
            </motion.h2>
            <motion.p variants={fadeUp} className="mt-4 text-lg text-muted-foreground">
              A team of finance and engineering experts building the future of portfolio analytics.
            </motion.p>
          </motion.div>

          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            variants={stagger}
            className="grid sm:grid-cols-2 lg:grid-cols-4 gap-6"
          >
            {team.map((person) => (
              <motion.div
                key={person.name}
                variants={fadeUp}
                className="text-center"
              >
                <div className="h-32 w-32 mx-auto rounded-full bg-surface-3 border border-border/60 flex items-center justify-center text-2xl font-bold text-muted-foreground mb-4">
                  {person.name.split(" ").map(n => n[0]).join("")}
                </div>
                <h3 className="font-semibold text-foreground">{person.name}</h3>
                <p className="text-sm text-muted-foreground mt-1">{person.role}</p>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-20 lg:py-28">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            variants={stagger}
          >
            <motion.h2 variants={fadeUp} className="text-3xl sm:text-4xl font-bold text-foreground">
              Join us on our mission
            </motion.h2>
            <motion.p variants={fadeUp} className="mt-4 text-lg text-muted-foreground">
              Whether you&apos;re an investor looking for better tools or a talented builder wanting 
              to shape the future of finance, we&apos;d love to hear from you.
            </motion.p>
            <motion.div variants={fadeUp} className="mt-8 flex items-center justify-center gap-4">
              <Button size="lg" className="h-12 px-8 gap-2" asChild>
                <Link href="/signup">
                  Get started
                  <ArrowRight className="h-4 w-4" />
                </Link>
              </Button>
              <Button size="lg" variant="outline" className="h-12 px-8" asChild>
                <Link href="/contact">Contact support</Link>
              </Button>
            </motion.div>
          </motion.div>
        </div>
      </section>
    </div>
  )
}
