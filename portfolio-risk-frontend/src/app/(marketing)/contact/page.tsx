"use client"

import { useState } from "react"
import { motion } from "framer-motion"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Label } from "@/components/ui/label"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Mail, MapPin, Phone, MessageSquare, Clock, CheckCircle2 } from "lucide-react"

const fadeUp = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0 },
}

const stagger = {
  visible: { transition: { staggerChildren: 0.1 } },
}

const contactMethods = [
  {
    icon: Mail,
    title: "Email",
    description: "Our team typically responds within 24 hours",
    value: "support@portfolio-risk.app",
  },
  {
    icon: MessageSquare,
    title: "Live Chat",
    description: "Available Monday to Friday, 9am-6pm EST",
    value: "Start a conversation",
  },
  {
    icon: Phone,
    title: "Phone",
    description: "For Enterprise customers only",
    value: "+1 (555) 123-4567",
  },
]

export default function ContactPage() {
  const [submitted, setSubmitted] = useState(false)

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    setSubmitted(true)
  }

  return (
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
              Contact
            </motion.p>
            <motion.h1 variants={fadeUp} className="text-4xl sm:text-5xl font-bold tracking-tight text-foreground">
              Get in touch
            </motion.h1>
            <motion.p variants={fadeUp} className="mt-6 text-xl text-muted-foreground">
              Have questions about the portfolio risk frontend? Support is here to help.
            </motion.p>
          </motion.div>
        </div>
      </section>

      {/* Contact Methods */}
      <section className="pb-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            variants={stagger}
            className="grid md:grid-cols-3 gap-6"
          >
            {contactMethods.map((method) => (
              <motion.div
                key={method.title}
                variants={fadeUp}
                className="p-6 bg-card border border-border/60 rounded-lg text-center hover:border-border transition-colors"
              >
                <div className="inline-flex items-center justify-center h-12 w-12 rounded-full bg-primary/10 text-primary mb-4">
                  <method.icon className="h-5 w-5" />
                </div>
                <h3 className="font-semibold text-foreground mb-1">{method.title}</h3>
                <p className="text-sm text-muted-foreground mb-3">{method.description}</p>
                <p className="text-sm font-medium text-primary">{method.value}</p>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* Contact Form */}
      <section className="py-16 lg:py-24 bg-surface-1">
        <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            variants={stagger}
          >
            <motion.div variants={fadeUp} className="text-center mb-12">
              <h2 className="text-2xl sm:text-3xl font-bold text-foreground">
                Send us a message
              </h2>
              <p className="mt-3 text-muted-foreground">
                Fill out the form below and we&apos;ll get back to you as soon as possible.
              </p>
            </motion.div>

            {submitted ? (
              <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                className="bg-card border border-border/60 rounded-lg p-12 text-center"
              >
                <div className="inline-flex items-center justify-center h-16 w-16 rounded-full bg-positive/10 text-positive mb-6">
                  <CheckCircle2 className="h-8 w-8" />
                </div>
                <h3 className="text-xl font-semibold text-foreground mb-2">Message sent!</h3>
                <p className="text-muted-foreground">
                  Thank you for reaching out. We&apos;ll get back to you within 24 hours.
                </p>
              </motion.div>
            ) : (
              <motion.form
                variants={fadeUp}
                onSubmit={handleSubmit}
                className="bg-card border border-border/60 rounded-lg p-6 lg:p-8 space-y-6"
              >
                <div className="grid sm:grid-cols-2 gap-6">
                  <div className="space-y-2">
                    <Label htmlFor="firstName">First name</Label>
                    <Input
                      id="firstName"
                      placeholder="John"
                      className="h-11 bg-surface-1"
                      required
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="lastName">Last name</Label>
                    <Input
                      id="lastName"
                      placeholder="Doe"
                      className="h-11 bg-surface-1"
                      required
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="email">Email</Label>
                  <Input
                    id="email"
                    type="email"
                    placeholder="john@example.com"
                    className="h-11 bg-surface-1"
                    required
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="subject">Subject</Label>
                  <Select>
                    <SelectTrigger className="h-11 bg-surface-1">
                      <SelectValue placeholder="Select a topic" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="general">General inquiry</SelectItem>
                      <SelectItem value="support">Technical support</SelectItem>
                      <SelectItem value="sales">Sales question</SelectItem>
                      <SelectItem value="enterprise">Enterprise pricing</SelectItem>
                      <SelectItem value="partnership">Partnership</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="message">Message</Label>
                  <Textarea
                    id="message"
                    placeholder="How can we help you?"
                    className="min-h-[150px] bg-surface-1 resize-none"
                    required
                  />
                </div>

                <Button type="submit" size="lg" className="w-full h-12">
                  Send message
                </Button>
              </motion.form>
            )}
          </motion.div>
        </div>
      </section>

      {/* Office Info */}
      <section className="py-16 lg:py-24">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            variants={stagger}
            className="grid md:grid-cols-2 gap-12 items-center"
          >
            <motion.div variants={fadeUp}>
              <h2 className="text-2xl sm:text-3xl font-bold text-foreground mb-6">
                Our office
              </h2>
              <div className="space-y-4">
                <div className="flex items-start gap-3">
                  <MapPin className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                  <div>
                    <p className="font-medium text-foreground">Address</p>
                    <p className="text-muted-foreground">
                      123 Financial District<br />
                      New York, NY 10004<br />
                      United States
                    </p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <Clock className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                  <div>
                    <p className="font-medium text-foreground">Hours</p>
                    <p className="text-muted-foreground">
                      Monday - Friday: 9:00 AM - 6:00 PM EST<br />
                      Saturday - Sunday: Closed
                    </p>
                  </div>
                </div>
              </div>
            </motion.div>

            <motion.div
              variants={fadeUp}
              className="h-64 md:h-80 bg-surface-2 rounded-lg border border-border/60 flex items-center justify-center"
            >
              <div className="text-center text-muted-foreground">
                <MapPin className="h-8 w-8 mx-auto mb-2 opacity-50" />
                <p className="text-sm">Map placeholder</p>
              </div>
            </motion.div>
          </motion.div>
        </div>
      </section>
    </div>
  )
}
